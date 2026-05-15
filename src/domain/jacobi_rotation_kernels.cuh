#pragma once

#include "src/domain/matrix_index.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>

namespace jacobi::svd::detail
{
    /**
     * @brief 计算非负模；Compute non-negative modulo.
     * @param value 输入值；Input value.
     * @param modulus 模数；Modulus.
     * @return 非负模结果；Non-negative modulo result.
     */
    __device__ inline int positive_mod(int value, int modulus)
    {
        const int result = value % modulus;
        return (result < 0) ? (result + modulus) : result;
    }

    /**
     * @brief 计算轮转调度中指定位置的列索引；Compute the column index at a round-robin schedule position.
     * @param position 轮转数组位置；Position in the rotated circle.
     * @param round 轮次编号；Round index.
     * @param n 真实列数；Real column count.
     * @param even_n 含 dummy 的偶数列数；Even column count including dummy.
     * @return 列索引，-1 表示 dummy；Column index, or -1 for dummy.
     */
    __device__ inline int round_robin_column_at(int position, int round, int n, int even_n)
    {
        if (position == 0)
        {
            return 0;
        }

        const int ring = even_n - 1;
        const int initial_index = 1 + positive_mod(position - 1 - round, ring);
        return (initial_index < n) ? initial_index : -1;
    }

    /**
     * @brief 初始化单位矩阵 V；Initialize identity matrix V.
     * @param v 输出矩阵指针；Output matrix pointer.
     * @param n 方阵维度；Square dimension.
     */
    __global__ void initialize_identity_kernel(double *v, int n)
    {
        const int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        const int total = n * n;
        if (linear >= total)
        {
            return;
        }

        const int row = linear / n;
        const int col = linear % n;
        v[linear] = (row == col) ? 1.0 : 0.0;
    }

    /**
     * @brief 计算列对统计量；Compute pair statistics for column pairs.
     * @param a 输入矩阵 A；Input matrix A.
     * @param m 行数；Row count.
     * @param n 列数；Column count.
     * @param pairs 列对数组；Column-pair array.
     * @param pair_count 列对数量；Number of pairs.
     * @param a_pp 输出 A_p 点积；Output A_p dot A_p.
     * @param a_qq 输出 A_q 点积；Output A_q dot A_q.
     * @param a_pq 输出 A_p 与 A_q 点积；Output A_p dot A_q.
     */
    template <bool ColumnMajorA>
    __global__ void pair_stats_kernel(const double *a,
                                      int m,
                                      int n,
                                      const int2 *pairs,
                                      int pair_count,
                                      double *a_pp,
                                      double *a_qq,
                                      double *a_pq)
    {
        const int pair_index = static_cast<int>(blockIdx.x);
        if (pair_index >= pair_count)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        const int p = pairs[pair_index].x;
        const int q = pairs[pair_index].y;

        double local_pp = 0.0;
        double local_qq = 0.0;
        double local_pq = 0.0;

        for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
        {
            const double ap = a[matrix_index<ColumnMajorA>(row, p, m, n)];
            const double aq = a[matrix_index<ColumnMajorA>(row, q, m, n)];
            local_pp += ap * ap;
            local_qq += aq * aq;
            local_pq += ap * aq;
        }

        extern __shared__ double shared[];
        double *shared_pp = shared;
        double *shared_qq = shared + blockDim.x;
        double *shared_pq = shared + (2 * blockDim.x);

        shared_pp[tid] = local_pp;
        shared_qq[tid] = local_qq;
        shared_pq[tid] = local_pq;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U)
        {
            if (tid < static_cast<int>(stride))
            {
                shared_pp[tid] += shared_pp[tid + stride];
                shared_qq[tid] += shared_qq[tid + stride];
                shared_pq[tid] += shared_pq[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            a_pp[pair_index] = shared_pp[0];
            a_qq[pair_index] = shared_qq[0];
            a_pq[pair_index] = shared_pq[0];
        }
    }

    /**
     * @brief 计算旋转参数并立即应用到 A 与 V；Compute rotation parameters and immediately apply them to A and V.
     * @param a 输入输出矩阵 A；Input/output matrix A.
     * @param v 输入输出矩阵 V；Input/output matrix V.
     * @param m A 的行数；Row count of A.
     * @param n A 的列数，同时也是 V 的维度；Column count of A and dimension of V.
     * @param pairs 列对数组；Column-pair array.
     * @param pair_count 列对数量；Number of pairs.
     * @param a_pp A_p 点积数组；A_p dot A_p array.
     * @param a_qq A_q 点积数组；A_q dot A_q array.
     * @param a_pq A_p 与 A_q 点积数组；A_p dot A_q array.
     * @param epsilon 收敛阈值；Convergence threshold.
     * @param any_rotation_flag 若发生旋转则置 1；Set to 1 if any rotation happens.
     * @note 每个 block 处理一个列对，先在 block 内广播旋转参数，再并行更新对应列；
     *       Each block handles one column pair, broadcasts rotation parameters inside the block, then updates the pair in parallel.
     */
    template <bool ColumnMajorA>
    __global__ void compute_and_apply_rotation_kernel(double *a,
                                                      double *v,
                                                      int m,
                                                      int n,
                                                      const int2 *pairs,
                                                      int pair_count,
                                                      const double *a_pp,
                                                      const double *a_qq,
                                                      const double *a_pq,
                                                      double epsilon,
                                                      int *any_rotation_flag)
    {
        const int pair_index = static_cast<int>(blockIdx.x);
        if (pair_index >= pair_count)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        __shared__ double shared_cosine;
        __shared__ double shared_sine;

        if (tid == 0)
        {
            const double pp = a_pp[pair_index];
            const double qq = a_qq[pair_index];
            const double pq = a_pq[pair_index];
            const double rhs = epsilon * sqrt(fmax(pp, 0.0) * fmax(qq, 0.0));

            if (fabs(pq) > rhs && fabs(pq) > 1.0e-300)
            {
                const double tau = (qq - pp) / (2.0 * pq);
                const double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                              : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                shared_cosine = 1.0 / sqrt(1.0 + t * t);
                shared_sine = t * shared_cosine;
                atomicExch(any_rotation_flag, 1);
            }
            else
            {
                shared_cosine = 1.0;
                shared_sine = 0.0;
            }
        }
        __syncthreads();

        if (shared_sine == 0.0)
        {
            return;
        }

        const int p = pairs[pair_index].x;
        const int q = pairs[pair_index].y;
        const double cosine = shared_cosine;
        const double sine = shared_sine;

        for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
        {
            const int idx_p = matrix_index<ColumnMajorA>(row, p, m, n);
            const int idx_q = matrix_index<ColumnMajorA>(row, q, m, n);
            const double value_p = a[idx_p];
            const double value_q = a[idx_q];
            a[idx_p] = cosine * value_p - sine * value_q;
            a[idx_q] = sine * value_p + cosine * value_q;
        }

        for (int row = tid; row < n; row += static_cast<int>(blockDim.x))
        {
            const int idx_p = row_major_index(row, p, n);
            const int idx_q = row_major_index(row, q, n);
            const double value_p = v[idx_p];
            const double value_q = v[idx_q];
            v[idx_p] = cosine * value_p - sine * value_q;
            v[idx_q] = sine * value_p + cosine * value_q;
        }
    }

    /**
     * @brief 执行一个完整 Jacobi sweep；Execute one full Jacobi sweep.
     * @param a 输入输出矩阵 A；Input/output matrix A.
     * @param v 输入输出矩阵 V；Input/output matrix V.
     * @param m A 的行数；Row count of A.
     * @param n A 的列数，同时也是 V 的维度；Column count of A and dimension of V.
     * @param epsilon 收敛阈值；Convergence threshold.
     * @param any_rotation_flag 若发生旋转则置 1；Set to 1 if any rotation happens.
     * @note 该 cooperative kernel 在 device 侧重算 round-robin schedule，并在每个 round 后执行 grid 同步；
     *       This cooperative kernel regenerates the round-robin schedule on device and performs a grid sync after each round.
     */
    template <bool ColumnMajorA>
    __global__ void jacobi_sweep_kernel(double *a, double *v, int m, int n, double epsilon, int *any_rotation_flag)
    {
        namespace cg = cooperative_groups;
        cg::grid_group grid = cg::this_grid();

        if (n < 2)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        const int even_n = n + (n & 1);
        const int round_count = even_n - 1;
        const int pair_slots = even_n / 2;

        extern __shared__ double shared[];
        double *shared_pp = shared;
        double *shared_qq = shared + blockDim.x;
        double *shared_pq = shared + (2 * blockDim.x);
        __shared__ double shared_cosine;
        __shared__ double shared_sine;

        for (int round = 0; round < round_count; ++round)
        {
            for (int pair_slot = static_cast<int>(blockIdx.x); pair_slot < pair_slots;
                 pair_slot += static_cast<int>(gridDim.x))
            {
                const int p = round_robin_column_at(pair_slot, round, n, even_n);
                const int q = round_robin_column_at(even_n - 1 - pair_slot, round, n, even_n);
                if (p < 0 || q < 0)
                {
                    continue;
                }

                double local_pp = 0.0;
                double local_qq = 0.0;
                double local_pq = 0.0;

                for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
                {
                    const double ap = a[matrix_index<ColumnMajorA>(row, p, m, n)];
                    const double aq = a[matrix_index<ColumnMajorA>(row, q, m, n)];
                    local_pp += ap * ap;
                    local_qq += aq * aq;
                    local_pq += ap * aq;
                }

                shared_pp[tid] = local_pp;
                shared_qq[tid] = local_qq;
                shared_pq[tid] = local_pq;
                __syncthreads();

                for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U)
                {
                    if (tid < static_cast<int>(stride))
                    {
                        shared_pp[tid] += shared_pp[tid + stride];
                        shared_qq[tid] += shared_qq[tid + stride];
                        shared_pq[tid] += shared_pq[tid + stride];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    const double pp = shared_pp[0];
                    const double qq = shared_qq[0];
                    const double pq = shared_pq[0];
                    const double rhs = epsilon * sqrt(fmax(pp, 0.0) * fmax(qq, 0.0));

                    if (fabs(pq) > rhs && fabs(pq) > 1.0e-300)
                    {
                        const double tau = (qq - pp) / (2.0 * pq);
                        const double t = (tau >= 0.0) ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                                                      : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
                        shared_cosine = 1.0 / sqrt(1.0 + t * t);
                        shared_sine = t * shared_cosine;
                        atomicExch(any_rotation_flag, 1);
                    }
                    else
                    {
                        shared_cosine = 1.0;
                        shared_sine = 0.0;
                    }
                }
                __syncthreads();

                if (shared_sine != 0.0)
                {
                    const double cosine = shared_cosine;
                    const double sine = shared_sine;

                    for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
                    {
                        const int idx_p = matrix_index<ColumnMajorA>(row, p, m, n);
                        const int idx_q = matrix_index<ColumnMajorA>(row, q, m, n);
                        const double value_p = a[idx_p];
                        const double value_q = a[idx_q];
                        a[idx_p] = cosine * value_p - sine * value_q;
                        a[idx_q] = sine * value_p + cosine * value_q;
                    }

                    for (int row = tid; row < n; row += static_cast<int>(blockDim.x))
                    {
                        const int idx_p = row_major_index(row, p, n);
                        const int idx_q = row_major_index(row, q, n);
                        const double value_p = v[idx_p];
                        const double value_q = v[idx_q];
                        v[idx_p] = cosine * value_p - sine * value_q;
                        v[idx_q] = sine * value_p + cosine * value_q;
                    }
                }
                __syncthreads();
            }

            grid.sync();
        }
    }

    /**
     * @brief 从收敛后的 A 构建 U 与 Sigma；Build U and Sigma from converged A.
     * @param a 输入矩阵 A（应为 U*Sigma）；Input matrix A (should be U*Sigma).
     * @param u 输出矩阵 U；Output matrix U.
     * @param sigma 输出奇异值数组；Output singular values.
     * @param m 行数；Row count.
     * @param n 列数；Column count.
     * @param epsilon 避免除零的阈值；Threshold to avoid divide-by-zero.
     */
    __global__ void build_u_sigma_kernel(const double *a, double *u, double *sigma, int m, int n, double epsilon)
    {
        const int col = static_cast<int>(blockIdx.x);
        if (col >= n)
        {
            return;
        }

        const int tid = static_cast<int>(threadIdx.x);
        double local_norm = 0.0;

        for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
        {
            const double value = a[row_major_index(row, col, n)];
            local_norm += value * value;
        }

        extern __shared__ double shared_norm[];
        shared_norm[tid] = local_norm;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1U)
        {
            if (tid < static_cast<int>(stride))
            {
                shared_norm[tid] += shared_norm[tid + stride];
            }
            __syncthreads();
        }

        __shared__ double sigma_col;
        if (tid == 0)
        {
            sigma_col = sqrt(fmax(shared_norm[0], 0.0));
            sigma[col] = sigma_col;
        }
        __syncthreads();

        for (int row = tid; row < m; row += static_cast<int>(blockDim.x))
        {
            const int index = row_major_index(row, col, n);
            u[index] = (sigma_col > epsilon) ? (a[index] / sigma_col) : 0.0;
        }
    }
} // namespace jacobi::svd::detail
