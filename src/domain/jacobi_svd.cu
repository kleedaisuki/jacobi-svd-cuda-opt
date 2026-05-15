#include "jacobi/svd/domain/jacobi_svd.cuh"

#include "jacobi/svd/domain/device_matrix.cuh"
#include "src/domain/cuda_check.cuh"
#include "src/domain/device_buffer.cuh"
#include "src/domain/jacobi_rotation_kernels.cuh"
#include "src/domain/jacobi_schedule.cuh"
#include "src/domain/jacobi_svd_detail.cuh"
#include "src/domain/layout_transpose_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace jacobi::svd::detail
{
    /**
     * @brief 计算矩阵元素数量；Compute matrix element count.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @return 元素数量；Element count.
     */
    [[nodiscard]] std::size_t matrix_element_count(std::size_t rows, std::size_t columns)
    {
        return rows * columns;
    }

    /**
     * @brief 校验布局转置配置；Validate layout-transpose related configuration.
     * @param config 算法配置；Algorithm configuration.
     */
    void validate_layout_transpose_config(const JacobiSvdConfig &config)
    {
        if (config.layout_transpose_min_columns <= 0)
        {
            throw std::invalid_argument("layout_transpose_min_columns must be positive.");
        }
        if (config.layout_transpose_min_elements == 0)
        {
            throw std::invalid_argument("layout_transpose_min_elements must be positive.");
        }
        if (config.layout_transpose_benchmark_repetitions <= 0)
        {
            throw std::invalid_argument("layout_transpose_benchmark_repetitions must be positive.");
        }
        if (config.layout_transpose_benchmark_sweeps <= 0)
        {
            throw std::invalid_argument("layout_transpose_benchmark_sweeps must be positive.");
        }
    }

    /**
     * @brief 判断是否启用布局转置路径；Decide whether to enable layout-transpose path.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @param config 算法配置；Algorithm configuration.
     * @return true 表示启用布局转置；true if layout transpose should be enabled.
     */
    [[nodiscard]] bool should_use_layout_transpose(int rows, int columns, const JacobiSvdConfig &config)
    {
        if (config.layout_transpose_mode == LayoutTransposeMode::force_enable)
        {
            return true;
        }
        if (config.layout_transpose_mode == LayoutTransposeMode::force_disable)
        {
            return false;
        }

        const std::size_t element_count =
            matrix_element_count(static_cast<std::size_t>(rows), static_cast<std::size_t>(columns));
        return columns >= config.layout_transpose_min_columns &&
               element_count >= config.layout_transpose_min_elements;
    }

    /**
     * @brief 计算 cooperative sweep kernel 的 grid 配置；Compute the grid configuration for the cooperative sweep kernel.
     * @param n 列数；Column count.
     * @param threads 每个 block 的线程数；Threads per block.
     * @param shared_bytes 每个 block 的动态共享内存字节数；Dynamic shared-memory bytes per block.
     * @param grid_blocks 输出 grid block 数；Output grid block count.
     * @return true 表示当前设备可执行 cooperative sweep；true if the current device can run the cooperative sweep.
     */
    template <bool ColumnMajorA>
    [[nodiscard]] bool cooperative_sweep_grid_blocks(int n, int threads, std::size_t shared_bytes, int &grid_blocks)
    {
        if (n < 2)
        {
            return false;
        }

        int device = 0;
        JACOBI_CUDA_CHECK(cudaGetDevice(&device));

        int cooperative_launch = 0;
        JACOBI_CUDA_CHECK(cudaDeviceGetAttribute(&cooperative_launch, cudaDevAttrCooperativeLaunch, device));
        if (cooperative_launch == 0)
        {
            return false;
        }

        int sm_count = 0;
        JACOBI_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

        int active_blocks_per_sm = 0;
        JACOBI_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm,
                                                                        jacobi_sweep_kernel<ColumnMajorA>,
                                                                        threads,
                                                                        shared_bytes));

        const int even_n = n + (n & 1);
        const int pair_slots = even_n / 2;
        const int resident_blocks = active_blocks_per_sm * sm_count;
        if (resident_blocks <= 0 || pair_slots <= 0)
        {
            return false;
        }

        grid_blocks = std::min(resident_blocks, pair_slots);
        return grid_blocks > 0;
    }

    /**
     * @brief 尝试用 cooperative kernel 执行所有 sweep；Try to execute all sweeps with a cooperative kernel.
     * @param a 输入输出矩阵 A；Input/output matrix A.
     * @param v 输入输出矩阵 V；Input/output matrix V.
     * @param m A 的行数；Row count of A.
     * @param n A 的列数，同时也是 V 的维度；Column count of A and dimension of V.
     * @param config 算法配置；Algorithm configuration.
     * @param d_any_rotation sweep 级收敛标志；Sweep-level convergence flag.
     * @param executed_sweeps 输出已执行 sweep 数；Output executed sweep count.
     * @return true 表示已使用 cooperative 路径完成；true if the cooperative path completed execution.
     */
    template <bool ColumnMajorA>
    [[nodiscard]] bool try_run_cooperative_sweeps(double *a,
                                                  double *v,
                                                  int m,
                                                  int n,
                                                  const JacobiSvdConfig &config,
                                                  DeviceBuffer<int> &d_any_rotation,
                                                  int &executed_sweeps)
    {
        const int threads = normalize_threads_per_block(config.threads_per_block);
        const std::size_t shared_bytes = static_cast<std::size_t>(threads) * 3 * sizeof(double);

        int grid_blocks = 0;
        if (!cooperative_sweep_grid_blocks<ColumnMajorA>(n, threads, shared_bytes, grid_blocks))
        {
            return false;
        }

        for (int sweep = 0; sweep < config.max_sweeps; ++sweep)
        {
            JACOBI_CUDA_CHECK(cudaMemset(d_any_rotation.data(), 0, sizeof(int)));

            double epsilon = config.epsilon;
            int *any_rotation_flag = d_any_rotation.data();
            void *kernel_args[] = {
                &a,
                &v,
                &m,
                &n,
                &epsilon,
                &any_rotation_flag,
            };
            JACOBI_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(jacobi_sweep_kernel<ColumnMajorA>),
                                                          dim3(grid_blocks),
                                                          dim3(threads),
                                                          kernel_args,
                                                          shared_bytes,
                                                          nullptr));

            int any_rotation = 0;
            JACOBI_CUDA_CHECK(cudaMemcpy(&any_rotation, d_any_rotation.data(), sizeof(int), cudaMemcpyDeviceToHost));

            executed_sweeps = sweep + 1;
            if (any_rotation == 0)
            {
                break;
            }
        }

        return true;
    }

    /**
     * @brief 校验单边雅可比输入参数；Validate one-sided Jacobi input arguments.
     * @param host_input 输入缓存；Input buffer.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @param config 算法配置；Algorithm configuration.
     */
    void validate_one_sided_inputs(std::span<const double> host_input,
                                   std::size_t rows,
                                   std::size_t columns,
                                   const JacobiSvdConfig &config)
    {
        if (rows == 0 || columns == 0)
        {
            throw std::invalid_argument("Input matrix shape must be non-zero.");
        }
        if (rows < columns)
        {
            throw std::invalid_argument("One-sided Jacobi SVD requires rows >= columns in this implementation.");
        }
        if (host_input.size() != matrix_element_count(rows, columns))
        {
            throw std::invalid_argument("Input buffer size mismatch.");
        }
        if (config.epsilon <= 0.0)
        {
            throw std::invalid_argument("epsilon must be positive.");
        }
        if (config.max_sweeps <= 0)
        {
            throw std::invalid_argument("max_sweeps must be positive.");
        }
        validate_layout_transpose_config(config);
    }

    /**
     * @brief 内部执行单边雅可比 SVD；Internal executor for one-sided Jacobi SVD.
     * @param host_input 输入矩阵（行主序）；Input matrix (row-major).
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @param config 算法配置；Algorithm configuration.
     * @param use_layout_transpose 是否启用布局转置路径；Whether to use layout-transpose path.
     * @return SVD 结果；SVD result.
     */
    [[nodiscard]] JacobiSvdResult run_one_sided_jacobi_svd_internal(std::span<const double> host_input,
                                                                    std::size_t rows,
                                                                    std::size_t columns,
                                                                    const JacobiSvdConfig &config,
                                                                    bool use_layout_transpose)
    {
        const int m = static_cast<int>(rows);
        const int n = static_cast<int>(columns);
        const int threads = normalize_threads_per_block(config.threads_per_block);

        DeviceMatrix d_a(rows, columns);
        DeviceMatrix d_a_layout;
        DeviceMatrix d_v(columns, columns);
        DeviceMatrix d_u(rows, columns);
        DeviceMatrix d_sigma(1, columns);

        d_a.copy_from_host(host_input);
        if (use_layout_transpose)
        {
            d_a_layout.reset(rows, columns);
            launch_row_to_column_layout_transpose(d_a.data(), d_a_layout.data(), m, n);
        }

        const int identity_total = n * n;
        const int identity_blocks = (identity_total + threads - 1) / threads;
        initialize_identity_kernel<<<identity_blocks, threads>>>(d_v.data(), n);
        JACOBI_CUDA_CHECK(cudaGetLastError());

        DeviceBuffer<int> d_any_rotation(1);

        int executed_sweeps = 0;
        const bool ran_cooperative =
            use_layout_transpose
                ? try_run_cooperative_sweeps<true>(
                      d_a_layout.data(), d_v.data(), m, n, config, d_any_rotation, executed_sweeps)
                : try_run_cooperative_sweeps<false>(
                      d_a.data(), d_v.data(), m, n, config, d_any_rotation, executed_sweeps);

        if (!ran_cooperative)
        {
            const auto rounds = build_round_robin_schedule(n);
            std::size_t max_pairs = 0;
            for (const auto &round : rounds)
            {
                max_pairs = std::max(max_pairs, round.size());
            }

            DeviceBuffer<int2> d_pairs(max_pairs);
            DeviceBuffer<double> d_app(max_pairs);
            DeviceBuffer<double> d_aqq(max_pairs);
            DeviceBuffer<double> d_apq(max_pairs);

            for (int sweep = 0; sweep < config.max_sweeps; ++sweep)
            {
                bool converged_this_sweep = true;

                for (const auto &round : rounds)
                {
                    const int pair_count = static_cast<int>(round.size());
                    if (pair_count == 0)
                    {
                        continue;
                    }

                    JACOBI_CUDA_CHECK(cudaMemcpy(d_pairs.data(),
                                                 round.data(),
                                                 static_cast<std::size_t>(pair_count) * sizeof(int2),
                                                 cudaMemcpyHostToDevice));
                    JACOBI_CUDA_CHECK(cudaMemset(d_any_rotation.data(), 0, sizeof(int)));

                    const std::size_t shared_bytes_stats = static_cast<std::size_t>(threads) * 3 * sizeof(double);
                    if (use_layout_transpose)
                    {
                        pair_stats_kernel<true><<<pair_count, threads, shared_bytes_stats>>>(d_a_layout.data(),
                                                                                             m,
                                                                                             n,
                                                                                             d_pairs.data(),
                                                                                             pair_count,
                                                                                             d_app.data(),
                                                                                             d_aqq.data(),
                                                                                             d_apq.data());
                    }
                    else
                    {
                        pair_stats_kernel<false><<<pair_count, threads, shared_bytes_stats>>>(d_a.data(),
                                                                                              m,
                                                                                              n,
                                                                                              d_pairs.data(),
                                                                                              pair_count,
                                                                                              d_app.data(),
                                                                                              d_aqq.data(),
                                                                                              d_apq.data());
                    }
                    JACOBI_CUDA_CHECK(cudaGetLastError());

                    if (use_layout_transpose)
                    {
                        compute_and_apply_rotation_kernel<true><<<pair_count, threads>>>(d_a_layout.data(),
                                                                                         d_v.data(),
                                                                                         m,
                                                                                         n,
                                                                                         d_pairs.data(),
                                                                                         pair_count,
                                                                                         d_app.data(),
                                                                                         d_aqq.data(),
                                                                                         d_apq.data(),
                                                                                         config.epsilon,
                                                                                         d_any_rotation.data());
                    }
                    else
                    {
                        compute_and_apply_rotation_kernel<false><<<pair_count, threads>>>(d_a.data(),
                                                                                          d_v.data(),
                                                                                          m,
                                                                                          n,
                                                                                          d_pairs.data(),
                                                                                          pair_count,
                                                                                          d_app.data(),
                                                                                          d_aqq.data(),
                                                                                          d_apq.data(),
                                                                                          config.epsilon,
                                                                                          d_any_rotation.data());
                    }
                    JACOBI_CUDA_CHECK(cudaGetLastError());

                    int any_rotation = 0;
                    JACOBI_CUDA_CHECK(
                        cudaMemcpy(&any_rotation, d_any_rotation.data(), sizeof(int), cudaMemcpyDeviceToHost));
                    if (any_rotation != 0)
                    {
                        converged_this_sweep = false;
                    }
                }

                executed_sweeps = sweep + 1;
                if (converged_this_sweep)
                {
                    break;
                }
            }
        }

        const double *u_sigma_input = d_a.data();
        if (use_layout_transpose)
        {
            launch_column_to_row_layout_transpose(d_a_layout.data(), d_a.data(), m, n);
            u_sigma_input = d_a.data();
        }

        const std::size_t shared_bytes_norm = static_cast<std::size_t>(threads) * sizeof(double);
        build_u_sigma_kernel<<<n, threads, shared_bytes_norm>>>(
            u_sigma_input, d_u.data(), d_sigma.data(), m, n, config.epsilon);
        JACOBI_CUDA_CHECK(cudaGetLastError());
        JACOBI_CUDA_CHECK(cudaDeviceSynchronize());

        JacobiSvdResult result;
        result.rows = rows;
        result.columns = columns;
        result.sweeps = executed_sweeps;
        result.u = d_u.copy_to_host();
        result.sigma = d_sigma.copy_to_host();
        result.v = d_v.copy_to_host();
        return result;
    }
} // namespace jacobi::svd::detail

namespace jacobi::svd
{
    JacobiSvdResult one_sided_jacobi_svd(std::span<const double> host_input,
                                         std::size_t rows,
                                         std::size_t columns,
                                         const JacobiSvdConfig &config)
    {
        detail::validate_one_sided_inputs(host_input, rows, columns, config);

        const bool use_layout_transpose =
            detail::should_use_layout_transpose(static_cast<int>(rows), static_cast<int>(columns), config);
        return detail::run_one_sided_jacobi_svd_internal(host_input, rows, columns, config, use_layout_transpose);
    }
} // namespace jacobi::svd

#undef JACOBI_CUDA_CHECK
