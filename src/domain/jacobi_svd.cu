#include "jacobi/svd/domain/jacobi_svd.hpp"

#include "jacobi/svd/domain/device_matrix.hpp"
#include "src/domain/cuda_check.cuh"
#include "src/domain/device_buffer.cuh"
#include "src/domain/jacobi_rotation_kernels.cuh"
#include "src/domain/jacobi_schedule.cuh"
#include "src/domain/jacobi_svd_detail.hpp"
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
        DeviceBuffer<double> d_c(max_pairs);
        DeviceBuffer<double> d_s(max_pairs);
        DeviceBuffer<int> d_any_rotation(1);

        int executed_sweeps = 0;

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

                const int rotation_blocks = (pair_count + threads - 1) / threads;
                compute_rotation_params_kernel<<<rotation_blocks, threads>>>(d_app.data(),
                                                                             d_aqq.data(),
                                                                             d_apq.data(),
                                                                             pair_count,
                                                                             config.epsilon,
                                                                             d_c.data(),
                                                                             d_s.data(),
                                                                             d_any_rotation.data());
                JACOBI_CUDA_CHECK(cudaGetLastError());

                if (use_layout_transpose)
                {
                    apply_rotation_kernel<true><<<pair_count, threads>>>(d_a_layout.data(),
                                                                         d_v.data(),
                                                                         m,
                                                                         n,
                                                                         d_pairs.data(),
                                                                         pair_count,
                                                                         d_c.data(),
                                                                         d_s.data());
                }
                else
                {
                    apply_rotation_kernel<false><<<pair_count, threads>>>(d_a.data(),
                                                                          d_v.data(),
                                                                          m,
                                                                          n,
                                                                          d_pairs.data(),
                                                                          pair_count,
                                                                          d_c.data(),
                                                                          d_s.data());
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
