#include "jacobi/svd/domain/jacobi_svd.cuh"

#include "src/domain/jacobi_svd_detail.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace jacobi::svd::detail
{
    /**
     * @brief 构造基准输入矩阵；Build benchmark input matrix.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @return 行主序矩阵数据；Row-major matrix data.
     */
    [[nodiscard]] std::vector<double> build_benchmark_matrix(std::size_t rows, std::size_t columns)
    {
        std::vector<double> values(matrix_element_count(rows, columns), 0.0);
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t col = 0; col < columns; ++col)
            {
                const double lhs = std::sin(0.013 * static_cast<double>((row + 1) * (col + 1)));
                const double rhs = std::cos(0.017 * static_cast<double>((row + 3) * (col + 5)));
                values[row * columns + col] = lhs + rhs;
            }
        }
        return values;
    }

    /**
     * @brief 评估单一路径平均耗时；Measure average latency of one path.
     * @param host_input 输入矩阵；Input matrix.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @param benchmark_config 基准配置；Benchmark configuration.
     * @param use_layout_transpose 是否启用布局转置；Whether to use layout transpose.
     * @return 平均耗时（毫秒）；Average latency in milliseconds.
     */
    [[nodiscard]] double benchmark_path_average_milliseconds(std::span<const double> host_input,
                                                             std::size_t rows,
                                                             std::size_t columns,
                                                             const JacobiSvdConfig &benchmark_config,
                                                             bool use_layout_transpose)
    {
        (void)run_one_sided_jacobi_svd_internal(host_input, rows, columns, benchmark_config, use_layout_transpose);

        double accumulated_ms = 0.0;
        int sweep_checksum = 0;
        for (int repetition = 0; repetition < benchmark_config.layout_transpose_benchmark_repetitions; ++repetition)
        {
            const auto started_at = std::chrono::steady_clock::now();
            const JacobiSvdResult result =
                run_one_sided_jacobi_svd_internal(host_input, rows, columns, benchmark_config, use_layout_transpose);
            const auto finished_at = std::chrono::steady_clock::now();
            sweep_checksum += result.sweeps;
            accumulated_ms +=
                std::chrono::duration<double, std::milli>(finished_at - started_at).count();
        }

        if (sweep_checksum < 0)
        {
            throw std::runtime_error("Unreachable sweep checksum guard triggered.");
        }

        return accumulated_ms / static_cast<double>(benchmark_config.layout_transpose_benchmark_repetitions);
    }
} // namespace jacobi::svd::detail

namespace jacobi::svd
{
    LayoutTransposeAutoTuneReport auto_tune_layout_transpose_threshold(const JacobiSvdConfig &config)
    {
        detail::validate_layout_transpose_config(config);
        if (config.max_sweeps <= 0)
        {
            throw std::invalid_argument("max_sweeps must be positive.");
        }

        JacobiSvdConfig benchmark_config = config;
        benchmark_config.layout_transpose_auto_tune = false;
        benchmark_config.layout_transpose_mode = LayoutTransposeMode::auto_select;
        benchmark_config.max_sweeps =
            std::min(config.max_sweeps, std::max(1, config.layout_transpose_benchmark_sweeps));

        constexpr std::array<int, 8> benchmark_columns = {8, 12, 16, 24, 32, 48, 64, 96};

        LayoutTransposeAutoTuneReport report{};
        report.executed = true;
        report.recommended_min_columns = config.layout_transpose_min_columns;
        report.recommended_min_elements = config.layout_transpose_min_elements;
        report.estimated_best_speedup = 1.0;
        report.sample_count = benchmark_columns.size();

        bool threshold_selected = false;

        for (const int columns : benchmark_columns)
        {
            const std::size_t rows = static_cast<std::size_t>(columns * 2);
            const std::size_t cols = static_cast<std::size_t>(columns);
            const std::vector<double> host_input = detail::build_benchmark_matrix(rows, cols);

            const double direct_ms =
                detail::benchmark_path_average_milliseconds(host_input, rows, cols, benchmark_config, false);
            const double transpose_ms =
                detail::benchmark_path_average_milliseconds(host_input, rows, cols, benchmark_config, true);
            const double safe_transpose = std::max(transpose_ms, std::numeric_limits<double>::min());
            const double speedup = direct_ms / safe_transpose;
            report.estimated_best_speedup = std::max(report.estimated_best_speedup, speedup);

            if (!threshold_selected && transpose_ms < direct_ms)
            {
                report.recommended_min_columns = columns;
                report.recommended_min_elements = detail::matrix_element_count(rows, cols);
                threshold_selected = true;
            }
        }

        return report;
    }
} // namespace jacobi::svd
