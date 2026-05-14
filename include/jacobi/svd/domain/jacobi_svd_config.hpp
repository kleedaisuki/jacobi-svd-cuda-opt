#pragma once

#include "jacobi/svd/domain/layout_transpose.hpp"

#include <cstddef>

namespace jacobi::svd
{
    /**
     * @brief 单边雅可比 SVD 参数；Configuration for one-sided Jacobi SVD.
     */
    struct JacobiSvdConfig final
    {
        /**
         * @brief 相对收敛阈值 epsilon；Relative convergence tolerance epsilon.
         */
        double epsilon = 1.0e-9;

        /**
         * @brief 最大 sweep 次数；Maximum number of sweeps.
         */
        int max_sweeps = 128;

        /**
         * @brief 每个 CUDA block 的线程数；Threads per CUDA block.
         */
        int threads_per_block = 256;

        /**
         * @brief 布局转置策略；Layout-transpose policy.
         */
        LayoutTransposeMode layout_transpose_mode = LayoutTransposeMode::auto_select;

        /**
         * @brief 自动策略下最小列数阈值；Minimum column threshold when mode is auto.
         */
        int layout_transpose_min_columns = 16;

        /**
         * @brief 自动策略下最小元素数阈值；Minimum element threshold when mode is auto.
         */
        std::size_t layout_transpose_min_elements = 4096;

        /**
         * @brief 是否在运行前执行阈值微基准自动调优；Whether to run micro-benchmark auto-tuning before execution.
         */
        bool layout_transpose_auto_tune = false;

        /**
         * @brief 自动调优时每个尺寸的重复次数；Repetition count per size during auto-tuning.
         */
        int layout_transpose_benchmark_repetitions = 2;

        /**
         * @brief 自动调优时的基准 sweep 上限；Benchmark sweep cap during auto-tuning.
         */
        int layout_transpose_benchmark_sweeps = 8;
    };
} // namespace jacobi::svd
