#pragma once

#include <cstddef>

namespace jacobi::svd
{
    /**
     * @brief 布局转置策略；Layout-transpose policy.
     */
    enum class LayoutTransposeMode
    {
        /**
         * @brief 自动按阈值启用；Enable automatically by thresholds.
         */
        auto_select,

        /**
         * @brief 强制启用布局转置；Force-enable layout transpose.
         */
        force_enable,

        /**
         * @brief 强制禁用布局转置；Force-disable layout transpose.
         */
        force_disable
    };

    /**
     * @brief 布局转置阈值自动调优报告；Auto-tuning report for layout-transpose thresholds.
     */
    struct LayoutTransposeAutoTuneReport final
    {
        /**
         * @brief 是否执行过基准扫描；Whether benchmark scan has been executed.
         */
        bool executed = false;

        /**
         * @brief 推荐最小列数阈值；Recommended minimum-column threshold.
         */
        int recommended_min_columns = 16;

        /**
         * @brief 推荐最小元素数阈值；Recommended minimum-element threshold.
         */
        std::size_t recommended_min_elements = 4096;

        /**
         * @brief 估计最优点加速比（direct/transpose）；Estimated best-point speedup ratio (direct/transpose).
         */
        double estimated_best_speedup = 1.0;

        /**
         * @brief 样本扫描数量；Number of scanned samples.
         */
        std::size_t sample_count = 0;
    };
} // namespace jacobi::svd
