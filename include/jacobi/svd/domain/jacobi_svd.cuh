#pragma once

#include "jacobi/svd/domain/jacobi_svd_config.cuh"
#include "jacobi/svd/domain/jacobi_svd_result.cuh"
#include "jacobi/svd/domain/layout_transpose.cuh"

#include <cstddef>
#include <span>

namespace jacobi::svd
{
    /**
     * @brief 执行单边雅可比奇异值分解；Run one-sided Jacobi singular value decomposition.
     * @param host_input 输入矩阵 A（行主序，m x n）；Input matrix A (row-major, m x n).
     * @param rows 输入行数 m；Input row count m.
     * @param columns 输入列数 n；Input column count n.
     * @param config 算法配置；Algorithm configuration.
     * @return 主机侧 SVD 结果；Host-side SVD result.
     * @note 当前实现假设 m >= n；Current implementation assumes m >= n.
     * @example
     * // 中文：输入 3x2 矩阵，输出 U、Sigma、V。
     * // English: Decompose a 3x2 matrix into U, Sigma, and V.
     * // auto result = jacobi::svd::one_sided_jacobi_svd(a, 3, 2, {});
     */
    [[nodiscard]] JacobiSvdResult one_sided_jacobi_svd(std::span<const double> host_input,
                                                       std::size_t rows,
                                                       std::size_t columns,
                                                       const JacobiSvdConfig &config = {});

    /**
     * @brief 自动扫描矩阵尺寸并给出布局转置阈值建议；Scan matrix sizes and recommend layout-transpose thresholds.
     * @param config 基准配置模板；Template configuration for benchmark.
     * @return 阈值自动调优报告；Threshold auto-tuning report.
     * @note 该函数在领域层执行微基准，不修改全局状态；This function runs micro-benchmark in domain layer without mutating global state.
     */
    [[nodiscard]] LayoutTransposeAutoTuneReport auto_tune_layout_transpose_threshold(
        const JacobiSvdConfig &config = {});
} // namespace jacobi::svd
