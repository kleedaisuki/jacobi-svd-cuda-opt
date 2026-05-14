#pragma once

#include "jacobi/svd/domain/jacobi_svd.hpp"

#include <cstddef>
#include <span>

namespace jacobi::svd::detail
{
    /**
     * @brief 计算矩阵元素数量；Compute matrix element count.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @return 元素数量；Element count.
     */
    [[nodiscard]] std::size_t matrix_element_count(std::size_t rows, std::size_t columns);

    /**
     * @brief 校验布局转置配置；Validate layout-transpose related configuration.
     * @param config 算法配置；Algorithm configuration.
     */
    void validate_layout_transpose_config(const JacobiSvdConfig &config);

    /**
     * @brief 判断是否启用布局转置路径；Decide whether to enable layout-transpose path.
     * @param rows 行数；Row count.
     * @param columns 列数；Column count.
     * @param config 算法配置；Algorithm configuration.
     * @return true 表示启用布局转置；true if layout transpose should be enabled.
     */
    [[nodiscard]] bool should_use_layout_transpose(int rows, int columns, const JacobiSvdConfig &config);

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
                                                                    bool use_layout_transpose);
} // namespace jacobi::svd::detail
