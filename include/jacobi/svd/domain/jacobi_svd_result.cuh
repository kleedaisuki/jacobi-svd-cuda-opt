#pragma once

#include <cstddef>
#include <vector>

namespace jacobi::svd
{
    /**
     * @brief Jacobi SVD 主机结果容器；Host-side result container for Jacobi SVD.
     */
    struct JacobiSvdResult final
    {
        /**
         * @brief 输入矩阵行数 m；Input row count m.
         */
        std::size_t rows = 0;

        /**
         * @brief 输入矩阵列数 n；Input column count n.
         */
        std::size_t columns = 0;

        /**
         * @brief 左奇异矩阵 U（m x n，行主序）；Left singular matrix U (m x n, row-major).
         */
        std::vector<double> u;

        /**
         * @brief 奇异值向量 Sigma（长度 n）；Singular values Sigma (length n).
         */
        std::vector<double> sigma;

        /**
         * @brief 右奇异矩阵 V（n x n，行主序）；Right singular matrix V (n x n, row-major).
         */
        std::vector<double> v;

        /**
         * @brief 实际执行 sweep 次数；Number of executed sweeps.
         */
        int sweeps = 0;
    };
} // namespace jacobi::svd
