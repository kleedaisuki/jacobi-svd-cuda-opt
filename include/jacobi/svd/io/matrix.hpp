#pragma once

#include <cstddef>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief 行主序矩阵容器；Row-major matrix container.
     */
    struct Matrix final
    {
        /**
         * @brief 矩阵行数；Matrix row count.
         */
        std::size_t rows = 0;

        /**
         * @brief 矩阵列数；Matrix column count.
         */
        std::size_t columns = 0;

        /**
         * @brief 行主序元素数据；Row-major element values.
         */
        std::vector<double> values;
    };
} // namespace jacobi::svd::io
