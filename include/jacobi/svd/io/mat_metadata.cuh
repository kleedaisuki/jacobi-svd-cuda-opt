#pragma once

#include <cstdint>

namespace jacobi::svd::io
{
    /**
     * @brief *.mat 元数据头；Metadata header for *.mat.
     * @note 文件中使用网络字节序（Network Byte Order）；Network byte order is used on disk.
     */
    struct MatMetaData final
    {
        /**
         * @brief 行数（64 位无符号整数）；Row count (64-bit unsigned integer).
         */
        std::uint64_t rows = 0;

        /**
         * @brief 列数（64 位无符号整数）；Column count (64-bit unsigned integer).
         */
        std::uint64_t columns = 0;
    };
} // namespace jacobi::svd::io
