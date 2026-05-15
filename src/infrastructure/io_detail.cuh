#pragma once

#include "jacobi/svd/io/io.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jacobi::svd::io::detail
{
    /**
     * @brief *.mat 头部字节数；Header bytes of *.mat.
     */
    constexpr std::size_t kMatHeaderBytes = sizeof(MatMetaData);

    /**
     * @brief *.mat 单元素字节数；Element bytes in *.mat.
     */
    constexpr std::size_t kMatElementBytes = sizeof(std::uint64_t);

    /**
     * @brief 解码分块元素数；Chunk element count for decoding.
     */
    constexpr std::size_t kDecodeChunkElements = 32U * 1024U;

    /**
     * @brief 编码分块元素数；Chunk element count for encoding.
     */
    constexpr std::size_t kEncodeChunkElements = 32U * 1024U;

    /**
     * @brief 输出映射最小容量；Minimum mapped capacity for output.
     */
    constexpr std::size_t kMinMappedCapacity = 1U << 20U;

    /**
     * @brief 任务页锁定缓冲最小容量；Minimum pinned-task buffer capacity.
     */
    constexpr std::size_t kMinPinnedTaskCapacity = 64U * 1024U;

    /**
     * @brief 检查并计算无符号乘法；Checked multiplication for unsigned sizes.
     * @param lhs 左操作数；Left operand.
     * @param rhs 右操作数；Right operand.
     * @return 乘积；Product.
     */
    [[nodiscard]] inline std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
    {
        if (lhs == 0 || rhs == 0)
        {
            return 0;
        }
        if (lhs > (std::numeric_limits<std::size_t>::max() / rhs))
        {
            throw std::overflow_error("Size multiplication overflow.");
        }
        return lhs * rhs;
    }

    /**
     * @brief 检查并计算无符号加法；Checked addition for unsigned sizes.
     * @param lhs 左操作数；Left operand.
     * @param rhs 右操作数；Right operand.
     * @return 和；Sum.
     */
    [[nodiscard]] inline std::size_t checked_add(std::size_t lhs, std::size_t rhs)
    {
        if (lhs > (std::numeric_limits<std::size_t>::max() - rhs))
        {
            throw std::overflow_error("Size addition overflow.");
        }
        return lhs + rhs;
    }

    /**
     * @brief 64 位字节交换；Byte-swap for 64-bit integers.
     * @param value 输入值；Input value.
     * @return 字节交换后结果；Byte-swapped result.
     */
    [[nodiscard]] constexpr std::uint64_t byte_swap_u64(std::uint64_t value) noexcept
    {
#if defined(__cpp_lib_byteswap) && (__cpp_lib_byteswap >= 202110L)
        return std::byteswap(value);
#else
        return ((value & 0x00000000000000FFULL) << 56U) |
               ((value & 0x000000000000FF00ULL) << 40U) |
               ((value & 0x0000000000FF0000ULL) << 24U) |
               ((value & 0x00000000FF000000ULL) << 8U) |
               ((value & 0x000000FF00000000ULL) >> 8U) |
               ((value & 0x0000FF0000000000ULL) >> 24U) |
               ((value & 0x00FF000000000000ULL) >> 40U) |
               ((value & 0xFF00000000000000ULL) >> 56U);
#endif
    }

    /**
     * @brief 主机序转网络序；Host to network byte order.
     * @param value 主机值；Host value.
     * @return 网络序值；Network-order value.
     */
    [[nodiscard]] constexpr std::uint64_t to_network_u64(std::uint64_t value) noexcept
    {
        if constexpr (std::endian::native == std::endian::little)
        {
            return byte_swap_u64(value);
        }
        return value;
    }

    /**
     * @brief 网络序转主机序；Network to host byte order.
     * @param value 网络序值；Network-order value.
     * @return 主机值；Host value.
     */
    [[nodiscard]] constexpr std::uint64_t from_network_u64(std::uint64_t value) noexcept
    {
        if constexpr (std::endian::native == std::endian::little)
        {
            return byte_swap_u64(value);
        }
        return value;
    }

    /**
     * @brief 将 double 编码为网络序；Encode double to network byte order.
     * @param value 输入浮点；Input floating-point value.
     * @return 网络序 64 位模式；Network-order 64-bit pattern.
     */
    [[nodiscard]] inline std::uint64_t encode_network_double(double value) noexcept
    {
        return to_network_u64(std::bit_cast<std::uint64_t>(value));
    }

    /**
     * @brief 将网络序解码为 double；Decode network byte order to double.
     * @param value 网络序 64 位模式；Network-order 64-bit pattern.
     * @return 主机浮点值；Host floating-point value.
     */
    [[nodiscard]] inline double decode_network_double(std::uint64_t value) noexcept
    {
        return std::bit_cast<double>(from_network_u64(value));
    }

    /**
     * @brief 验证矩阵布局；Validate matrix layout.
     * @param matrix 输入矩阵；Input matrix.
     */
    inline void validate_matrix_layout(const Matrix &matrix)
    {
        const std::size_t expected_count = checked_multiply(matrix.rows, matrix.columns);
        if (matrix.values.size() != expected_count)
        {
            throw std::invalid_argument("Matrix payload size does not match rows * columns.");
        }
    }

    /**
     * @brief 判断文本行是否为空白；Test whether text line is blank.
     * @param line 输入文本行；Input text line.
     * @return true 表示空白；true if blank.
     */
    [[nodiscard]] inline bool is_blank_line(const std::string &line)
    {
        return std::all_of(line.begin(), line.end(), [](unsigned char ch)
                           { return std::isspace(ch) != 0; });
    }

    /**
     * @brief 解析文本行浮点值；Parse floating values from one text line.
     * @param line 输入文本行；Input text line.
     * @return 浮点序列；Floating-point sequence.
     */
    [[nodiscard]] inline std::vector<double> parse_txt_row(const std::string &line)
    {
        std::istringstream row_stream(line);
        row_stream.imbue(std::locale::classic());

        std::vector<double> row_values;
        double parsed = 0.0;
        while (row_stream >> parsed)
        {
            row_values.push_back(parsed);
        }

        if (row_stream.fail() && !row_stream.eof())
        {
            throw std::invalid_argument("Invalid numeric token in text matrix row.");
        }
        return row_values;
    }

    /**
     * @brief 将 size_t 安全转换为 streamsize；Safely convert size_t to streamsize.
     * @param bytes 字节数；Byte count.
     * @return streamsize 数值；Converted streamsize value.
     */
    [[nodiscard]] inline std::streamsize checked_to_streamsize(std::size_t bytes)
    {
        if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::overflow_error("Byte size exceeds streamsize range.");
        }
        return static_cast<std::streamsize>(bytes);
    }

    /**
     * @brief 计算增长后容量；Compute grown capacity.
     * @param current 当前容量；Current capacity.
     * @param required 需求容量；Required capacity.
     * @return 新容量；New capacity.
     */
    [[nodiscard]] inline std::size_t grow_capacity(std::size_t current, std::size_t required)
    {
        if (required == 0)
        {
            return 0;
        }

        std::size_t grown = (current == 0) ? kMinPinnedTaskCapacity : current;
        while (grown < required)
        {
            if (grown > (std::numeric_limits<std::size_t>::max() / 2))
            {
                return required;
            }
            grown *= 2;
        }
        return grown;
    }

#ifdef _WIN32
    /**
     * @brief 路径转 Windows 宽字符串；Convert path to Windows wide string.
     * @param path 输入路径；Input path.
     * @return 宽字符串路径；Wide-string path.
     */
    [[nodiscard]] inline std::wstring to_windows_path(const std::filesystem::path &path)
    {
        return path.wstring();
    }
#endif
} // namespace jacobi::svd::io::detail
