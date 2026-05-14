#pragma once

#include "jacobi/svd/application/pipeline.hpp"
#include "jacobi/svd/io/io.hpp"

#include <cstddef>
#include <filesystem>

namespace jacobi::svd::pipeline::detail
{
/**
 * @brief 计算 rows*columns（带溢出检查）；Compute rows*columns with overflow check.
 * @param rows 行数；Row count.
 * @param columns 列数；Column count.
 * @return 元素数量；Element count.
 */
[[nodiscard]] std::size_t checked_element_count(std::size_t rows, std::size_t columns);

/**
 * @brief 校验矩阵布局一致性；Validate matrix layout consistency.
 * @param matrix 输入矩阵；Input matrix.
 * @param testcase_index 测试用例索引；Testcase index.
 */
void validate_testcase_matrix(const io::Matrix &matrix, std::size_t testcase_index);

/**
 * @brief 解析文件格式（支持 auto）；Resolve file format with auto-detection.
 * @param requested 请求格式；Requested format.
 * @param path 文件路径；File path.
 * @return 实际格式；Resolved format.
 */
[[nodiscard]] MatrixFileFormat resolve_file_format(MatrixFileFormat requested, const std::filesystem::path &path);

/**
 * @brief 确保输出目录存在；Ensure output directory exists.
 * @param output_path 输出文件路径；Output file path.
 */
void ensure_output_directory(const std::filesystem::path &output_path);

        /**
         * @brief 输出数据包（单个 testcase 的 U/Sigma/V）；Output packet for one testcase (U/Sigma/V).
         */
        struct OutputPacket final
        {
            /**
             * @brief 测试用例索引；Testcase index.
             */
            std::size_t testcase_index = 0;

            /**
             * @brief 当前用例 sweep 次数；Sweep count for this testcase.
             */
            int sweeps = 0;

            /**
             * @brief 左奇异矩阵 U；Left singular matrix U.
             */
            io::Matrix u;

            /**
             * @brief 奇异值矩阵 Sigma(1xn)；Singular value matrix Sigma(1xn).
             */
            io::Matrix sigma;

            /**
             * @brief 右奇异矩阵 V；Right singular matrix V.
             */
            io::Matrix v;
        };
} // namespace jacobi::svd::pipeline::detail
