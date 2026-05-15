#include "src/application/pipeline_detail.cuh"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 计算 rows*columns（带溢出检查）；Compute rows*columns with overflow check.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         * @return 元素数量；Element count.
         */
        [[nodiscard]] std::size_t checked_element_count(std::size_t rows, std::size_t columns)
        {
            if (rows == 0 || columns == 0)
            {
                return 0;
            }
            if (rows > (std::numeric_limits<std::size_t>::max() / columns))
            {
                throw std::overflow_error("Matrix element count overflow.");
            }
            return rows * columns;
        }

        /**
         * @brief 校验矩阵布局一致性；Validate matrix layout consistency.
         * @param matrix 输入矩阵；Input matrix.
         * @param testcase_index 测试用例索引；Testcase index.
         */
        void validate_testcase_matrix(const io::Matrix &matrix, std::size_t testcase_index)
        {
            if (matrix.rows == 0 || matrix.columns == 0)
            {
                throw std::invalid_argument("Testcase[" + std::to_string(testcase_index) +
                                            "] has zero dimension.");
            }

            const std::size_t expected = checked_element_count(matrix.rows, matrix.columns);
            if (matrix.values.size() != expected)
            {
                throw std::invalid_argument("Testcase[" + std::to_string(testcase_index) +
                                            "] payload size does not match rows*columns.");
            }
        }

        /**
         * @brief 解析文件格式（支持 auto）；Resolve file format with auto-detection.
         * @param requested 请求格式；Requested format.
         * @param path 文件路径；File path.
         * @return 实际格式；Resolved format.
         */
        [[nodiscard]] MatrixFileFormat resolve_file_format(MatrixFileFormat requested,
                                                           const std::filesystem::path &path)
        {
            if (requested != MatrixFileFormat::auto_detect)
            {
                return requested;
            }

            std::string extension = path.extension().string();
            std::transform(extension.begin(),
                           extension.end(),
                           extension.begin(),
                           [](unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });

            if (extension == ".mat")
            {
                return MatrixFileFormat::mat;
            }
            if (extension == ".txt")
            {
                return MatrixFileFormat::txt;
            }

            throw std::invalid_argument("Cannot auto-detect file format for path: " + path.string());
        }

        /**
         * @brief 确保输出目录存在；Ensure output directory exists.
         * @param output_path 输出文件路径；Output file path.
         */
        void ensure_output_directory(const std::filesystem::path &output_path)
        {
            const std::filesystem::path parent = output_path.parent_path();
            if (!parent.empty())
            {
                std::filesystem::create_directories(parent);
            }
        }
} // namespace jacobi::svd::pipeline::detail
