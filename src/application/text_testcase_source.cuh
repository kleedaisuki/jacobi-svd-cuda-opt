#pragma once

#include "jacobi/svd/io/io.cuh"

#include <filesystem>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 文本输入读取阶段；Text input stage.
         */
        class TextTestcaseSource final
        {
        public:
            /**
             * @brief 构造文本输入阶段；Construct text input stage.
             * @param input_path 输入路径；Input path.
             */
            explicit TextTestcaseSource(const std::filesystem::path &input_path)
                : stream_(io::TxtInputStream(input_path))
            {
            }

            /**
             * @brief 读取下一张矩阵；Read next matrix.
             * @param matrix 输出矩阵；Output matrix.
             * @return 读取成功返回 true，EOF 返回 false；Returns true on success, false on EOF.
             */
            bool read_next(io::Matrix &matrix)
            {
                return stream_.read_one(matrix);
            }

        private:
            /**
             * @brief 文本输入流；Text input stream.
             */
            io::TxtInputStream stream_;
        };
} // namespace jacobi::svd::pipeline::detail
