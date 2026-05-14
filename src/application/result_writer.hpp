#pragma once

#include "src/application/pipeline_detail.hpp"

#include <variant>

namespace jacobi::svd::pipeline::detail
{
        /**
         * @brief 结果写出器（单线程）；Result writer (single thread).
         */
        class ResultWriter final
        {
        public:
            /**
             * @brief 构造结果写出器；Construct result writer.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             */
            ResultWriter(const std::filesystem::path &output_path, MatrixFileFormat format)
                : stream_(build_stream(output_path, format))
            {
            }

            /**
             * @brief 写出一个数据包；Write one output packet.
             * @param packet 输出数据包；Output packet.
             */
            void write_packet(const OutputPacket &packet)
            {
                write_matrix(packet.u);
                write_matrix(packet.sigma);
                write_matrix(packet.v);
            }

            /**
             * @brief 刷新输出；Flush output stream.
             */
            void flush()
            {
                std::visit([](auto &stream) {
                    stream.flush();
                },
                           stream_);
            }

        private:
            /**
             * @brief 输出流变体类型；Output stream variant type.
             */
            using OutputStreamVariant = std::variant<io::MatOutputStream, io::TxtOutputStream>;

            /**
             * @brief 构造输出流；Build output stream.
             * @param output_path 输出路径；Output path.
             * @param format 输出格式；Output format.
             * @return 输出流变体；Output stream variant.
             */
            [[nodiscard]] static OutputStreamVariant build_stream(const std::filesystem::path &output_path,
                                                                  MatrixFileFormat format)
            {
                if (format == MatrixFileFormat::mat)
                {
                    return io::MatOutputStream(output_path);
                }
                if (format == MatrixFileFormat::txt)
                {
                    return io::TxtOutputStream(output_path);
                }

                throw std::invalid_argument("Unsupported pipeline output format.");
            }

            /**
             * @brief 写出单张矩阵；Write one matrix.
             * @param matrix 输入矩阵；Input matrix.
             */
            void write_matrix(const io::Matrix &matrix)
            {
                std::visit([&matrix](auto &stream) {
                    stream.write_one(matrix);
                },
                           stream_);
            }

            /**
             * @brief 输出流实例；Output stream instance.
             */
            OutputStreamVariant stream_;
        };
} // namespace jacobi::svd::pipeline::detail
