#pragma once

#include "jacobi/svd/io/mat_file.hpp"
#include "jacobi/svd/io/txt_file.hpp"

#include <concepts>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief 矩阵输入 policy 概念；Concept for matrix input policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixInputPolicy = requires(const std::filesystem::path &path, typename Policy::Reader &reader, Matrix &matrix)
    {
        typename Policy::Reader;
        {
            Policy::open_reader(path)
        } -> std::same_as<typename Policy::Reader>;
        {
            Policy::read_next(reader, matrix)
        } -> std::same_as<bool>;
    };

    /**
     * @brief 矩阵输出 policy 概念；Concept for matrix output policy.
     * @tparam Policy policy 类型；Policy type.
     */
    template <typename Policy>
    concept MatrixOutputPolicy = requires(const std::filesystem::path &path,
                                          typename Policy::Writer &writer,
                                          const Matrix &matrix)
    {
        typename Policy::Writer;
        {
            Policy::open_writer(path)
        } -> std::same_as<typename Policy::Writer>;
        {
            Policy::write_next(writer, matrix)
        } -> std::same_as<void>;
        {
            Policy::flush(writer)
        } -> std::same_as<void>;
    };

    /**
     * @brief 矩阵输入流模板；Matrix input stream template.
     * @tparam Policy 输入 policy；Input policy.
     */
    template <MatrixInputPolicy Policy>
    class MatrixInputStream final
    {
    public:
        /**
         * @brief 构造输入流；Construct input stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixInputStream(const std::filesystem::path &path)
            : reader_(Policy::open_reader(path))
        {
        }

        /**
         * @brief 读取下一张矩阵；Read one matrix.
         * @param matrix 输出矩阵；Output matrix.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one matrix is read, false on EOF.
         */
        bool read_one(Matrix &matrix)
        {
            if (eof_)
            {
                return false;
            }

            const bool has_matrix = Policy::read_next(reader_, matrix);
            eof_ = !has_matrix;
            return has_matrix;
        }

        /**
         * @brief 操作符重载：读取下一张矩阵；Operator overload: read one matrix.
         * @param matrix 输出矩阵；Output matrix.
         * @return 当前输入流对象；Current input stream object.
         */
        MatrixInputStream &operator>>(Matrix &matrix)
        {
            (void)read_one(matrix);
            return *this;
        }

        /**
         * @brief 是否已到文件末尾；Whether EOF is reached.
         * @return EOF 状态；EOF state.
         */
        [[nodiscard]] bool eof() const noexcept
        {
            return eof_;
        }

        /**
         * @brief 读取全部矩阵（兼容接口）；Read all matrices (compatibility API).
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] std::vector<Matrix> read_all()
        {
            std::vector<Matrix> matrices;
            Matrix matrix;
            while (read_one(matrix))
            {
                matrices.push_back(std::move(matrix));
            }
            return matrices;
        }

        /**
         * @brief 布尔语义：尚未 EOF；Boolean semantics: not EOF yet.
         * @return true 表示可继续尝试读取；true means stream can still be read.
         */
        explicit operator bool() const noexcept
        {
            return !eof_;
        }

    private:
        /**
         * @brief 读取器状态；Reader state.
         */
        typename Policy::Reader reader_;

        /**
         * @brief EOF 状态；EOF state.
         */
        bool eof_ = false;
    };

    /**
     * @brief 矩阵输出流模板；Matrix output stream template.
     * @tparam Policy 输出 policy；Output policy.
     */
    template <MatrixOutputPolicy Policy>
    class MatrixOutputStream final
    {
    public:
        /**
         * @brief 构造输出流；Construct output stream.
         * @param path 文件路径；File path.
         */
        explicit MatrixOutputStream(const std::filesystem::path &path)
            : writer_(Policy::open_writer(path))
        {
        }

        /**
         * @brief 写入一张矩阵；Write one matrix.
         * @param matrix 输入矩阵；Input matrix.
         */
        void write_one(const Matrix &matrix)
        {
            Policy::write_next(writer_, matrix);
        }

        /**
         * @brief 操作符重载：写入一张矩阵；Operator overload: write one matrix.
         * @param matrix 输入矩阵；Input matrix.
         * @return 当前输出流对象；Current output stream object.
         */
        MatrixOutputStream &operator<<(const Matrix &matrix)
        {
            write_one(matrix);
            return *this;
        }

        /**
         * @brief 刷新输出；Flush output.
         */
        void flush()
        {
            Policy::flush(writer_);
        }

        /**
         * @brief 写入全部矩阵（兼容接口）；Write all matrices (compatibility API).
         * @param matrices 矩阵序列；Matrix sequence.
         */
        void write_all(std::span<const Matrix> matrices)
        {
            for (const Matrix &matrix : matrices)
            {
                write_one(matrix);
            }
            flush();
        }

    private:
        /**
         * @brief 写入器状态；Writer state.
         */
        typename Policy::Writer writer_;
    };
    /**
     * @brief 类型别名：*.mat 输入流；Type alias: *.mat input stream.
     */
    using MatInputStream = MatrixInputStream<MatFilePolicy>;

    /**
     * @brief 类型别名：*.mat 输出流；Type alias: *.mat output stream.
     */
    using MatOutputStream = MatrixOutputStream<MatFilePolicy>;

    /**
     * @brief 类型别名：*.txt 输入流；Type alias: *.txt input stream.
     */
    using TxtInputStream = MatrixInputStream<TxtFilePolicy>;

    /**
     * @brief 类型别名：*.txt 输出流；Type alias: *.txt output stream.
     */
    using TxtOutputStream = MatrixOutputStream<TxtFilePolicy>;
} // namespace jacobi::svd::io
