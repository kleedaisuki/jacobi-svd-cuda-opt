#pragma once

#include "jacobi/svd/io/matrix.cuh"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief *.mat 读取器前置声明；Forward declaration of *.mat reader.
     */
    class MatReader;

    /**
     * @brief *.mat 写入器前置声明；Forward declaration of *.mat writer.
     */
    class MatWriter;
    /**
     * @brief *.mat 文件 policy；Policy for *.mat files.
     */
    struct MatFilePolicy final
    {
        /**
         * @brief 输入状态类型；Input state type.
         */
        using Reader = MatReader;

        /**
         * @brief 输出状态类型；Output state type.
         */
        using Writer = MatWriter;

        /**
         * @brief 打开 *.mat 读取器；Open *.mat reader.
         * @param path 输入路径；Input path.
         * @return 读取器对象；Reader object.
         */
        [[nodiscard]] static Reader open_reader(const std::filesystem::path &path);

        /**
         * @brief 打开 *.mat 写入器；Open *.mat writer.
         * @param path 输出路径；Output path.
         * @return 写入器对象；Writer object.
         */
        [[nodiscard]] static Writer open_writer(const std::filesystem::path &path);

        /**
         * @brief 读取下一张矩阵；Read next matrix.
         * @param reader 读取器；Reader.
         * @param matrix 输出矩阵；Output matrix.
         * @return 成功读取返回 true，EOF 返回 false；Returns true if one matrix is read, false on EOF.
         */
        [[nodiscard]] static bool read_next(Reader &reader, Matrix &matrix);

        /**
         * @brief 写入下一张矩阵；Write next matrix.
         * @param writer 写入器；Writer.
         * @param matrix 输入矩阵；Input matrix.
         */
        static void write_next(Writer &writer, const Matrix &matrix);

        /**
         * @brief 刷新写入缓冲；Flush output state.
         * @param writer 写入器；Writer.
         */
        static void flush(Writer &writer);

        /**
         * @brief 批量读取（兼容接口）；Bulk read (compatibility API).
         * @param path 输入路径；Input path.
         * @return 矩阵序列；Matrix sequence.
         */
        [[nodiscard]] static std::vector<Matrix> read(const std::filesystem::path &path);

        /**
         * @brief 批量写入（兼容接口）；Bulk write (compatibility API).
         * @param path 输出路径；Output path.
         * @param matrices 矩阵序列；Matrix sequence.
         */
        static void write(const std::filesystem::path &path, std::span<const Matrix> matrices);
    };
    /**
     * @brief *.mat 读取器实现包装；Implementation wrapper of *.mat reader.
     */
    class MatReader final
    {
    public:
        /**
         * @brief 通过路径构造读取器；Construct reader from path.
         * @param path 输入路径；Input path.
         */
        explicit MatReader(const std::filesystem::path &path);

        /**
         * @brief 析构读取器；Destroy reader.
         */
        ~MatReader();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        MatReader(const MatReader &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        MatReader &operator=(const MatReader &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        MatReader(MatReader &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        MatReader &operator=(MatReader &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct MatFilePolicy;
    };

    /**
     * @brief *.mat 写入器实现包装；Implementation wrapper of *.mat writer.
     */
    class MatWriter final
    {
    public:
        /**
         * @brief 通过路径构造写入器；Construct writer from path.
         * @param path 输出路径；Output path.
         */
        explicit MatWriter(const std::filesystem::path &path);

        /**
         * @brief 析构写入器；Destroy writer.
         */
        ~MatWriter();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        MatWriter(const MatWriter &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        MatWriter &operator=(const MatWriter &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        MatWriter(MatWriter &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        MatWriter &operator=(MatWriter &&other) noexcept;

    private:
        /**
         * @brief 实现体前置声明；Forward declaration of implementation.
         */
        struct Impl;

        /**
         * @brief 唯一实现体指针；Unique pointer of implementation.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 授权 policy 访问实现体；Grant policy access to implementation.
         */
        friend struct MatFilePolicy;
    };
} // namespace jacobi::svd::io
