#pragma once

#include "jacobi/svd/io/matrix.cuh"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief *.txt 读取器前置声明；Forward declaration of *.txt reader.
     */
    class TxtReader;

    /**
     * @brief *.txt 写入器前置声明；Forward declaration of *.txt writer.
     */
    class TxtWriter;
    /**
     * @brief 文本矩阵文件 policy；Policy for text matrix files.
     * @note 行内空格分隔，行间换行分隔，矩阵之间空行分隔；Values are space-separated, rows are newline-separated, matrices are separated by blank lines.
     */
    struct TxtFilePolicy final
    {
        /**
         * @brief 输入状态类型；Input state type.
         */
        using Reader = TxtReader;

        /**
         * @brief 输出状态类型；Output state type.
         */
        using Writer = TxtWriter;

        /**
         * @brief 打开文本读取器；Open text reader.
         * @param path 输入路径；Input path.
         * @return 读取器对象；Reader object.
         */
        [[nodiscard]] static Reader open_reader(const std::filesystem::path &path);

        /**
         * @brief 打开文本写入器；Open text writer.
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
     * @brief *.txt 读取器实现包装；Implementation wrapper of *.txt reader.
     */
    class TxtReader final
    {
    public:
        /**
         * @brief 通过路径构造读取器；Construct reader from path.
         * @param path 输入路径；Input path.
         */
        explicit TxtReader(const std::filesystem::path &path);

        /**
         * @brief 析构读取器；Destroy reader.
         */
        ~TxtReader();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        TxtReader(const TxtReader &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        TxtReader &operator=(const TxtReader &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        TxtReader(TxtReader &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        TxtReader &operator=(TxtReader &&other) noexcept;

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
        friend struct TxtFilePolicy;
    };

    /**
     * @brief *.txt 写入器实现包装；Implementation wrapper of *.txt writer.
     */
    class TxtWriter final
    {
    public:
        /**
         * @brief 通过路径构造写入器；Construct writer from path.
         * @param path 输出路径；Output path.
         */
        explicit TxtWriter(const std::filesystem::path &path);

        /**
         * @brief 析构写入器；Destroy writer.
         */
        ~TxtWriter();

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        TxtWriter(const TxtWriter &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        TxtWriter &operator=(const TxtWriter &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        TxtWriter(TxtWriter &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        TxtWriter &operator=(TxtWriter &&other) noexcept;

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
        friend struct TxtFilePolicy;
    };
} // namespace jacobi::svd::io
