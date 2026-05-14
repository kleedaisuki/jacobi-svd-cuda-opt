#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace jacobi::svd
{
    /**
     * @brief 行主序设备矩阵封装；Row-major device matrix wrapper.
     * @note 该类型负责 GPU 内存生命周期，kernel 侧仅接收裸指针；This type owns GPU memory lifecycle and kernels only receive raw pointers.
     */
    class DeviceMatrix final
    {
    public:
        /**
         * @brief 默认构造空矩阵；Default construct an empty matrix.
         */
        DeviceMatrix() = default;

        /**
         * @brief 构造并分配设备矩阵；Construct and allocate a device matrix.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         */
        DeviceMatrix(std::size_t rows, std::size_t columns);

        /**
         * @brief 析构并释放设备内存；Destroy and release device memory.
         */
        ~DeviceMatrix();

        /**
         * @brief 禁止拷贝构造；Copy construction is disabled.
         */
        DeviceMatrix(const DeviceMatrix &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceMatrix &operator=(const DeviceMatrix &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        DeviceMatrix(DeviceMatrix &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceMatrix &operator=(DeviceMatrix &&other) noexcept;

        /**
         * @brief 重新分配矩阵尺寸；Reallocate matrix with new shape.
         * @param rows 行数；Row count.
         * @param columns 列数；Column count.
         */
        void reset(std::size_t rows, std::size_t columns);

        /**
         * @brief 读取行数；Get number of rows.
         * @return 行数；Row count.
         */
        [[nodiscard]] std::size_t rows() const noexcept;

        /**
         * @brief 读取列数；Get number of columns.
         * @return 列数；Column count.
         */
        [[nodiscard]] std::size_t columns() const noexcept;

        /**
         * @brief 元素总数；Total number of elements.
         * @return 元素数；Element count.
         */
        [[nodiscard]] std::size_t size() const noexcept;

        /**
         * @brief 字节总数；Total bytes.
         * @return 字节数；Byte count.
         */
        [[nodiscard]] std::size_t bytes() const noexcept;

        /**
         * @brief 获取可写裸指针；Get mutable raw pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] double *data() noexcept;

        /**
         * @brief 获取只读裸指针；Get const raw pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] const double *data() const noexcept;

        /**
         * @brief 将主机数据拷贝到设备；Copy host data into device matrix.
         * @param host_values 主机行主序数据；Host row-major values.
         */
        void copy_from_host(std::span<const double> host_values);

        /**
         * @brief 将设备数据拷贝回主机；Copy device data back to host.
         * @return 主机行主序数据；Host row-major values.
         */
        [[nodiscard]] std::vector<double> copy_to_host() const;

    private:
        /**
         * @brief 行数成员；Row count field.
         */
        std::size_t rows_ = 0;

        /**
         * @brief 列数成员；Column count field.
         */
        std::size_t columns_ = 0;

        /**
         * @brief 设备数据指针；Device data pointer.
         */
        double *data_ = nullptr;
    };
} // namespace jacobi::svd
