#pragma once

#include "src/domain/cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace jacobi::svd::detail
{
    /**
     * @brief 设备缓冲区 RAII 封装；RAII wrapper for device buffers.
     * @tparam T 元素类型；Element type.
     */
    template <typename T>
    class DeviceBuffer final
    {
    public:
        /**
         * @brief 默认构造空缓冲区；Default construct an empty buffer.
         */
        DeviceBuffer() = default;

        /**
         * @brief 构造并分配缓冲区；Construct and allocate buffer.
         * @param count 元素数量；Element count.
         */
        explicit DeviceBuffer(std::size_t count)
        {
            reset(count);
        }

        /**
         * @brief 析构释放资源；Destroy and release resource.
         */
        ~DeviceBuffer()
        {
            release();
        }

        /**
         * @brief 禁止拷贝；Copy is disabled.
         */
        DeviceBuffer(const DeviceBuffer &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源缓冲区；Source buffer.
         */
        DeviceBuffer(DeviceBuffer &&other) noexcept
            : count_(std::exchange(other.count_, 0)), data_(std::exchange(other.data_, nullptr))
        {
        }

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源缓冲区；Source buffer.
         * @return 当前对象引用；Reference to current object.
         */
        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                release();
                count_ = std::exchange(other.count_, 0);
                data_ = std::exchange(other.data_, nullptr);
            }
            return *this;
        }

        /**
         * @brief 重新分配缓冲区；Reallocate buffer.
         * @param count 元素数量；Element count.
         */
        void reset(std::size_t count)
        {
            release();
            count_ = count;
            if (count_ == 0)
            {
                return;
            }
            JACOBI_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
        }

        /**
         * @brief 获取数据指针；Get data pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] T *data() noexcept
        {
            return data_;
        }

        /**
         * @brief 获取常量数据指针；Get const data pointer.
         * @return 设备指针；Device pointer.
         */
        [[nodiscard]] const T *data() const noexcept
        {
            return data_;
        }

        /**
         * @brief 获取元素数量；Get element count.
         * @return 元素数量；Element count.
         */
        [[nodiscard]] std::size_t size() const noexcept
        {
            return count_;
        }

    private:
        /**
         * @brief 释放设备内存；Release device memory.
         */
        void release() noexcept
        {
            if (data_ != nullptr)
            {
                (void)cudaFree(data_);
                data_ = nullptr;
            }
            count_ = 0;
        }

        /**
         * @brief 元素数量；Element count.
         */
        std::size_t count_ = 0;

        /**
         * @brief 设备数据指针；Device data pointer.
         */
        T *data_ = nullptr;
    };
} // namespace jacobi::svd::detail
