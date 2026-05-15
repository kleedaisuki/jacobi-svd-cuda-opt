#pragma once

#include <cstddef>
#include <span>

namespace jacobi::svd::io
{
    /**
     * @brief 页锁定主机缓冲区；Pinned host buffer with one contiguous allocation.
     * @note 该缓冲区通过 cudaMallocHost/cudaFreeHost 管理；This buffer is managed by cudaMallocHost/cudaFreeHost.
     */
    class PinnedHostTaskBuffer final
    {
    public:
        /**
         * @brief 构造空缓冲；Construct an empty buffer.
         */
        PinnedHostTaskBuffer() = default;

        /**
         * @brief 析构并释放缓冲；Destroy and release the buffer.
         */
        ~PinnedHostTaskBuffer();

        /**
         * @brief 禁止拷贝构造；Copy construction is disabled.
         */
        PinnedHostTaskBuffer(const PinnedHostTaskBuffer &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        PinnedHostTaskBuffer &operator=(const PinnedHostTaskBuffer &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        PinnedHostTaskBuffer(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        PinnedHostTaskBuffer &operator=(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 预留输入区与工作区；Reserve one block for input and workspace.
         * @param input_bytes 输入区字节数；Input byte size.
         * @param workspace_bytes 工作区字节数；Workspace byte size.
         */
        void reserve(std::size_t input_bytes, std::size_t workspace_bytes);

        /**
         * @brief 获取可写输入区；Get mutable input region.
         * @return 输入区字节视图；Input byte span.
         */
        [[nodiscard]] std::span<std::byte> mutable_input_bytes() noexcept;

        /**
         * @brief 获取只读输入区；Get const input region.
         * @return 输入区字节视图；Input byte span.
         */
        [[nodiscard]] std::span<const std::byte> input_bytes() const noexcept;

        /**
         * @brief 获取可写工作区；Get mutable workspace region.
         * @return 工作区字节视图；Workspace byte span.
         */
        [[nodiscard]] std::span<std::byte> mutable_workspace_bytes() noexcept;

        /**
         * @brief 获取只读工作区；Get const workspace region.
         * @return 工作区字节视图；Workspace byte span.
         */
        [[nodiscard]] std::span<const std::byte> workspace_bytes() const noexcept;

        /**
         * @brief 当前容量（字节）；Current allocated capacity in bytes.
         * @return 容量；Capacity.
         */
        [[nodiscard]] std::size_t capacity_bytes() const noexcept;

        /**
         * @brief 当前输入区大小（字节）；Current input size in bytes.
         * @return 输入区大小；Input size.
         */
        [[nodiscard]] std::size_t input_size_bytes() const noexcept;

        /**
         * @brief 当前工作区大小（字节）；Current workspace size in bytes.
         * @return 工作区大小；Workspace size.
         */
        [[nodiscard]] std::size_t workspace_size_bytes() const noexcept;

    private:
        /**
         * @brief 释放底层缓冲；Release underlying allocation.
         */
        void release() noexcept;

        /**
         * @brief 从另一个对象移动资源；Move resources from another object.
         * @param other 源对象；Source object.
         */
        void move_from(PinnedHostTaskBuffer &&other) noexcept;

        /**
         * @brief 输入区起始地址；Input region base address.
         */
        std::byte *data_ = nullptr;

        /**
         * @brief 当前容量（字节）；Current capacity in bytes.
         */
        std::size_t capacity_bytes_ = 0;

        /**
         * @brief 输入区大小（字节）；Input size in bytes.
         */
        std::size_t input_size_bytes_ = 0;

        /**
         * @brief 工作区大小（字节）；Workspace size in bytes.
         */
        std::size_t workspace_size_bytes_ = 0;
    };
} // namespace jacobi::svd::io
