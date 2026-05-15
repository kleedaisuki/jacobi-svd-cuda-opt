#include "jacobi/svd/io/pinned_host_task_buffer.cuh"

#include "src/infrastructure/io_detail.cuh"

namespace jacobi::svd::io
{
    using namespace detail;
    PinnedHostTaskBuffer::~PinnedHostTaskBuffer()
    {
        release();
    }

    PinnedHostTaskBuffer::PinnedHostTaskBuffer(PinnedHostTaskBuffer &&other) noexcept
    {
        move_from(std::move(other));
    }

    PinnedHostTaskBuffer &PinnedHostTaskBuffer::operator=(PinnedHostTaskBuffer &&other) noexcept
    {
        if (this != &other)
        {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    void PinnedHostTaskBuffer::reserve(std::size_t input_bytes, std::size_t workspace_bytes)
    {
        const std::size_t required_bytes = checked_add(input_bytes, workspace_bytes);
        if (required_bytes > capacity_bytes_)
        {
            const std::size_t new_capacity = grow_capacity(capacity_bytes_, required_bytes);
            void *new_data = nullptr;
            const cudaError_t status = ::cudaMallocHost(&new_data, new_capacity);
            if (status != cudaSuccess)
            {
                throw std::runtime_error("cudaMallocHost failed: " +
                                         std::string(::cudaGetErrorString(status)));
            }

            release();
            data_ = static_cast<std::byte *>(new_data);
            capacity_bytes_ = new_capacity;
        }

        input_size_bytes_ = input_bytes;
        workspace_size_bytes_ = workspace_bytes;
    }

    std::span<std::byte> PinnedHostTaskBuffer::mutable_input_bytes() noexcept
    {
        return {data_, input_size_bytes_};
    }

    std::span<const std::byte> PinnedHostTaskBuffer::input_bytes() const noexcept
    {
        return {data_, input_size_bytes_};
    }

    std::span<std::byte> PinnedHostTaskBuffer::mutable_workspace_bytes() noexcept
    {
        return {data_ + input_size_bytes_, workspace_size_bytes_};
    }

    std::span<const std::byte> PinnedHostTaskBuffer::workspace_bytes() const noexcept
    {
        return {data_ + input_size_bytes_, workspace_size_bytes_};
    }

    std::size_t PinnedHostTaskBuffer::capacity_bytes() const noexcept
    {
        return capacity_bytes_;
    }

    std::size_t PinnedHostTaskBuffer::input_size_bytes() const noexcept
    {
        return input_size_bytes_;
    }

    std::size_t PinnedHostTaskBuffer::workspace_size_bytes() const noexcept
    {
        return workspace_size_bytes_;
    }

    void PinnedHostTaskBuffer::release() noexcept
    {
        if (data_ != nullptr)
        {
            (void)::cudaFreeHost(data_);
            data_ = nullptr;
        }
        capacity_bytes_ = 0;
        input_size_bytes_ = 0;
        workspace_size_bytes_ = 0;
    }

    void PinnedHostTaskBuffer::move_from(PinnedHostTaskBuffer &&other) noexcept
    {
        data_ = std::exchange(other.data_, nullptr);
        capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
        input_size_bytes_ = std::exchange(other.input_size_bytes_, 0);
        workspace_size_bytes_ = std::exchange(other.workspace_size_bytes_, 0);
    }
} // namespace jacobi::svd::io
