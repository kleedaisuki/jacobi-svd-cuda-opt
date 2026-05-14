#pragma once

#include "src/infrastructure/io_detail.hpp"

namespace jacobi::svd::io::detail
{
    class AppendMappedOutputFile final
    {
    public:
        /**
         * @brief 创建输出文件；Create output file.
         * @param path 文件路径；File path.
         */
        explicit AppendMappedOutputFile(const std::filesystem::path &path)
        {
#ifdef _WIN32
            const std::wstring wide_path = to_windows_path(path);
            file_ = ::CreateFileW(wide_path.c_str(),
                                  GENERIC_READ | GENERIC_WRITE,
                                  0,
                                  nullptr,
                                  CREATE_ALWAYS,
                                  FILE_ATTRIBUTE_NORMAL,
                                  nullptr);
            if (file_ == INVALID_HANDLE_VALUE)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileW failed");
            }
#else
            file_descriptor_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
            if (file_descriptor_ < 0)
            {
                throw std::system_error(errno, std::system_category(), "open failed");
            }
#endif
        }

        /**
         * @brief 析构并落盘；Destroy and flush.
         */
        ~AppendMappedOutputFile()
        {
            close();
        }

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        AppendMappedOutputFile(const AppendMappedOutputFile &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        AppendMappedOutputFile &operator=(const AppendMappedOutputFile &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        AppendMappedOutputFile(AppendMappedOutputFile &&other) noexcept
        {
            move_from(std::move(other));
        }

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        AppendMappedOutputFile &operator=(AppendMappedOutputFile &&other) noexcept
        {
            if (this != &other)
            {
                close();
                move_from(std::move(other));
            }
            return *this;
        }

        /**
         * @brief 追加字节数据；Append byte payload.
         * @param payload 输入字节序列；Input byte sequence.
         */
        void append(std::span<const std::byte> payload)
        {
            if (payload.empty())
            {
                return;
            }

            const std::size_t required = checked_add(size_, payload.size());
            ensure_capacity(required);
            std::memcpy(data_ + size_, payload.data(), payload.size());
            size_ = required;
        }

        /**
         * @brief 刷新已写内容；Flush written content.
         */
        void flush()
        {
            flush_mapped_prefix();
        }

    private:
        /**
         * @brief 确保映射容量；Ensure mapped capacity.
         * @param required 目标最小容量；Required minimum capacity.
         */
        void ensure_capacity(std::size_t required)
        {
            if (required <= capacity_)
            {
                return;
            }

            std::size_t new_capacity = (capacity_ == 0) ? kMinMappedCapacity : capacity_;
            while (new_capacity < required)
            {
                if (new_capacity > (std::numeric_limits<std::size_t>::max() / 2))
                {
                    new_capacity = required;
                    break;
                }
                new_capacity *= 2;
            }
            if (new_capacity < required)
            {
                new_capacity = required;
            }

            remap(new_capacity);
        }

        /**
         * @brief 重新映射到新容量；Remap to new capacity.
         * @param new_capacity 新容量；New capacity.
         */
        void remap(std::size_t new_capacity)
        {
            flush_mapped_prefix();
            unmap();
            resize_file(new_capacity);
            map(new_capacity);
            capacity_ = new_capacity;
        }

        /**
         * @brief 映射文件；Map file.
         * @param mapped_size 映射字节数；Mapped byte size.
         */
        void map(std::size_t mapped_size)
        {
            if (mapped_size == 0)
            {
                data_ = nullptr;
                return;
            }
#ifdef _WIN32
            const std::uint64_t upper = static_cast<std::uint64_t>(mapped_size) >> 32U;
            const std::uint64_t lower = static_cast<std::uint64_t>(mapped_size) & 0xFFFFFFFFULL;
            mapping_ = ::CreateFileMappingW(file_,
                                            nullptr,
                                            PAGE_READWRITE,
                                            static_cast<DWORD>(upper),
                                            static_cast<DWORD>(lower),
                                            nullptr);
            if (mapping_ == nullptr)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileMappingW failed");
            }

            void *mapped = ::MapViewOfFile(mapping_, FILE_MAP_WRITE, 0, 0, mapped_size);
            if (mapped == nullptr)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
            }
            data_ = static_cast<std::byte *>(mapped);
#else
            void *mapped = ::mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, file_descriptor_, 0);
            if (mapped == MAP_FAILED)
            {
                throw std::system_error(errno, std::system_category(), "mmap failed");
            }
            data_ = static_cast<std::byte *>(mapped);
#endif
        }

        /**
         * @brief 取消当前映射；Unmap current view.
         */
        void unmap() noexcept
        {
#ifdef _WIN32
            if (data_ != nullptr)
            {
                (void)::UnmapViewOfFile(data_);
                data_ = nullptr;
            }
            if (mapping_ != nullptr)
            {
                (void)::CloseHandle(mapping_);
                mapping_ = nullptr;
            }
#else
            if (data_ != nullptr)
            {
                (void)::munmap(data_, capacity_);
                data_ = nullptr;
            }
#endif
        }

        /**
         * @brief 调整底层文件大小；Resize underlying file.
         * @param target_size 目标文件大小；Target file size.
         */
        void resize_file(std::size_t target_size)
        {
#ifdef _WIN32
            LARGE_INTEGER position{};
            position.QuadPart = static_cast<LONGLONG>(target_size);
            if (::SetFilePointerEx(file_, position, nullptr, FILE_BEGIN) == 0)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetFilePointerEx failed");
            }
            if (::SetEndOfFile(file_) == 0)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "SetEndOfFile failed");
            }
#else
            if (::ftruncate(file_descriptor_, static_cast<off_t>(target_size)) != 0)
            {
                throw std::system_error(errno, std::system_category(), "ftruncate failed");
            }
#endif
        }

        /**
         * @brief 刷新当前已写前缀；Flush currently written prefix.
         */
        void flush_mapped_prefix()
        {
            if (data_ == nullptr || size_ == 0)
            {
                return;
            }
#ifdef _WIN32
            if (::FlushViewOfFile(data_, size_) == 0)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "FlushViewOfFile failed");
            }
            if (::FlushFileBuffers(file_) == 0)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "FlushFileBuffers failed");
            }
#else
            if (::msync(data_, size_, MS_SYNC) != 0)
            {
                throw std::system_error(errno, std::system_category(), "msync failed");
            }
#endif
        }

        /**
         * @brief 从另一个对象移动资源；Move resources from another object.
         * @param other 源对象；Source object.
         */
        void move_from(AppendMappedOutputFile &&other) noexcept
        {
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
            data_ = std::exchange(other.data_, nullptr);
#ifdef _WIN32
            file_ = std::exchange(other.file_, INVALID_HANDLE_VALUE);
            mapping_ = std::exchange(other.mapping_, nullptr);
#else
            file_descriptor_ = std::exchange(other.file_descriptor_, -1);
#endif
        }

        /**
         * @brief 关闭映射和句柄；Close mapping and handles.
         */
        void close() noexcept
        {
            try
            {
                flush_mapped_prefix();
            }
            catch (...)
            {
            }

            unmap();

#ifdef _WIN32
            if (file_ != INVALID_HANDLE_VALUE)
            {
                LARGE_INTEGER position{};
                position.QuadPart = static_cast<LONGLONG>(size_);
                (void)::SetFilePointerEx(file_, position, nullptr, FILE_BEGIN);
                (void)::SetEndOfFile(file_);
                (void)::CloseHandle(file_);
                file_ = INVALID_HANDLE_VALUE;
            }
#else
            if (file_descriptor_ >= 0)
            {
                (void)::ftruncate(file_descriptor_, static_cast<off_t>(size_));
                (void)::close(file_descriptor_);
                file_descriptor_ = -1;
            }
#endif
            size_ = 0;
            capacity_ = 0;
        }

        /**
         * @brief 已写字节数；Written byte count.
         */
        std::size_t size_ = 0;

        /**
         * @brief 映射容量；Mapped capacity.
         */
        std::size_t capacity_ = 0;

        /**
         * @brief 映射地址；Mapped address.
         */
        std::byte *data_ = nullptr;

#ifdef _WIN32
        /**
         * @brief 文件句柄；File handle.
         */
        HANDLE file_ = INVALID_HANDLE_VALUE;

        /**
         * @brief 映射句柄；Mapping handle.
         */
        HANDLE mapping_ = nullptr;
#else
        /**
         * @brief POSIX 文件描述符；POSIX file descriptor.
         */
        int file_descriptor_ = -1;
#endif
    };
} // namespace jacobi::svd::io::detail
