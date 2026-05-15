#pragma once

#include "src/infrastructure/io_detail.cuh"

namespace jacobi::svd::io::detail
{
    /**
     * @brief 只读内存映射文件；Read-only memory-mapped file.
     */
    class MemoryMappedInputFile final
    {
    public:
        /**
         * @brief 打开并映射输入文件；Open and map input file.
         * @param path 文件路径；File path.
         */
        explicit MemoryMappedInputFile(const std::filesystem::path &path)
        {
#ifdef _WIN32
            const std::wstring wide_path = to_windows_path(path);
            file_ = ::CreateFileW(wide_path.c_str(),
                                  GENERIC_READ,
                                  FILE_SHARE_READ,
                                  nullptr,
                                  OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL,
                                  nullptr);
            if (file_ == INVALID_HANDLE_VALUE)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileW failed");
            }

            LARGE_INTEGER file_size{};
            if (::GetFileSizeEx(file_, &file_size) == 0)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "GetFileSizeEx failed");
            }
            if (file_size.QuadPart < 0)
            {
                throw std::runtime_error("Negative file size is invalid.");
            }
            size_ = static_cast<std::size_t>(file_size.QuadPart);
            if (size_ == 0)
            {
                return;
            }

            mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateFileMappingW failed");
            }

            void *mapped = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (mapped == nullptr)
            {
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "MapViewOfFile failed");
            }
            data_ = static_cast<const std::byte *>(mapped);
#else
            file_descriptor_ = ::open(path.c_str(), O_RDONLY);
            if (file_descriptor_ < 0)
            {
                throw std::system_error(errno, std::system_category(), "open failed");
            }

            struct stat file_state{};
            if (::fstat(file_descriptor_, &file_state) != 0)
            {
                throw std::system_error(errno, std::system_category(), "fstat failed");
            }
            if (file_state.st_size < 0)
            {
                throw std::runtime_error("Negative file size is invalid.");
            }

            size_ = static_cast<std::size_t>(file_state.st_size);
            if (size_ == 0)
            {
                return;
            }

            void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
            if (mapped == MAP_FAILED)
            {
                throw std::system_error(errno, std::system_category(), "mmap failed");
            }
            data_ = static_cast<const std::byte *>(mapped);
#endif
        }

        /**
         * @brief 析构释放资源；Destroy and release resources.
         */
        ~MemoryMappedInputFile()
        {
            close();
        }

        /**
         * @brief 禁止拷贝构造；Copy constructor is disabled.
         */
        MemoryMappedInputFile(const MemoryMappedInputFile &) = delete;

        /**
         * @brief 禁止拷贝赋值；Copy assignment is disabled.
         * @return 当前对象引用；Reference to current object.
         */
        MemoryMappedInputFile &operator=(const MemoryMappedInputFile &) = delete;

        /**
         * @brief 移动构造；Move constructor.
         * @param other 源对象；Source object.
         */
        MemoryMappedInputFile(MemoryMappedInputFile &&other) noexcept
        {
            move_from(std::move(other));
        }

        /**
         * @brief 移动赋值；Move assignment.
         * @param other 源对象；Source object.
         * @return 当前对象引用；Reference to current object.
         */
        MemoryMappedInputFile &operator=(MemoryMappedInputFile &&other) noexcept
        {
            if (this != &other)
            {
                close();
                move_from(std::move(other));
            }
            return *this;
        }

        /**
         * @brief 获取映射字节视图；Get mapped bytes.
         * @return 只读字节视图；Read-only byte span.
         */
        [[nodiscard]] std::span<const std::byte> bytes() const noexcept
        {
            return {data_, size_};
        }

    private:
        /**
         * @brief 从另一个对象移动资源；Move resources from another object.
         * @param other 源对象；Source object.
         */
        void move_from(MemoryMappedInputFile &&other) noexcept
        {
            size_ = std::exchange(other.size_, 0);
            data_ = std::exchange(other.data_, nullptr);
#ifdef _WIN32
            mapping_ = std::exchange(other.mapping_, nullptr);
            file_ = std::exchange(other.file_, INVALID_HANDLE_VALUE);
#else
            file_descriptor_ = std::exchange(other.file_descriptor_, -1);
#endif
        }

        /**
         * @brief 关闭映射并释放资源；Close mapping and release resources.
         */
        void close() noexcept
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
            if (file_ != INVALID_HANDLE_VALUE)
            {
                (void)::CloseHandle(file_);
                file_ = INVALID_HANDLE_VALUE;
            }
#else
            if (data_ != nullptr)
            {
                (void)::munmap(const_cast<std::byte *>(data_), size_);
                data_ = nullptr;
            }
            if (file_descriptor_ >= 0)
            {
                (void)::close(file_descriptor_);
                file_descriptor_ = -1;
            }
#endif
            size_ = 0;
        }

        /**
         * @brief 映射长度；Mapped length.
         */
        std::size_t size_ = 0;

        /**
         * @brief 映射地址；Mapped address.
         */
        const std::byte *data_ = nullptr;

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
