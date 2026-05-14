#include "jacobi/svd/domain/device_matrix.hpp"

#include "src/domain/cuda_check.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace jacobi::svd
{
    DeviceMatrix::DeviceMatrix(std::size_t rows, std::size_t columns)
    {
        reset(rows, columns);
    }

    DeviceMatrix::~DeviceMatrix()
    {
        if (data_ != nullptr)
        {
            (void)cudaFree(data_);
            data_ = nullptr;
        }
    }

    DeviceMatrix::DeviceMatrix(DeviceMatrix &&other) noexcept
        : rows_(std::exchange(other.rows_, 0)),
          columns_(std::exchange(other.columns_, 0)),
          data_(std::exchange(other.data_, nullptr))
    {
    }

    DeviceMatrix &DeviceMatrix::operator=(DeviceMatrix &&other) noexcept
    {
        if (this != &other)
        {
            if (data_ != nullptr)
            {
                (void)cudaFree(data_);
            }
            rows_ = std::exchange(other.rows_, 0);
            columns_ = std::exchange(other.columns_, 0);
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    void DeviceMatrix::reset(std::size_t rows, std::size_t columns)
    {
        if (rows == 0 || columns == 0)
        {
            throw std::invalid_argument("DeviceMatrix dimensions must be positive.");
        }

        if (data_ != nullptr)
        {
            JACOBI_CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }

        rows_ = rows;
        columns_ = columns;
        JACOBI_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), bytes()));
    }

    std::size_t DeviceMatrix::rows() const noexcept
    {
        return rows_;
    }

    std::size_t DeviceMatrix::columns() const noexcept
    {
        return columns_;
    }

    std::size_t DeviceMatrix::size() const noexcept
    {
        return rows_ * columns_;
    }

    std::size_t DeviceMatrix::bytes() const noexcept
    {
        return size() * sizeof(double);
    }

    double *DeviceMatrix::data() noexcept
    {
        return data_;
    }

    const double *DeviceMatrix::data() const noexcept
    {
        return data_;
    }

    void DeviceMatrix::copy_from_host(std::span<const double> host_values)
    {
        if (host_values.size() != size())
        {
            throw std::invalid_argument("Host data size does not match device matrix shape.");
        }
        JACOBI_CUDA_CHECK(cudaMemcpy(data_, host_values.data(), bytes(), cudaMemcpyHostToDevice));
    }

    std::vector<double> DeviceMatrix::copy_to_host() const
    {
        std::vector<double> host(size());
        JACOBI_CUDA_CHECK(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
        return host;
    }
} // namespace jacobi::svd

#undef JACOBI_CUDA_CHECK
