#pragma once

#include "jacobi/svd/domain/cuda_error.cuh"

#include <cuda_runtime.h>

#include <sstream>

namespace jacobi::svd::detail
{
    /**
     * @brief 将 CUDA 错误码转换为异常；Convert CUDA status to exception.
     * @param status CUDA 返回码；CUDA return status.
     * @param expression 触发检查的表达式文本；Checked expression text.
     * @param file 源文件名；Source file name.
     * @param line 源码行号；Source line number.
     */
    inline void throw_if_cuda_failed(cudaError_t status, const char *expression, const char *file, int line)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        std::ostringstream stream;
        stream << "CUDA call failed: " << expression << " @ " << file << ':' << line
               << ", code=" << static_cast<int>(status) << ", message=" << cudaGetErrorString(status);
        throw CudaError(stream.str().c_str());
    }
} // namespace jacobi::svd::detail

/**
 * @brief CUDA 调用检查宏；CUDA call checking macro.
 */
#define JACOBI_CUDA_CHECK(EXPR) ::jacobi::svd::detail::throw_if_cuda_failed((EXPR), #EXPR, __FILE__, __LINE__)
