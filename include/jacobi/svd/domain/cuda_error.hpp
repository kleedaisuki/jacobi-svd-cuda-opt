#pragma once

#include <stdexcept>

namespace jacobi::svd
{
    /**
     * @brief CUDA 运行时错误封装；CUDA runtime error wrapper.
     */
    class CudaError final : public std::runtime_error
    {
    public:
        /**
         * @brief 构造 CUDA 异常对象；Construct a CUDA exception object.
         * @param message 错误消息；Error message.
         */
        explicit CudaError(const char *message);
    };
} // namespace jacobi::svd
