#include "jacobi/svd/domain/cuda_error.hpp"

namespace jacobi::svd
{
    CudaError::CudaError(const char *message)
        : std::runtime_error(message)
    {
    }
} // namespace jacobi::svd
