#include "jacobi/svd/io/files.hpp"

#include "jacobi/svd/io/mat_file.hpp"
#include "jacobi/svd/io/txt_file.hpp"

namespace jacobi::svd::io
{
    std::vector<Matrix> read_mat_file(const std::filesystem::path &path)
    {
        return MatFilePolicy::read(path);
    }

    void write_mat_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        MatFilePolicy::write(path, matrices);
    }

    std::vector<Matrix> read_txt_file(const std::filesystem::path &path)
    {
        return TxtFilePolicy::read(path);
    }

    void write_txt_file(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        TxtFilePolicy::write(path, matrices);
    }
} // namespace jacobi::svd::io
