#pragma once

#include "jacobi/svd/io/matrix.cuh"

#include <filesystem>
#include <span>
#include <vector>

namespace jacobi::svd::io
{
    /**
     * @brief 读取 *.mat 中的矩阵流；Read matrix stream from *.mat.
     * @param path 输入路径；Input path.
     * @return 矩阵序列；Matrix sequence.
     */
    [[nodiscard]] std::vector<Matrix> read_mat_file(const std::filesystem::path &path);

    /**
     * @brief 写入 *.mat 矩阵流；Write matrix stream to *.mat.
     * @param path 输出路径；Output path.
     * @param matrices 矩阵序列；Matrix sequence.
     */
    void write_mat_file(const std::filesystem::path &path, std::span<const Matrix> matrices);

    /**
     * @brief 读取文本矩阵流；Read matrix stream from text file.
     * @param path 输入路径；Input path.
     * @return 矩阵序列；Matrix sequence.
     */
    [[nodiscard]] std::vector<Matrix> read_txt_file(const std::filesystem::path &path);

    /**
     * @brief 写入文本矩阵流；Write matrix stream to text file.
     * @param path 输出路径；Output path.
     * @param matrices 矩阵序列；Matrix sequence.
     */
    void write_txt_file(const std::filesystem::path &path, std::span<const Matrix> matrices);
} // namespace jacobi::svd::io
