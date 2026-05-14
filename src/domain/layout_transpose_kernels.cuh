#pragma once

#include "src/domain/cuda_check.cuh"
#include "src/domain/matrix_index.cuh"

#include <cuda_runtime.h>

namespace jacobi::svd::detail
{
    /**
     * @brief 转置 tile 边长；Transpose tile width/height.
     */
    constexpr int k_transpose_tile_dim = 32;

    /**
     * @brief 转置 block 的行步进；Row-stride inside transpose block.
     */
    constexpr int k_transpose_block_rows = 8;

    /**
     * @brief 行主序矩阵转置核；Row-major matrix transpose kernel.
     * @param src 输入矩阵（行主序，src_rows x src_cols）；Input matrix (row-major, src_rows x src_cols).
     * @param dst 输出矩阵（行主序，src_cols x src_rows）；Output matrix (row-major, src_cols x src_rows).
     * @param src_rows 输入行数；Input row count.
     * @param src_cols 输入列数；Input column count.
     * @note 使用共享内存 tile，分别保证读写阶段的内存合并；Uses shared-memory tiles to coalesce both read and write phases.
     */
    __global__ void transpose_row_major_kernel(const double *src, double *dst, int src_rows, int src_cols)
    {
        __shared__ double tile[k_transpose_tile_dim][k_transpose_tile_dim + 1];

        const int x = static_cast<int>(blockIdx.x * k_transpose_tile_dim + threadIdx.x);
        const int y = static_cast<int>(blockIdx.y * k_transpose_tile_dim + threadIdx.y);

        for (int offset = 0; offset < k_transpose_tile_dim; offset += k_transpose_block_rows)
        {
            const int src_row = y + offset;
            if (x < src_cols && src_row < src_rows)
            {
                tile[threadIdx.y + offset][threadIdx.x] = src[row_major_index(src_row, x, src_cols)];
            }
        }
        __syncthreads();

        const int transposed_x = static_cast<int>(blockIdx.y * k_transpose_tile_dim + threadIdx.x);
        const int transposed_y = static_cast<int>(blockIdx.x * k_transpose_tile_dim + threadIdx.y);

        for (int offset = 0; offset < k_transpose_tile_dim; offset += k_transpose_block_rows)
        {
            const int dst_row = transposed_y + offset;
            if (transposed_x < src_rows && dst_row < src_cols)
            {
                dst[row_major_index(dst_row, transposed_x, src_rows)] =
                    tile[threadIdx.x][threadIdx.y + offset];
            }
        }
    }

    /**
     * @brief 启动 row-major 到 column-major 的布局转置；Launch layout transpose from row-major to column-major.
     * @param src_row_major 输入逻辑矩阵（行主序，rows x columns）；Input logical matrix (row-major, rows x columns).
     * @param dst_column_major 输出逻辑矩阵（列主序，rows x columns）；Output logical matrix (column-major, rows x columns).
     * @param rows 逻辑行数；Logical row count.
     * @param columns 逻辑列数；Logical column count.
     * @note 通过“row-major 转置”实现布局重排；Layout conversion is implemented via a row-major transpose.
     */
    void launch_row_to_column_layout_transpose(const double *src_row_major,
                                               double *dst_column_major,
                                               int rows,
                                               int columns)
    {
        const dim3 block(static_cast<unsigned int>(k_transpose_tile_dim),
                         static_cast<unsigned int>(k_transpose_block_rows),
                         1U);
        const dim3 grid(static_cast<unsigned int>((columns + k_transpose_tile_dim - 1) / k_transpose_tile_dim),
                        static_cast<unsigned int>((rows + k_transpose_tile_dim - 1) / k_transpose_tile_dim),
                        1U);
        transpose_row_major_kernel<<<grid, block>>>(src_row_major, dst_column_major, rows, columns);
        JACOBI_CUDA_CHECK(cudaGetLastError());
    }

    /**
     * @brief 启动 column-major 到 row-major 的布局转置；Launch layout transpose from column-major to row-major.
     * @param src_column_major 输入逻辑矩阵（列主序，rows x columns）；Input logical matrix (column-major, rows x columns).
     * @param dst_row_major 输出逻辑矩阵（行主序，rows x columns）；Output logical matrix (row-major, rows x columns).
     * @param rows 逻辑行数；Logical row count.
     * @param columns 逻辑列数；Logical column count.
     * @note 源缓冲区按 row-major(cols x rows) 解释后再转置回 row-major(rows x columns)；Interpret source as row-major(cols x rows), then transpose back.
     */
    void launch_column_to_row_layout_transpose(const double *src_column_major,
                                               double *dst_row_major,
                                               int rows,
                                               int columns)
    {
        const dim3 block(static_cast<unsigned int>(k_transpose_tile_dim),
                         static_cast<unsigned int>(k_transpose_block_rows),
                         1U);
        const dim3 grid(static_cast<unsigned int>((rows + k_transpose_tile_dim - 1) / k_transpose_tile_dim),
                        static_cast<unsigned int>((columns + k_transpose_tile_dim - 1) / k_transpose_tile_dim),
                        1U);
        transpose_row_major_kernel<<<grid, block>>>(src_column_major, dst_row_major, columns, rows);
        JACOBI_CUDA_CHECK(cudaGetLastError());
    }
} // namespace jacobi::svd::detail
