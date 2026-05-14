#pragma once

namespace jacobi::svd::detail
{
    /**
     * @brief 行主序索引映射；Row-major index mapping.
     * @param row 行索引；Row index.
     * @param col 列索引；Column index.
     * @param columns 总列数；Total columns.
     * @return 一维偏移；Linear offset.
     */
    __host__ __device__ inline int row_major_index(int row, int col, int columns)
    {
        return row * columns + col;
    }

    /**
     * @brief 列主序索引映射；Column-major index mapping.
     * @param row 行索引；Row index.
     * @param col 列索引；Column index.
     * @param rows 总行数；Total rows.
     * @return 一维偏移；Linear offset.
     */
    __host__ __device__ inline int column_major_index(int row, int col, int rows)
    {
        return col * rows + row;
    }

    /**
     * @brief 按策略返回矩阵索引；Return matrix index by layout policy.
     * @tparam ColumnMajor 是否列主序；Whether layout is column-major.
     * @param row 行索引；Row index.
     * @param col 列索引；Column index.
     * @param rows 总行数；Total rows.
     * @param columns 总列数；Total columns.
     * @return 一维偏移；Linear offset.
     */
    template <bool ColumnMajor>
    __host__ __device__ inline int matrix_index(int row, int col, int rows, int columns)
    {
        if constexpr (ColumnMajor)
        {
            return column_major_index(row, col, rows);
        }
        return row_major_index(row, col, columns);
    }
} // namespace jacobi::svd::detail
