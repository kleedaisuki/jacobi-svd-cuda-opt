#!/usr/bin/env bash

# Fused cooperative-kernel timing sweep using the baseline case matrix.
# 使用 baseline 用例矩阵的融合 cooperative kernel timing 扫描。

RUNS=3
CASE_JOBS=1
MODES=(timing)

APP_ARGS=(
    --max-sweeps 32
    --threads-per-block 256
    --queue-capacity 2
)

add_case mat_grid_small_auto experiments/cases/baseline/mat/grid_small.mat --format mat --layout-transpose-mode auto
add_case mat_square_medium_auto experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto
add_case mat_square_medium_off experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode off
add_case mat_square_medium_on experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode on
add_case mat_tall_skinny_medium_auto experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto
add_case mat_tall_skinny_medium_off experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode off
add_case mat_tall_skinny_medium_on experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode on
add_case mat_ill_conditioned_auto experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto
add_case mat_low_rank_auto experiments/cases/baseline/mat/low_rank.mat --format mat --layout-transpose-mode auto
add_case mat_sparse_auto experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto
add_case mat_zero_columns_auto experiments/cases/baseline/mat/zero_columns.mat --format mat --layout-transpose-mode auto
add_case txt_grid_small_auto experiments/cases/baseline/txt/grid_small.txt --format txt --layout-transpose-mode auto
add_case txt_ill_conditioned_small_auto experiments/cases/baseline/txt/ill_conditioned_small.txt --format txt --layout-transpose-mode auto
add_case txt_sparse_small_auto experiments/cases/baseline/txt/sparse_small.txt --format txt --layout-transpose-mode auto
