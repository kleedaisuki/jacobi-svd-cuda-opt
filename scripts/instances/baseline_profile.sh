#!/usr/bin/env bash

# Baseline Nsight profiler sweep generated for machine-readable extraction.
# 生成可由 extract 转换为机器可读数据的 baseline Nsight 扫描用例。

RUNS=1
CASE_JOBS=1
MODES=(nsys ncu-basic ncu-deep)

APP_ARGS=(
    --max-sweeps 32
    --threads-per-block 256
    --queue-capacity 2
)

add_case mat_grid_small_auto experiments/cases/baseline/mat/grid_small.mat --format mat --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_square_medium_auto experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_square_medium_off experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode off
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_square_medium_on experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode on
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_tall_skinny_medium_auto experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_tall_skinny_medium_off experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode off
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_tall_skinny_medium_on experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode on
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_ill_conditioned_auto experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case mat_sparse_auto experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1

add_case txt_grid_small_auto experiments/cases/baseline/txt/grid_small.txt --format txt --layout-transpose-mode auto
case_nsys_args --sample=none
case_ncu_basic_args --kernel-name regex:pair_stats_kernel --launch-count 2
case_ncu_deep_args --kernel-name regex:apply_rotation_kernel --launch-count 1
