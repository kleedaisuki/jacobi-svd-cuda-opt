#!/usr/bin/env bash

# Pipeline thread-pool profiling sweep for the fused cooperative-kernel path.
# 面向融合 cooperative kernel 路径的 pipeline 线程池剖析扫描。

RUNS=1
CASE_JOBS=1
MODES=(timing nsys)

APP_ARGS=(
    --max-sweeps 32
    --threads-per-block 256
)

add_case mat_square_medium_auto_q1 experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 1
case_nsys_args --sample=none

add_case mat_square_medium_auto_q2 experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 2
case_nsys_args --sample=none

add_case mat_square_medium_auto_q4 experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 4
case_nsys_args --sample=none

add_case mat_square_medium_auto_q8 experiments/cases/baseline/mat/square_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 8
case_nsys_args --sample=none

add_case mat_tall_skinny_medium_auto_q1 experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 1
case_nsys_args --sample=none

add_case mat_tall_skinny_medium_auto_q2 experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 2
case_nsys_args --sample=none

add_case mat_tall_skinny_medium_auto_q4 experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 4
case_nsys_args --sample=none

add_case mat_tall_skinny_medium_auto_q8 experiments/cases/baseline/mat/tall_skinny_medium.mat --format mat --layout-transpose-mode auto --queue-capacity 8
case_nsys_args --sample=none

add_case mat_sparse_auto_q1 experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto --queue-capacity 1
case_nsys_args --sample=none

add_case mat_sparse_auto_q2 experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto --queue-capacity 2
case_nsys_args --sample=none

add_case mat_sparse_auto_q4 experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto --queue-capacity 4
case_nsys_args --sample=none

add_case mat_sparse_auto_q8 experiments/cases/baseline/mat/sparse.mat --format mat --layout-transpose-mode auto --queue-capacity 8
case_nsys_args --sample=none

add_case mat_ill_conditioned_auto_q1 experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto --queue-capacity 1
case_nsys_args --sample=none

add_case mat_ill_conditioned_auto_q2 experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto --queue-capacity 2
case_nsys_args --sample=none

add_case mat_ill_conditioned_auto_q4 experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto --queue-capacity 4
case_nsys_args --sample=none

add_case mat_ill_conditioned_auto_q8 experiments/cases/baseline/mat/ill_conditioned.mat --format mat --layout-transpose-mode auto --queue-capacity 8
case_nsys_args --sample=none
