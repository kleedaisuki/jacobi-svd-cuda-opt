#!/usr/bin/env bash

# Sample experiment instance for ./bench.sh.
#
# Usage:
#   ./bench.sh sample --dry-run
#   ./bench.sh sample --modes timing --runs 3
#
# The root pipeline script provides add_case(name, input, extra app args...).

RUNS=1
CASE_JOBS=3
MODES=(timing nsys ncu-basic ncu-deep)

APP_ARGS=(
    --format txt
    --max-sweeps 16
    --threads-per-block 256
    --queue-capacity 2
)

add_case \
    txt_auto \
    experiments/cases/txt/generated.txt \
    --layout-transpose-mode auto
case_nsys_args \
    --sample=none
case_ncu_basic_args \
    --kernel-name regex:pair_stats_kernel \
    --launch-count 2
case_ncu_deep_args \
    --kernel-name regex:apply_rotation_kernel \
    --launch-count 1

add_case \
    txt_transpose_off \
    experiments/cases/txt/generated.txt \
    --layout-transpose-mode off
case_nsys_args \
    --sample=none
case_ncu_basic_args \
    --kernel-name regex:pair_stats_kernel \
    --launch-count 2
case_ncu_deep_args \
    --kernel-name regex:apply_rotation_kernel \
    --launch-count 1

add_case \
    txt_transpose_on \
    experiments/cases/txt/generated.txt \
    --layout-transpose-mode on
case_nsys_args \
    --sample=none
case_ncu_basic_args \
    --kernel-name regex:pair_stats_kernel \
    --launch-count 2
case_ncu_deep_args \
    --kernel-name regex:apply_rotation_kernel \
    --launch-count 1
