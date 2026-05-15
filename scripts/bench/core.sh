#!/usr/bin/env bash

usage() {
    cat <<'EOF'
Usage:
  ./bench.sh INSTANCE [OPTIONS]

INSTANCE is resolved as:
  experiments/instances/INSTANCE.sh
  INSTANCE, if it is an existing path

Options:
  --runs N              Override repeat count from the instance script.
  --modes LIST          Comma list: timing,nsys,ncu-basic,ncu-deep.
  --build-dir PATH      CMake build directory. Default: ./build
  --skip-build          Do not run CMake configure/build.
  --dry-run             Print commands without executing them.
  -h, --help            Show this help.

Environment:
  BUILD_DIR             Build directory override.
  EXE_PATH              Executable path override.
  INSTANCES_DIR         Instance directory override.
  PROF_DIR              Profiling result directory override.

Instance script contract:
  RUNS=3
  MODES=(timing nsys ncu-basic ncu-deep)
  APP_ARGS=(--format mat --max-sweeps 128)
  add_case small experiments/cases/mat/small.mat --layout-transpose-mode auto
  add_case big   experiments/cases/mat/big.mat   --layout-transpose-mode on

Results are written under:
  experiments/prof/<instance-name>-<timestamp>/
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

log() {
    printf '[pipeline] %s\n' "$*"
}

join_by_comma() {
    local IFS=,
    echo "$*"
}

split_csv_modes() {
    local raw="$1"
    local item
    IFS=',' read -r -a MODES <<<"${raw}"
    for item in "${MODES[@]}"; do
        case "${item}" in
            timing | nsys | ncu-basic | ncu-deep) ;;
            *) die "unsupported mode: ${item}" ;;
        esac
    done
}

quote_command() {
    printf '%q ' "$@"
}

run_logged() {
    local log_file="$1"
    shift

    {
        echo '$' "$(quote_command "$@")"
    } >>"${log_file}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY-RUN: $(quote_command "$@")"
        return 0
    fi

    "$@" >>"${log_file}" 2>&1
}

output_suffix_for_input() {
    local input="$1"
    case "${input##*.}" in
        mat) echo ".mat" ;;
        txt) echo ".txt" ;;
        *) echo ".out" ;;
    esac
}
