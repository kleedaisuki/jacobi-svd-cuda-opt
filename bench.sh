#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
ORIGINAL_CWD="$(pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
EXE_PATH="${EXE_PATH:-${BUILD_DIR}/jacobi-svd-cuda}"
INSTANCES_DIR="${INSTANCES_DIR:-${REPO_ROOT}/experiments/instances}"
PROF_DIR="${PROF_DIR:-${REPO_ROOT}/experiments/prof}"
cd "${REPO_ROOT}"

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

require_tool_for_mode() {
    local mode="$1"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    case "${mode}" in
        nsys)
            command -v nsys >/dev/null 2>&1 || die "nsys mode requested but nsys was not found in PATH"
            ;;
        ncu-basic | ncu-deep)
            command -v ncu >/dev/null 2>&1 || die "${mode} mode requested but ncu was not found in PATH"
            ;;
    esac
}

CASE_NAMES=()
CASE_INPUTS=()
CASE_ARGS=()

add_case() {
    local name="$1"
    local input="$2"
    shift 2

    [[ -n "${name}" ]] || die "add_case requires a non-empty name"
    [[ -n "${input}" ]] || die "add_case ${name} requires an input path"

    CASE_NAMES+=("${name}")
    CASE_INPUTS+=("${input}")
    CASE_ARGS+=("$*")
}

resolve_instance() {
    local requested="$1"
    if [[ -f "${requested}" ]]; then
        INSTANCE_PATH="$(cd -- "$(dirname -- "${requested}")" && pwd)/$(basename -- "${requested}")"
        INSTANCE_NAME="$(basename -- "${requested}")"
        INSTANCE_NAME="${INSTANCE_NAME%.sh}"
        return
    fi

    if [[ -f "${ORIGINAL_CWD}/${requested}" ]]; then
        INSTANCE_PATH="$(cd -- "$(dirname -- "${ORIGINAL_CWD}/${requested}")" && pwd)/$(basename -- "${requested}")"
        INSTANCE_NAME="$(basename -- "${requested}")"
        INSTANCE_NAME="${INSTANCE_NAME%.sh}"
        return
    fi

    local candidate="${INSTANCES_DIR}/${requested%.sh}.sh"
    [[ -f "${candidate}" ]] || die "instance not found: ${requested} (looked for ${candidate})"
    INSTANCE_PATH="${candidate}"
    INSTANCE_NAME="${requested%.sh}"
}

build_project() {
    log "configuring CMake: ${BUILD_DIR}"
    run_logged "${RUN_ROOT}/build.log" cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"

    log "building target"
    run_logged "${RUN_ROOT}/build.log" cmake --build "${BUILD_DIR}" --target jacobi-svd-cuda

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    [[ -x "${EXE_PATH}" ]] || die "expected executable not found or not executable: ${EXE_PATH}"
}

output_suffix_for_input() {
    local input="$1"
    case "${input##*.}" in
        mat) echo ".mat" ;;
        txt) echo ".txt" ;;
        *) echo ".out" ;;
    esac
}

run_timing() {
    local case_name="$1"
    local input="$2"
    local output="$3"
    local log_file="$4"
    shift 4

    run_logged "${log_file}" /usr/bin/time -f "elapsed_seconds=%e user_seconds=%U sys_seconds=%S max_rss_kb=%M" \
        "${EXE_PATH}" "${input}" "${output}" --force --json-report "$@"
}

run_nsys() {
    local input="$1"
    local output="$2"
    local report_prefix="$3"
    local log_file="$4"
    shift 4

    run_logged "${log_file}" nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
        --stats=true -o "${report_prefix}" "${EXE_PATH}" "${input}" "${output}" --force --quiet "$@"
}

run_ncu_basic() {
    local input="$1"
    local output="$2"
    local report_file="$3"
    local log_file="$4"
    shift 4

    run_logged "${log_file}" ncu --force-overwrite --set basic --export "${report_file}" \
        "${EXE_PATH}" "${input}" "${output}" --force --quiet "$@"
}

run_ncu_deep() {
    local input="$1"
    local output="$2"
    local report_file="$3"
    local log_file="$4"
    shift 4

    run_logged "${log_file}" ncu --force-overwrite --set full --export "${report_file}" \
        "${EXE_PATH}" "${input}" "${output}" --force --quiet "$@"
}

RUNS_OVERRIDE=""
MODES_OVERRIDE=""
SKIP_BUILD=0
DRY_RUN=0
INSTANCE_REQUEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)
            [[ $# -ge 2 ]] || die "--runs requires a value"
            RUNS_OVERRIDE="$2"
            shift 2
            ;;
        --modes)
            [[ $# -ge 2 ]] || die "--modes requires a value"
            MODES_OVERRIDE="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "--build-dir requires a value"
            BUILD_DIR="$2"
            EXE_PATH="${BUILD_DIR}/jacobi-svd-cuda"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -n "${INSTANCE_REQUEST}" ]]; then
                die "unexpected positional argument: $1"
            fi
            INSTANCE_REQUEST="$1"
            shift
            ;;
    esac
done

[[ -n "${INSTANCE_REQUEST}" ]] || {
    usage
    exit 2
}

resolve_instance "${INSTANCE_REQUEST}"

RUNS=1
MODES=(timing)
APP_ARGS=()

# shellcheck source=/dev/null
source "${INSTANCE_PATH}"

if [[ -n "${RUNS_OVERRIDE}" ]]; then
    RUNS="${RUNS_OVERRIDE}"
fi
[[ "${RUNS}" =~ ^[1-9][0-9]*$ ]] || die "RUNS must be a positive integer"

if [[ -n "${MODES_OVERRIDE}" ]]; then
    split_csv_modes "${MODES_OVERRIDE}"
fi

[[ "${#CASE_NAMES[@]}" -gt 0 ]] || die "instance did not register any cases via add_case"

for mode in "${MODES[@]}"; do
    case "${mode}" in
        timing | nsys | ncu-basic | ncu-deep) require_tool_for_mode "${mode}" ;;
        *) die "unsupported mode in instance: ${mode}" ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROF_DIR}/${INSTANCE_NAME}-${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"

{
    echo "instance=${INSTANCE_NAME}"
    echo "instance_path=${INSTANCE_PATH}"
    echo "timestamp=${TIMESTAMP}"
    echo "runs=${RUNS}"
    echo "modes=$(join_by_comma "${MODES[@]}")"
    echo "exe=${EXE_PATH}"
    echo "build_dir=${BUILD_DIR}"
    echo "prof_dir=${PROF_DIR}"
    echo "dry_run=${DRY_RUN}"
} >"${RUN_ROOT}/manifest.env"

if [[ "${SKIP_BUILD}" == "0" ]]; then
    build_project
else
    if [[ "${DRY_RUN}" != "1" ]]; then
        [[ -x "${EXE_PATH}" ]] || die "--skip-build used but executable not found: ${EXE_PATH}"
    fi
fi

for case_index in "${!CASE_NAMES[@]}"; do
    case_name="${CASE_NAMES[case_index]}"
    input_path="${CASE_INPUTS[case_index]}"
    case_extra="${CASE_ARGS[case_index]}"

    [[ -f "${input_path}" ]] || die "input for case ${case_name} not found: ${input_path}"

    read -r -a case_extra_args <<<"${case_extra}"
    case_dir="${RUN_ROOT}/${case_name}"
    mkdir -p "${case_dir}"

    log "case ${case_name}: input=${input_path}"

    for mode in "${MODES[@]}"; do
        for run_index in $(seq 1 "${RUNS}"); do
            suffix="$(output_suffix_for_input "${input_path}")"
            run_tag="${mode}_run_${run_index}"
            output_path="${case_dir}/${run_tag}_output${suffix}"
            log_file="${case_dir}/${run_tag}.log"

            app_args=("${APP_ARGS[@]}" "${case_extra_args[@]}")

            log "mode=${mode} run=${run_index}/${RUNS}"
            case "${mode}" in
                timing)
                    run_timing "${case_name}" "${input_path}" "${output_path}" "${log_file}" "${app_args[@]}"
                    ;;
                nsys)
                    run_nsys "${input_path}" "${output_path}" "${case_dir}/${run_tag}_nsys" "${log_file}" "${app_args[@]}"
                    ;;
                ncu-basic)
                    run_ncu_basic "${input_path}" "${output_path}" "${case_dir}/${run_tag}_ncu_basic" "${log_file}" "${app_args[@]}"
                    ;;
                ncu-deep)
                    run_ncu_deep "${input_path}" "${output_path}" "${case_dir}/${run_tag}_ncu_deep" "${log_file}" "${app_args[@]}"
                    ;;
            esac
        done
    done
done

log "done: ${RUN_ROOT}"
