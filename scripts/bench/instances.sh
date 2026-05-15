#!/usr/bin/env bash

CASE_NAMES=()
CASE_INPUTS=()
CASE_ARGS=()
CASE_NSYS_ARGS=()
CASE_NCU_BASIC_ARGS=()
CASE_NCU_DEEP_ARGS=()

CURRENT_CASE_INDEX=""

add_case() {
    local name="$1"
    local input="$2"
    shift 2

    [[ -n "${name}" ]] || die "add_case requires a non-empty name"
    [[ -n "${input}" ]] || die "add_case ${name} requires an input path"

    CASE_NAMES+=("${name}")
    CASE_INPUTS+=("${input}")
    CASE_ARGS+=("$*")
    CASE_NSYS_ARGS+=("")
    CASE_NCU_BASIC_ARGS+=("")
    CASE_NCU_DEEP_ARGS+=("")
    CURRENT_CASE_INDEX="$((${#CASE_NAMES[@]} - 1))"
}

require_current_case() {
    [[ -n "${CURRENT_CASE_INDEX}" ]] || die "$1 must follow add_case"
}

case_app_args() {
    require_current_case case_app_args
    CASE_ARGS["${CURRENT_CASE_INDEX}"]="$*"
}

case_nsys_args() {
    require_current_case case_nsys_args
    CASE_NSYS_ARGS["${CURRENT_CASE_INDEX}"]="$*"
}

case_ncu_basic_args() {
    require_current_case case_ncu_basic_args
    CASE_NCU_BASIC_ARGS["${CURRENT_CASE_INDEX}"]="$*"
}

case_ncu_deep_args() {
    require_current_case case_ncu_deep_args
    CASE_NCU_DEEP_ARGS["${CURRENT_CASE_INDEX}"]="$*"
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

load_instance() {
    RUNS=1
    CASE_JOBS=1
    MODES=(timing)
    APP_ARGS=()

    # shellcheck source=/dev/null
    source "${INSTANCE_PATH}"
}

apply_overrides() {
    if [[ -n "${RUNS_OVERRIDE}" ]]; then
        RUNS="${RUNS_OVERRIDE}"
    fi
    validate_positive_int RUNS "${RUNS}"

    if [[ -n "${CASE_JOBS_OVERRIDE}" ]]; then
        CASE_JOBS="${CASE_JOBS_OVERRIDE}"
    fi
    validate_positive_int CASE_JOBS "${CASE_JOBS}"

    if [[ -n "${MODES_OVERRIDE}" ]]; then
        split_csv_modes "${MODES_OVERRIDE}"
    fi
}

validate_instance() {
    [[ "${#CASE_NAMES[@]}" -gt 0 ]] || die "instance did not register any cases via add_case"

    for mode in "${MODES[@]}"; do
        case "${mode}" in
            timing | nsys | ncu-basic | ncu-deep) require_tool_for_mode "${mode}" ;;
            *) die "unsupported mode in instance: ${mode}" ;;
        esac
    done

    local case_index
    for case_index in "${!CASE_NAMES[@]}"; do
        if mode_enabled ncu-basic; then
            validate_ncu_filter "case ${CASE_NAMES[case_index]} ncu-basic" "${CASE_NCU_BASIC_ARGS[case_index]}"
        fi
        if mode_enabled ncu-deep; then
            validate_ncu_filter "case ${CASE_NAMES[case_index]} ncu-deep" "${CASE_NCU_DEEP_ARGS[case_index]}"
        fi
    done
}

write_manifest() {
    {
        echo "instance=${INSTANCE_NAME}"
        echo "instance_path=${INSTANCE_PATH}"
        echo "timestamp=${TIMESTAMP}"
        echo "runs=${RUNS}"
        echo "case_jobs=${CASE_JOBS}"
        echo "modes=$(join_by_comma "${MODES[@]}")"
        echo "exe=${EXE_PATH}"
        echo "build_dir=${BUILD_DIR}"
        echo "prof_dir=${PROF_DIR}"
        echo "dry_run=${DRY_RUN}"
    } >"${RUN_ROOT}/manifest.env"
}
