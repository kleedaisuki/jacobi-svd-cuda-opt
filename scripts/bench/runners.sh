#!/usr/bin/env bash

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

run_case_mode_once() {
    local mode="$1"
    local case_dir="$2"
    local input_path="$3"
    local run_index="$4"
    shift 4

    local suffix
    local run_tag
    local output_path
    local log_file

    suffix="$(output_suffix_for_input "${input_path}")"
    run_tag="${mode}_run_${run_index}"
    output_path="${case_dir}/${run_tag}_output${suffix}"
    log_file="${case_dir}/${run_tag}.log"

    case "${mode}" in
        timing)
            run_timing "${case_name}" "${input_path}" "${output_path}" "${log_file}" "$@"
            ;;
        nsys)
            run_nsys "${input_path}" "${output_path}" "${case_dir}/${run_tag}_nsys" "${log_file}" "$@"
            ;;
        ncu-basic)
            run_ncu_basic "${input_path}" "${output_path}" "${case_dir}/${run_tag}_ncu_basic" "${log_file}" "$@"
            ;;
        ncu-deep)
            run_ncu_deep "${input_path}" "${output_path}" "${case_dir}/${run_tag}_ncu_deep" "${log_file}" "$@"
            ;;
    esac
}
