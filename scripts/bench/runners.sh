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
    local mode_dir
    local output_path
    local log_file

    suffix="$(output_suffix_for_input "${input_path}")"
    run_tag="run_${run_index}"
    mode_dir="${case_dir}/${mode}"
    mkdir -p "${mode_dir}"
    output_path="${mode_dir}/${run_tag}_output${suffix}"
    log_file="${mode_dir}/${run_tag}.log"

    case "${mode}" in
        timing)
            run_timing "${case_name}" "${input_path}" "${output_path}" "${log_file}" "$@"
            ;;
        nsys)
            run_nsys "${input_path}" "${output_path}" "${mode_dir}/${run_tag}_nsys" "${log_file}" "$@"
            ;;
        ncu-basic)
            run_ncu_basic "${input_path}" "${output_path}" "${mode_dir}/${run_tag}_ncu_basic" "${log_file}" "$@"
            ;;
        ncu-deep)
            run_ncu_deep "${input_path}" "${output_path}" "${mode_dir}/${run_tag}_ncu_deep" "${log_file}" "$@"
            ;;
    esac
}

run_case() {
    local case_index="$1"
    local case_name="${CASE_NAMES[case_index]}"
    local input_path="${CASE_INPUTS[case_index]}"
    local case_extra="${CASE_ARGS[case_index]}"

    [[ -f "${input_path}" ]] || die "input for case ${case_name} not found: ${input_path}"

    local case_extra_args=()
    read -r -a case_extra_args <<<"${case_extra}"

    local case_dir="${RUN_ROOT}/${case_name}"
    mkdir -p "${case_dir}"

    log "case ${case_name}: input=${input_path}"

    local mode
    local run_index
    local app_args
    for mode in "${MODES[@]}"; do
        for run_index in $(seq 1 "${RUNS}"); do
            app_args=("${APP_ARGS[@]}" "${case_extra_args[@]}")
            log "case=${case_name} mode=${mode} run=${run_index}/${RUNS}"
            run_case_mode_once "${mode}" "${case_dir}" "${input_path}" "${run_index}" "${app_args[@]}"
        done
    done
}

wait_for_one_case_job() {
    local finished_pid
    local status

    if wait -n -p finished_pid; then
        status=0
    else
        status=$?
    fi

    if [[ -n "${finished_pid:-}" ]]; then
        unset "CASE_JOB_PIDS[${finished_pid}]"
    fi

    if [[ "${status}" -ne 0 ]]; then
        CASE_JOB_FAILURE=1
        log "case job failed with status ${status}"
    fi
}

run_cases() {
    CASE_JOB_FAILURE=0
    declare -gA CASE_JOB_PIDS=()

    local case_index
    local pid
    for case_index in "${!CASE_NAMES[@]}"; do
        while [[ "${#CASE_JOB_PIDS[@]}" -ge "${CASE_JOBS}" ]]; do
            wait_for_one_case_job
        done

        run_case "${case_index}" &
        pid=$!
        CASE_JOB_PIDS["${pid}"]=1
    done

    while [[ "${#CASE_JOB_PIDS[@]}" -gt 0 ]]; do
        wait_for_one_case_job
    done

    if [[ "${CASE_JOB_FAILURE}" -ne 0 ]]; then
        die "one or more case jobs failed"
    fi
}
