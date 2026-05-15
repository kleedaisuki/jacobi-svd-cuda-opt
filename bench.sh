#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
ORIGINAL_CWD="$(pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
EXE_PATH="${EXE_PATH:-${BUILD_DIR}/jacobi-svd-cuda}"
INSTANCES_DIR="${INSTANCES_DIR:-${REPO_ROOT}/scripts/instances}"
PROF_DIR="${PROF_DIR:-${REPO_ROOT}/experiments/prof}"

# shellcheck source=scripts/bench/core.sh
source "${REPO_ROOT}/scripts/bench/core.sh"
# shellcheck source=scripts/bench/runners.sh
source "${REPO_ROOT}/scripts/bench/runners.sh"
# shellcheck source=scripts/bench/instances.sh
source "${REPO_ROOT}/scripts/bench/instances.sh"

cd "${REPO_ROOT}"

RUNS_OVERRIDE=""
MODES_OVERRIDE=""
CASE_JOBS_OVERRIDE=""
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
        --case-jobs)
            [[ $# -ge 2 ]] || die "--case-jobs requires a value"
            CASE_JOBS_OVERRIDE="$2"
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
load_instance
apply_overrides
validate_instance

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PROF_DIR}/${INSTANCE_NAME}-${TIMESTAMP}"
mkdir -p "${RUN_ROOT}"
write_manifest

if [[ "${SKIP_BUILD}" == "0" ]]; then
    build_project
elif [[ "${DRY_RUN}" != "1" ]]; then
    [[ -x "${EXE_PATH}" ]] || die "--skip-build used but executable not found: ${EXE_PATH}"
fi

run_cases

log "done: ${RUN_ROOT}"
