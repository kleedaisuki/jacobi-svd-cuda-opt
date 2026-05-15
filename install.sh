#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${PYTHON:-python3}"
venv_dir="${VENV_DIR:-${repo_root}/.venv}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "error: Python interpreter not found: ${python_bin}" >&2
    echo "hint: set PYTHON=/path/to/python3.10+ and rerun this script." >&2
    exit 1
fi

python_version="$("${python_bin}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

"${python_bin}" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("error: Python 3.10 or newer is required.")
PY

echo "Using Python ${python_version}: ${python_bin}"
echo "Virtual environment: ${venv_dir}"

if [[ ! -d "${venv_dir}" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
fi

venv_python="${venv_dir}/bin/python"
if [[ ! -x "${venv_python}" ]]; then
    echo "error: virtual environment Python not found: ${venv_python}" >&2
    exit 1
fi

"${venv_python}" -m pip install --upgrade pip setuptools wheel
"${venv_python}" -m pip install --editable "${repo_root}"

echo
echo "Python environment is ready."
echo "Activate it with:"
echo "  source \"${venv_dir}/bin/activate\""
