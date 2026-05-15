"""Log parsing and artifact discovery."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_text
from .constants import NVIDIA_PROGRESS_RE, PROFILING_RE, REPORT_RE, RUN_LOG_RE


def build_record(log_file: Path, source_dir: Path) -> dict[str, Any]:
    """@brief 构建运行记录。Build one per-log run record."""

    mode = log_file.parent.name
    case = log_file.parent.parent.name
    match = RUN_LOG_RE.match(log_file.name)
    run_index = int(match.group("index")) if match else None
    log_info = parse_log(read_text(log_file))
    artifacts = find_artifacts(log_file.parent, mode, run_index)

    return {
        "case": case,
        "mode": mode,
        "run_index": run_index,
        "status": status_from_log(log_info, artifacts),
        "source_log": str(log_file.relative_to(source_dir)),
        "command": log_info["command"],
        "artifacts": {key: str(path.relative_to(source_dir)) for key, path in artifacts.items()},
        "warnings": log_info["warnings"],
        "errors": log_info["errors"],
        "profiled_kernels": log_info["profiled_kernels"],
        "generated": log_info["generated"],
        "report_paths": log_info["report_paths"],
    }


def parse_log(text: str) -> dict[str, Any]:
    """@brief 解析 profiler 日志。Parse useful facts from a rough profiler log."""

    command = None
    warnings: list[str] = []
    errors: list[str] = []
    generated: list[str] = []
    report_paths: list[str] = []
    profiled = Counter()
    passes = Counter()

    for raw_line in text.splitlines():
        line = NVIDIA_PROGRESS_RE.sub("", raw_line).strip()
        if not line:
            continue
        if command is None and line.startswith("$ "):
            command = line[2:].strip()
            continue
        if line.startswith("SKIPPED:") or "==WARNING==" in line:
            warnings.append(line)
            continue
        if "==ERROR==" in line or line.lower().startswith("error:"):
            errors.append(line)
            continue

        match = PROFILING_RE.search(line)
        if match:
            kernel, _, pass_count = match.groups()
            profiled[kernel] += 1
            passes[kernel] += int(pass_count)
            continue

        match = REPORT_RE.search(line)
        if match:
            report_paths.append(match.group("path").strip())
            continue
        if line.startswith("/") and (line.endswith(".nsys-rep") or line.endswith(".sqlite")):
            generated.append(line)

    return {
        "command": command,
        "warnings": sorted(set(warnings)),
        "errors": sorted(set(errors)),
        "generated": generated,
        "report_paths": report_paths,
        "profiled_kernels": {
            kernel: {"launches": count, "passes": passes[kernel]} for kernel, count in sorted(profiled.items())
        },
    }


def status_from_log(log_info: dict[str, Any], artifacts: dict[str, Path]) -> str:
    """@brief 推断运行状态。Infer run status from log and artifacts."""

    if log_info["errors"]:
        return "partial" if artifacts else "failed"
    return "ok"


def find_artifacts(mode_dir: Path, mode: str, run_index: int | None) -> dict[str, Path]:
    """@brief 查找 profiler artifact。Find report artifacts near a log."""

    artifacts: dict[str, Path] = {}
    if run_index is None:
        return artifacts

    tag = f"run_{run_index}"
    if mode == "nsys":
        add_if_exists(artifacts, "nsys_rep", mode_dir / f"{tag}_nsys.nsys-rep")
        add_if_exists(artifacts, "nsys_sqlite", mode_dir / f"{tag}_nsys.sqlite")
    elif mode == "ncu-basic":
        add_if_exists(artifacts, "ncu_rep", mode_dir / f"{tag}_ncu_basic.ncu-rep")
    elif mode == "ncu-deep":
        add_if_exists(artifacts, "ncu_rep", mode_dir / f"{tag}_ncu_deep.ncu-rep")
    return artifacts


def add_if_exists(mapping: dict[str, Path], key: str, path: Path) -> None:
    """@brief 条件加入路径。Add path to mapping when it exists."""

    if path.exists():
        mapping[key] = path
