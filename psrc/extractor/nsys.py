"""Nsight Systems report extraction."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .common import ExtractError, extract_json_payload, write_json


def extract_nsys(record: dict[str, Any], mode_dir: Path, out_dir: Path, reports: tuple[str, ...]) -> None:
    """@brief 导出 nsys JSON。Export nsys reports as JSON."""

    input_report = mode_dir / f"run_{record['run_index']}_nsys.sqlite"
    if not input_report.exists():
        input_report = mode_dir / f"run_{record['run_index']}_nsys.nsys-rep"
    if not input_report.exists():
        record["warnings"].append("nsys report artifact is missing")
        return

    exported: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    for report in reports:
        output = out_dir / f"{record['case']}_run_{record['run_index']}_{report}.json"
        try:
            data = run_nsys_stats(input_report, report)
        except ExtractError as exc:
            record["warnings"].append(str(exc))
            continue
        write_json(output, data)
        exported[report] = str(output.relative_to(out_dir.parent))
        summaries[report] = data

    record["extracted"] = {"nsys": exported}
    record["nsys"] = normalize_nsys(summaries)


def run_nsys_stats(input_report: Path, report: str) -> Any:
    """@brief 运行 nsys stats。Run nsys stats and parse JSON output."""

    command = [
        "nsys",
        "stats",
        "--report",
        report,
        "--format",
        "json",
        "--output",
        "-",
        str(input_report),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    payload = extract_json_payload(f"{result.stdout}\n{result.stderr}")
    if result.returncode != 0 and payload is None:
        raise ExtractError(f"nsys stats failed for {input_report.name} report {report}: {result.stderr.strip()}")
    if payload is None:
        raise ExtractError(f"nsys stats produced no JSON for {input_report.name} report {report}")
    return payload


def normalize_nsys(reports: dict[str, Any]) -> dict[str, Any]:
    """@brief 归一化 nsys 报告。Normalize selected nsys report rows."""

    normalized: dict[str, Any] = {}
    cuda_api = reports.get("cuda_api_sum")
    if isinstance(cuda_api, list):
        normalized["cuda_api"] = {row.get("Name", ""): row for row in cuda_api if isinstance(row, dict)}

    for key in ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_gpu_mem_size_sum", "osrt_sum"):
        rows = reports.get(key)
        if isinstance(rows, list):
            normalized[key] = rows
    return normalized
