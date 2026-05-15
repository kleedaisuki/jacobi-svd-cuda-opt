"""Nsight Compute report extraction."""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import ExtractError, coerce_value, float_or_zero, write_json
from .constants import NCU_METRICS


def extract_ncu(record: dict[str, Any], mode_dir: Path, out_dir: Path, page: str) -> None:
    """@brief 导出 ncu CSV。Export ncu report as CSV and normalize selected metrics."""

    if record["mode"] == "ncu-basic":
        input_report = mode_dir / f"run_{record['run_index']}_ncu_basic.ncu-rep"
    else:
        input_report = mode_dir / f"run_{record['run_index']}_ncu_deep.ncu-rep"

    if not input_report.exists():
        record["warnings"].append("ncu report artifact is missing")
        return

    csv_output = out_dir / f"{record['case']}_{record['mode']}_run_{record['run_index']}_{page}.csv"
    metrics_output = out_dir / f"{record['case']}_{record['mode']}_run_{record['run_index']}_metrics.json"
    try:
        run_ncu_export(input_report, page, csv_output)
        metrics = normalize_ncu_csv(csv_output)
    except ExtractError as exc:
        record["warnings"].append(str(exc))
        return

    write_json(metrics_output, metrics)
    record["extracted"] = {
        "ncu": {
            "csv": str(csv_output.relative_to(out_dir.parent)),
            "metrics": str(metrics_output.relative_to(out_dir.parent)),
        }
    }
    record["ncu"] = metrics


def run_ncu_export(input_report: Path, page: str, output: Path) -> None:
    """@brief 运行 ncu 导出。Run ncu import and save CSV output."""

    command = [
        "ncu",
        "--import",
        str(input_report),
        "--csv",
        "--page",
        page,
        "--print-units",
        "base",
        "--print-fp",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 and not result.stdout.strip():
        raise ExtractError(f"ncu import failed for {input_report.name}: {result.stderr.strip()}")
    output.write_text(result.stdout, encoding="utf-8")


def normalize_ncu_csv(path: Path) -> dict[str, Any]:
    """@brief 归一化 ncu CSV。Normalize selected ncu metric columns."""

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))

    selected_rows: list[dict[str, Any]] = []
    by_kernel: dict[str, dict[str, Any]] = defaultdict(lambda: {"launches": 0, "gpu_time_duration_sum": 0.0})
    present_metrics = [metric for metric in NCU_METRICS if rows and metric in rows[0]]

    for row in rows:
        selected_rows.append({metric: coerce_value(row.get(metric, "")) for metric in present_metrics})

        kernel = str(row.get("Kernel Name") or row.get("launch__kernel_name") or "")
        if kernel:
            by_kernel[kernel]["launches"] += 1
            by_kernel[kernel]["gpu_time_duration_sum"] += float_or_zero(row.get("gpu__time_duration.sum"))

    return {
        "row_count": len(rows),
        "present_metrics": present_metrics,
        "kernels": selected_rows,
        "by_kernel": dict(sorted(by_kernel.items())),
    }
