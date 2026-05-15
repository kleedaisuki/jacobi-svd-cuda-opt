"""Summary writers for extracted profiler records."""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def summarize_records(records: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """@brief 汇总运行记录。Summarize extracted records."""

    warnings = sum((record.get("warnings", []) for record in records), [])
    errors = sum((record.get("errors", []) for record in records), [])
    timing_elapsed = [
        record["time"]["elapsed_seconds"]
        for record in records
        if isinstance(record.get("time"), dict) and "elapsed_seconds" in record["time"]
    ]

    summary: dict[str, Any] = {
        "manifest": manifest,
        "record_count": len(records),
        "by_mode": dict(sorted(Counter(record["mode"] for record in records).items())),
        "by_status": dict(sorted(Counter(record["status"] for record in records).items())),
        "warning_count": len(warnings),
        "error_count": len(errors),
    }
    if timing_elapsed:
        summary["timing_elapsed_seconds"] = numeric_summary(timing_elapsed)
    return summary


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    """@brief 数值汇总。Summarize numeric values."""

    items = list(values)
    result = {
        "count": float(len(items)),
        "min": min(items),
        "max": max(items),
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
    }
    if len(items) > 1:
        result["stdev"] = statistics.stdev(items)
    return result


def write_summary_csv(path: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """@brief 写 summary.csv。Write a flat CSV summary."""

    fields = [
        "instance",
        "timestamp",
        "case",
        "mode",
        "run_index",
        "status",
        "elapsed_seconds",
        "max_rss_kb",
        "cudaMemcpy_total_ns",
        "cudaMemcpy_calls",
        "cudaLaunchKernel_total_ns",
        "cudaLaunchKernel_calls",
        "ncu_row_count",
        "ncu_kernel_count",
        "warning_count",
        "error_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            cuda_api = record.get("nsys", {}).get("cuda_api", {})
            cuda_memcpy = cuda_api.get("cudaMemcpy", {})
            cuda_launch = cuda_api.get("cudaLaunchKernel", {})
            ncu = record.get("ncu", {})
            writer.writerow(
                {
                    "instance": manifest.get("instance", ""),
                    "timestamp": manifest.get("timestamp", ""),
                    "case": record["case"],
                    "mode": record["mode"],
                    "run_index": record["run_index"],
                    "status": record["status"],
                    "elapsed_seconds": record.get("time", {}).get("elapsed_seconds", ""),
                    "max_rss_kb": record.get("time", {}).get("max_rss_kb", ""),
                    "cudaMemcpy_total_ns": cuda_memcpy.get("Total Time (ns)", ""),
                    "cudaMemcpy_calls": cuda_memcpy.get("Num Calls", ""),
                    "cudaLaunchKernel_total_ns": cuda_launch.get("Total Time (ns)", ""),
                    "cudaLaunchKernel_calls": cuda_launch.get("Num Calls", ""),
                    "ncu_row_count": ncu.get("row_count", ""),
                    "ncu_kernel_count": len(ncu.get("by_kernel", {})) if isinstance(ncu, dict) else "",
                    "warning_count": len(record.get("warnings", [])),
                    "error_count": len(record.get("errors", [])),
                }
            )


def write_report(path: Path, records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """@brief 写 Markdown 报告。Write a compact Markdown report."""

    lines = [
        f"# Perf Extraction: {manifest.get('run_name', '')}",
        "",
        "## Metadata",
        "",
        f"- instance: `{manifest.get('instance', '')}`",
        f"- timestamp: `{manifest.get('timestamp', '')}`",
        f"- source: `{manifest.get('source_dir', '')}`",
        "",
        "## Records",
        "",
        "| case | mode | run | status | warnings | errors |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| {record['case']} | {record['mode']} | {record['run_index']} | "
            f"{record['status']} | {len(record.get('warnings', []))} | {len(record.get('errors', []))} |"
        )
    lines.extend(["", "## Outputs", "", "- `runs.jsonl`: normalized per-run records"])
    lines.extend(["- `summary.csv`: flat table for scripts and spreadsheets"])
    lines.extend(["- `summary.json`: aggregate counts and numeric summaries"])
    lines.extend(["- `nsys/`: JSON exports from `nsys stats`"])
    lines.extend(["- `ncu/`: CSV exports and selected metric JSON files"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
