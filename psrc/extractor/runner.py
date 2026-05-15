"""Top-level extraction workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import ExtractError, write_json, write_jsonl
from .constants import RUN_DIR_RE
from .logs import build_record
from .ncu import extract_ncu
from .nsys import extract_nsys
from .summary import summarize_records, write_report, write_summary_csv
from .timing import extract_timing


def resolve_source(source: str, prof_dir: Path) -> Path:
    """@brief 解析输入运行目录。Resolve the requested profiling run directory."""

    if source == "latest":
        candidates = [path for path in prof_dir.iterdir() if path.is_dir() and RUN_DIR_RE.match(path.name)]
        if not candidates:
            raise ExtractError(f"no profiling runs found under {prof_dir}")
        return max(candidates, key=lambda path: RUN_DIR_RE.match(path.name).group("timestamp"))  # type: ignore[union-attr]

    path = Path(source)
    if path.is_dir():
        return path.resolve()

    candidate = prof_dir / source
    if candidate.is_dir():
        return candidate.resolve()

    raise ExtractError(f"profiling run not found: {source}")


def extract_run(
    *,
    source_dir: Path,
    output_dir: Path,
    force: bool,
    skip_nsys: bool,
    skip_ncu: bool,
    nsys_reports: tuple[str, ...],
    ncu_page: str,
) -> None:
    """@brief 提取一次 bench 运行。Extract one bench.sh run."""

    if output_dir.exists():
        if not force:
            raise ExtractError(f"output already exists: {output_dir} (use --force)")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True)
    (output_dir / "nsys").mkdir()
    (output_dir / "ncu").mkdir()

    manifest = read_manifest(source_dir / "manifest.env")
    manifest_json = {"source_dir": str(source_dir), "run_name": source_dir.name, **parse_run_name(source_dir.name), **manifest}
    write_json(output_dir / "manifest.json", manifest_json)

    records = collect_records(source_dir, output_dir, skip_nsys, skip_ncu, nsys_reports, ncu_page)
    write_jsonl(output_dir / "runs.jsonl", records)
    write_json(output_dir / "summary.json", summarize_records(records, manifest_json))
    write_summary_csv(output_dir / "summary.csv", records, manifest_json)
    write_report(output_dir / "report.md", records, manifest_json)


def collect_records(
    source_dir: Path,
    output_dir: Path,
    skip_nsys: bool,
    skip_ncu: bool,
    nsys_reports: tuple[str, ...],
    ncu_page: str,
) -> list[dict[str, Any]]:
    """@brief 收集运行记录。Collect and enrich all per-log records."""

    records: list[dict[str, Any]] = []
    for log_file in sorted(source_dir.glob("*/*/run_*.log")):
        record = build_record(log_file, source_dir)
        if record["mode"] == "nsys" and not skip_nsys:
            extract_nsys(record, log_file.parent, output_dir / "nsys", nsys_reports)
        elif record["mode"].startswith("ncu-") and not skip_ncu:
            extract_ncu(record, log_file.parent, output_dir / "ncu", ncu_page)
        elif record["mode"] == "timing":
            extract_timing(record, log_file)
        records.append(record)
    return records


def parse_run_name(name: str) -> dict[str, str]:
    """@brief 解析运行名。Parse <instance>-<timestamp>."""

    match = RUN_DIR_RE.match(name)
    if not match:
        return {}
    return match.groupdict()


def read_manifest(path: Path) -> dict[str, str]:
    """@brief 读取 manifest.env。Read a simple key=value manifest."""

    if not path.exists():
        return {}

    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key] = value
    return manifest
