"""Command-line interface for profiler extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import ExtractError
from .constants import NSYS_REPORTS
from .runner import extract_run, resolve_source


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """@brief 解析命令行参数。Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="extract",
        description="Extract bench.sh Nsight outputs from experiments/prof to experiments/perf.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="latest",
        help="prof run directory, run name, or 'latest' (default: latest)",
    )
    parser.add_argument("--prof-dir", default="experiments/prof", help="profiling result root")
    parser.add_argument("--perf-dir", default="experiments/perf", help="extracted result root")
    parser.add_argument("--force", action="store_true", help="overwrite an existing perf run directory")
    parser.add_argument("--skip-nsys", action="store_true", help="skip nsys stats extraction")
    parser.add_argument("--skip-ncu", action="store_true", help="skip ncu report extraction")
    parser.add_argument(
        "--ncu-page",
        choices=("raw", "details"),
        default="raw",
        help="ncu page to export as CSV (default: raw)",
    )
    parser.add_argument(
        "--nsys-report",
        action="append",
        choices=NSYS_REPORTS,
        help="nsys report to export; may be repeated (default: common reports)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """@brief CLI 入口。CLI entry point."""

    args = parse_args(argv)
    repo_root = Path.cwd()
    prof_dir = (repo_root / args.prof_dir).resolve()
    perf_dir = (repo_root / args.perf_dir).resolve()

    try:
        source_dir = resolve_source(args.source, prof_dir)
        extract_run(
            source_dir=source_dir,
            output_dir=perf_dir / source_dir.name,
            force=args.force,
            skip_nsys=args.skip_nsys,
            skip_ncu=args.skip_ncu,
            nsys_reports=tuple(args.nsys_report or NSYS_REPORTS),
            ncu_page=args.ncu_page,
        )
    except ExtractError as exc:
        print(f"extract: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
