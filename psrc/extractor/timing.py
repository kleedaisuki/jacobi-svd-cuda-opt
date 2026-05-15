"""Timing-log extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import extract_json_payload, read_text
from .constants import TIME_RE


def extract_timing(record: dict[str, Any], log_file: Path) -> None:
    """@brief 提取 timing 日志。Extract timing log metrics."""

    text = read_text(log_file)
    match = TIME_RE.search(text)
    if match:
        record["time"] = {
            "elapsed_seconds": float(match.group("elapsed")),
            "user_seconds": float(match.group("user")),
            "sys_seconds": float(match.group("sys")),
            "max_rss_kb": int(match.group("rss")),
        }

    payload = extract_json_payload(text)
    if payload is not None:
        record["app_report"] = payload
