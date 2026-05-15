"""Shared helpers for profiler extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ExtractError(RuntimeError):
    """@brief 提取错误。Extraction failure reported to the CLI."""


def read_text(path: Path) -> str:
    """@brief 读取文本。Read text with replacement for malformed bytes."""

    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    """@brief 写 JSON。Write pretty JSON."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """@brief 写 JSONL。Write JSON lines."""

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def extract_json_payload(text: str) -> Any | None:
    """@brief 提取 JSON payload。Extract the first valid JSON object or array from text."""

    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return None

    decoder = json.JSONDecoder()
    start = min(starts)
    while start < len(text):
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            next_starts = [index for index in (text.find("[", start + 1), text.find("{", start + 1)) if index >= 0]
            if not next_starts:
                return None
            start = min(next_starts)
    return None


def coerce_value(value: str | None) -> Any:
    """@brief 转换 CSV 值。Coerce CSV scalars when possible."""

    if value is None or value == "":
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def float_or_zero(value: str | None) -> float:
    """@brief 转为浮点数。Convert a value to float or zero."""

    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0
