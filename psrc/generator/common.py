"""Shared testcase generation utilities."""

from __future__ import annotations

import argparse
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Matrix:
    """Row-major dense matrix."""

    rows: int
    columns: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class GenerateOptions:
    """Generator runtime options."""

    output: Path
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    count_per_shape: int
    seed: int
    min_value: float
    max_value: float
    allow_wide: bool
    overwrite: bool


def parse_positive_int(raw: str) -> int:
    """Parse a positive integer."""

    try:
        value = int(raw, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {raw}") from exc

    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {raw}")
    return value


def parse_int_range(raw: str) -> tuple[int, ...]:
    """Parse N or START:STOP[:STEP] as an inclusive positive integer range."""

    parts = raw.split(":")
    if len(parts) == 1:
        return (parse_positive_int(parts[0]),)
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(f"invalid range syntax: {raw}")

    start = parse_positive_int(parts[0])
    stop = parse_positive_int(parts[1])
    step = parse_positive_int(parts[2]) if len(parts) == 3 else 1

    if stop < start:
        raise argparse.ArgumentTypeError(f"range stop must be >= start: {raw}")

    return tuple(range(start, stop + 1, step))


def add_common_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
    """Register common generator arguments."""

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output matrix stream path. Default: {default_output}",
    )
    parser.add_argument(
        "--rows",
        type=parse_int_range,
        default=parse_int_range("8:64:8"),
        help="Row counts as N or START:STOP[:STEP]. Inclusive range. Default: 8:64:8",
    )
    parser.add_argument(
        "--columns",
        "--cols",
        dest="columns",
        type=parse_int_range,
        default=parse_int_range("4:32:4"),
        help="Column counts as N or START:STOP[:STEP]. Inclusive range. Default: 4:32:4",
    )
    parser.add_argument(
        "--count-per-shape",
        type=parse_positive_int,
        default=1,
        help="Number of matrices generated for each valid (rows, columns) shape. Default: 1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260515,
        help="Base pseudo-random seed. Default: 20260515",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=-1.0,
        help="Minimum random value. Default: -1.0",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=1.0,
        help="Maximum random value. Default: 1.0",
    )
    parser.add_argument(
        "--allow-wide",
        action="store_true",
        help="Generate rows < columns cases too. The CUDA SVD currently rejects them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file.",
    )


def options_from_args(args: argparse.Namespace) -> GenerateOptions:
    """Build typed generator options from argparse output."""

    if args.max_value < args.min_value:
        raise ValueError("--max-value must be >= --min-value.")

    return GenerateOptions(
        output=args.output,
        rows=tuple(args.rows),
        columns=tuple(args.columns),
        count_per_shape=args.count_per_shape,
        seed=args.seed,
        min_value=args.min_value,
        max_value=args.max_value,
        allow_wide=args.allow_wide,
        overwrite=args.overwrite,
    )


def prepare_output_path(path: Path, overwrite: bool) -> None:
    """Create parent directory and guard existing output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path} (use --overwrite)")


def iter_matrices(options: GenerateOptions) -> Iterable[Matrix]:
    """Yield deterministic dense matrices for every selected shape."""

    case_index = 0
    for rows in options.rows:
        for columns in options.columns:
            if rows < columns and not options.allow_wide:
                continue

            for duplicate_index in range(options.count_per_shape):
                rng = random.Random(options.seed + case_index)
                values = []
                for row in range(rows):
                    for column in range(columns):
                        random_part = rng.uniform(options.min_value, options.max_value)
                        smooth_part = math.sin(0.013 * (row + 1) * (column + 1))
                        smooth_part += math.cos(0.017 * (row + 3) * (column + 5))
                        diagonal_boost = 2.0 if row == column else 0.0
                        duplicate_shift = 0.001 * duplicate_index
                        values.append(random_part + smooth_part + diagonal_boost + duplicate_shift)

                yield Matrix(rows=rows, columns=columns, values=tuple(values))
                case_index += 1


def write_mat_stream(path: Path, matrices: Iterable[Matrix]) -> int:
    """Write a .mat matrix stream compatible with the C++ reader."""

    count = 0
    with path.open("wb") as output:
        for matrix in matrices:
            output.write(struct.pack(">QQ", matrix.rows, matrix.columns))
            output.write(struct.pack(f">{len(matrix.values)}d", *matrix.values))
            count += 1
    return count


def write_txt_stream(path: Path, matrices: Iterable[Matrix]) -> int:
    """Write a text matrix stream compatible with the C++ reader."""

    count = 0
    with path.open("w", encoding="ascii", newline="\n") as output:
        for matrix in matrices:
            if count > 0:
                output.write("\n")

            for row in range(matrix.rows):
                start = row * matrix.columns
                end = start + matrix.columns
                output.write(" ".join(f"{value:.17g}" for value in matrix.values[start:end]))
                output.write("\n")

            count += 1
    return count
