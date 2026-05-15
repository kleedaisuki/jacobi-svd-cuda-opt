"""Shared testcase generation utilities."""

from __future__ import annotations

import argparse
import math
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


Shape = tuple[int, int]
Writer = Callable[[Path, Iterable["Matrix"]], int]


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
    files: int
    cases_per_file: tuple[int, ...]
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    shape_distribution: str
    value_distribution: str
    seed: int
    min_value: float
    max_value: float
    mean: float
    stddev: float
    sparsity: float
    rank: int
    condition: float
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


def parse_probability(raw: str) -> float:
    """Parse a probability in [0, 1]."""

    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid probability: {raw}") from exc

    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"expected probability in [0, 1]: {raw}")
    return value


def parse_int_range(raw: str) -> tuple[int, ...]:
    """Parse N, CSV, or START:STOP[:STEP] as positive integers."""

    if "," in raw:
        values = tuple(parse_positive_int(part.strip()) for part in raw.split(",") if part.strip())
        if not values:
            raise argparse.ArgumentTypeError(f"empty integer list: {raw}")
        return values

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


def parse_cases_per_file(raw: str) -> tuple[int, ...]:
    """Parse cases-per-file as N, CSV, or inclusive START:STOP[:STEP]."""

    return parse_int_range(raw)


def add_common_arguments(parser: argparse.ArgumentParser, default_output: Path) -> None:
    """Register common generator arguments."""

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output file, directory, or filename pattern with {{index}}. Default: {default_output}",
    )
    parser.add_argument(
        "--files",
        type=parse_positive_int,
        default=1,
        help="Number of output files. Default: 1",
    )
    parser.add_argument(
        "--cases-per-file",
        type=parse_cases_per_file,
        default=parse_cases_per_file("32"),
        help="Cases per output file as N, CSV, or START:STOP[:STEP]. Default: 32",
    )
    parser.add_argument(
        "--rows",
        type=parse_int_range,
        default=parse_int_range("8:64:8"),
        help="Candidate row counts as N, CSV, or START:STOP[:STEP]. Default: 8:64:8",
    )
    parser.add_argument(
        "--columns",
        "--cols",
        dest="columns",
        type=parse_int_range,
        default=parse_int_range("4:32:4"),
        help="Candidate column counts as N, CSV, or START:STOP[:STEP]. Default: 4:32:4",
    )
    parser.add_argument(
        "--shape-distribution",
        choices=("grid", "uniform", "log-uniform", "square", "tall-skinny", "wide"),
        default="grid",
        help="Shape sampling policy. Default: grid",
    )
    parser.add_argument(
        "--value-distribution",
        choices=(
            "uniform",
            "normal",
            "log-uniform",
            "diagonal-dominant",
            "low-rank",
            "ill-conditioned",
            "sparse",
            "zero-columns",
        ),
        default="uniform",
        help="Matrix value distribution or stress scenario. Default: uniform",
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
        help="Minimum value for bounded distributions. Default: -1.0",
    )
    parser.add_argument(
        "--max-value",
        type=float,
        default=1.0,
        help="Maximum value for bounded distributions. Default: 1.0",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.0,
        help="Mean for normal distribution. Default: 0.0",
    )
    parser.add_argument(
        "--stddev",
        type=float,
        default=1.0,
        help="Standard deviation for normal and structured factors. Default: 1.0",
    )
    parser.add_argument(
        "--sparsity",
        type=parse_probability,
        default=0.9,
        help="Zero probability for sparse distribution. Default: 0.9",
    )
    parser.add_argument(
        "--rank",
        type=parse_positive_int,
        default=4,
        help="Target rank for low-rank and ill-conditioned scenarios. Default: 4",
    )
    parser.add_argument(
        "--condition",
        type=float,
        default=1.0e8,
        help="Approximate condition ratio for ill-conditioned scenario. Default: 1e8",
    )
    parser.add_argument(
        "--allow-wide",
        action="store_true",
        help="Allow rows < columns cases. The CUDA SVD currently rejects them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )


def options_from_args(args: argparse.Namespace) -> GenerateOptions:
    """Build typed generator options from argparse output."""

    if args.max_value < args.min_value:
        raise ValueError("--max-value must be >= --min-value.")
    if args.stddev <= 0.0:
        raise ValueError("--stddev must be positive.")
    if args.condition < 1.0:
        raise ValueError("--condition must be >= 1.")

    cases_per_file = tuple(args.cases_per_file)
    if len(cases_per_file) not in (1, args.files):
        raise ValueError("--cases-per-file must contain either one value or exactly --files values.")

    return GenerateOptions(
        output=args.output,
        files=args.files,
        cases_per_file=cases_per_file,
        rows=tuple(args.rows),
        columns=tuple(args.columns),
        shape_distribution=args.shape_distribution,
        value_distribution=args.value_distribution,
        seed=args.seed,
        min_value=args.min_value,
        max_value=args.max_value,
        mean=args.mean,
        stddev=args.stddev,
        sparsity=args.sparsity,
        rank=args.rank,
        condition=args.condition,
        allow_wide=args.allow_wide,
        overwrite=args.overwrite,
    )


def output_path_for_index(base: Path, index: int, file_count: int, suffix: str) -> Path:
    """Resolve one output path from a file, directory, or pattern."""

    if "{index" in str(base):
        return Path(str(base).format(index=index, index1=index + 1))

    if file_count == 1 and base.suffix:
        return base

    if base.suffix and file_count > 1:
        stem = base.stem
        return base.with_name(f"{stem}_{index:04d}{suffix}")

    return base / f"cases_{index:04d}{suffix}"


def prepare_output_path(path: Path, overwrite: bool) -> None:
    """Create parent directory and guard existing output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path} (use --overwrite)")


def build_shape_candidates(options: GenerateOptions) -> tuple[Shape, ...]:
    """Build candidate shapes according to row/column sets and policy."""

    candidates = [(rows, columns) for rows in options.rows for columns in options.columns]
    if not options.allow_wide:
        candidates = [(rows, columns) for rows, columns in candidates if rows >= columns]

    if options.shape_distribution == "square":
        candidates = [(rows, columns) for rows, columns in candidates if rows == columns]
    elif options.shape_distribution == "tall-skinny":
        candidates = [(rows, columns) for rows, columns in candidates if rows >= 2 * columns]
    elif options.shape_distribution == "wide":
        candidates = [(rows, columns) for rows, columns in candidates if rows < columns]

    if not candidates:
        raise ValueError("shape options produced no valid matrix shapes.")

    return tuple(candidates)


def choose_shape(candidates: Sequence[Shape], index: int, rng: random.Random, distribution: str) -> Shape:
    """Choose one shape from candidates."""

    if distribution == "grid":
        return candidates[index % len(candidates)]

    if distribution == "uniform":
        return rng.choice(candidates)

    if distribution == "log-uniform":
        weights = [1.0 / math.sqrt(rows * columns) for rows, columns in candidates]
        total = sum(weights)
        threshold = rng.random() * total
        prefix = 0.0
        for candidate, weight in zip(candidates, weights):
            prefix += weight
            if prefix >= threshold:
                return candidate

    return candidates[index % len(candidates)]


def bounded_uniform(rng: random.Random, options: GenerateOptions) -> float:
    """Sample one bounded uniform value."""

    return rng.uniform(options.min_value, options.max_value)


def signed_log_uniform(rng: random.Random, options: GenerateOptions) -> float:
    """Sample one signed log-uniform value inside the configured magnitude range."""

    min_abs = max(min(abs(options.min_value), abs(options.max_value)), 1.0e-300)
    max_abs = max(abs(options.min_value), abs(options.max_value), min_abs)
    magnitude = math.exp(rng.uniform(math.log(min_abs), math.log(max_abs)))
    return magnitude if rng.random() < 0.5 else -magnitude


def generate_dense_values(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate dense elementwise distributions."""

    values = []
    for row in range(rows):
        for column in range(columns):
            if options.value_distribution == "normal":
                value = rng.gauss(options.mean, options.stddev)
            elif options.value_distribution == "log-uniform":
                value = signed_log_uniform(rng, options)
            else:
                value = bounded_uniform(rng, options)

            smooth = 0.05 * math.sin(0.013 * (row + 1) * (column + 1))
            values.append(value + smooth)
    return values


def generate_diagonal_dominant(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate a diagonal-dominant rectangular matrix."""

    values = []
    diagonal_scale = max(abs(options.max_value), abs(options.min_value), 1.0)
    for row in range(rows):
        for column in range(columns):
            noise = 0.01 * bounded_uniform(rng, options)
            if row == column:
                values.append(diagonal_scale + abs(bounded_uniform(rng, options)) + noise)
            else:
                values.append(noise)
    return values


def generate_low_rank(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate a low-rank-ish matrix from two dense factors."""

    rank = min(options.rank, rows, columns)
    left = [[rng.gauss(0.0, options.stddev) for _ in range(rank)] for _ in range(rows)]
    right = [[rng.gauss(0.0, options.stddev) for _ in range(columns)] for _ in range(rank)]

    values = []
    for row in range(rows):
        for column in range(columns):
            value = sum(left[row][inner] * right[inner][column] for inner in range(rank))
            value += 1.0e-6 * rng.gauss(0.0, options.stddev)
            values.append(value)
    return values


def generate_ill_conditioned(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate a simple ill-conditioned factor product."""

    rank = min(options.rank, rows, columns)
    left = [[rng.gauss(0.0, 1.0) for _ in range(rank)] for _ in range(rows)]
    right = [[rng.gauss(0.0, 1.0) for _ in range(columns)] for _ in range(rank)]
    scales = [options.condition ** (-inner / max(rank - 1, 1)) for inner in range(rank)]

    values = []
    for row in range(rows):
        for column in range(columns):
            value = sum(scales[inner] * left[row][inner] * right[inner][column] for inner in range(rank))
            value += 1.0e-12 * rng.gauss(0.0, 1.0)
            values.append(value)
    return values


def generate_sparse(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate a sparse matrix with configured zero probability."""

    values = []
    for row in range(rows):
        for column in range(columns):
            if rng.random() < options.sparsity:
                values.append(0.0)
            else:
                values.append(bounded_uniform(rng, options))
    return values


def generate_zero_columns(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> list[float]:
    """Generate a matrix with deterministic zero columns."""

    zero_stride = max(2, columns // 4)
    values = []
    for row in range(rows):
        for column in range(columns):
            if column % zero_stride == 0:
                values.append(0.0)
            else:
                values.append(bounded_uniform(rng, options))
    return values


def generate_matrix(rows: int, columns: int, rng: random.Random, options: GenerateOptions) -> Matrix:
    """Generate one matrix for the selected value distribution."""

    if options.value_distribution in ("uniform", "normal", "log-uniform"):
        values = generate_dense_values(rows, columns, rng, options)
    elif options.value_distribution == "diagonal-dominant":
        values = generate_diagonal_dominant(rows, columns, rng, options)
    elif options.value_distribution == "low-rank":
        values = generate_low_rank(rows, columns, rng, options)
    elif options.value_distribution == "ill-conditioned":
        values = generate_ill_conditioned(rows, columns, rng, options)
    elif options.value_distribution == "sparse":
        values = generate_sparse(rows, columns, rng, options)
    elif options.value_distribution == "zero-columns":
        values = generate_zero_columns(rows, columns, rng, options)
    else:
        raise ValueError(f"unsupported value distribution: {options.value_distribution}")

    return Matrix(rows=rows, columns=columns, values=tuple(values))


def iter_matrices(options: GenerateOptions, file_index: int, case_count: int) -> Iterable[Matrix]:
    """Yield deterministic matrices for one output file."""

    candidates = build_shape_candidates(options)
    for case_index in range(case_count):
        global_index = file_index * 1_000_000 + case_index
        rng = random.Random(options.seed + global_index)
        rows, columns = choose_shape(candidates, case_index, rng, options.shape_distribution)
        yield generate_matrix(rows, columns, rng, options)


def cases_for_file(options: GenerateOptions, file_index: int) -> int:
    """Return configured case count for a file."""

    if len(options.cases_per_file) == 1:
        return options.cases_per_file[0]
    return options.cases_per_file[file_index]


def generate_files(options: GenerateOptions, suffix: str, writer: Writer) -> int:
    """Generate all configured files and return total testcase count."""

    total = 0
    for file_index in range(options.files):
        path = output_path_for_index(options.output, file_index, options.files, suffix)
        case_count = cases_for_file(options, file_index)
        prepare_output_path(path, options.overwrite)
        written = writer(path, iter_matrices(options, file_index, case_count))
        print(f"generated {written} testcase(s): {path}")
        total += written
    return total


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
