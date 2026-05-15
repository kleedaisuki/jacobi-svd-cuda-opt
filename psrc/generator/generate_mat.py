"""Generate binary .mat testcase streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from generator.common import add_common_arguments, generate_files, options_from_args, write_mat_stream


DEFAULT_OUTPUT = Path("experiments/cases/mat/generated.mat")


def build_parser() -> argparse.ArgumentParser:
    """Build the .mat generator argument parser."""

    parser = argparse.ArgumentParser(description="Generate binary .mat testcase streams.")
    add_common_arguments(parser, DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run .mat generation from parsed arguments."""

    options = options_from_args(args)
    count = generate_files(options, ".mat", write_mat_stream)
    print(f"generated {count} total .mat testcase(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry for direct .mat generation."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
