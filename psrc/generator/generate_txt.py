"""Generate text testcase streams."""

from __future__ import annotations

import argparse
from pathlib import Path

from generator.common import add_common_arguments, generate_files, options_from_args, write_txt_stream


DEFAULT_OUTPUT = Path("experiments/cases/txt/generated.txt")


def build_parser() -> argparse.ArgumentParser:
    """Build the text generator argument parser."""

    parser = argparse.ArgumentParser(description="Generate text testcase streams.")
    add_common_arguments(parser, DEFAULT_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run text generation from parsed arguments."""

    options = options_from_args(args)
    count = generate_files(options, ".txt", write_txt_stream)
    print(f"generated {count} total text testcase(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry for direct text generation."""

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
