"""Top-level generator command dispatcher."""

from __future__ import annotations

import argparse

from generator import generate_mat, generate_txt


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level generator parser."""

    parser = argparse.ArgumentParser(prog="generate", description="Generate Jacobi SVD testcase streams.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mat_parser = subparsers.add_parser("mat", help="Generate binary .mat testcase stream.")
    generate_mat.add_common_arguments(mat_parser, generate_mat.DEFAULT_OUTPUT)
    mat_parser.set_defaults(handler=generate_mat.run)

    txt_parser = subparsers.add_parser("txt", help="Generate text testcase stream.")
    generate_txt.add_common_arguments(txt_parser, generate_txt.DEFAULT_OUTPUT)
    txt_parser.set_defaults(handler=generate_txt.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch generate <mat|txt> commands."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
