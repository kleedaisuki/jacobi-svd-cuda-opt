"""Top-level generator command dispatcher."""

from __future__ import annotations

import argparse

from generator import generate_mat, generate_txt


FULL_HELP = """examples:
  generate mat --files 4 --cases-per-file 16,32,64,128 --output experiments/cases/mat --overwrite
  generate txt --shape-distribution tall-skinny --value-distribution ill-conditioned --rows 128:1024:128 --cols 8,16,32
  generate mat --output 'experiments/cases/mat/sweep_{index}.mat' --files 3 --cases-per-file 10

common options after generate <mat|txt>:
  -o, --output PATH
      Output file, directory, or filename pattern containing {index}.
  --files N
      Number of output files.
  --cases-per-file SPEC
      Cases per file as N, CSV, or START:STOP[:STEP]. One value applies to all files;
      multiple values must match --files.
  --rows SPEC
      Candidate row counts as N, CSV, or START:STOP[:STEP].
  --columns, --cols SPEC
      Candidate column counts as N, CSV, or START:STOP[:STEP].
  --shape-distribution {grid,uniform,log-uniform,square,tall-skinny,wide}
      Shape sampling policy.
  --value-distribution {uniform,normal,log-uniform,diagonal-dominant,low-rank,ill-conditioned,sparse,zero-columns}
      Element distribution or numerical stress scenario.
  --min-value FLOAT --max-value FLOAT
      Bounds for uniform/log-uniform/sparse values.
  --mean FLOAT --stddev FLOAT
      Parameters for normal and structured factors.
  --sparsity P
      Zero probability for sparse matrices.
  --rank N
      Target rank for low-rank and ill-conditioned matrices.
  --condition FLOAT
      Approximate condition ratio for ill-conditioned matrices.
  --allow-wide
      Allow rows < columns cases.
  --overwrite
      Overwrite existing outputs.

Run "generate mat --help" or "generate txt --help" for argparse's exact defaults.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level generator parser."""

    parser = argparse.ArgumentParser(
        prog="generate",
        description="Generate Jacobi SVD testcase streams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=FULL_HELP,
    )
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
