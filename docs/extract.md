# Perf Extractor

`extract` turns the rough profiler output from `bench.sh` into script-readable
files under `experiments/perf/`.

The tool does not rerun benchmarks. It reads one directory from
`experiments/prof/<instance>-<timestamp>/`, keeps those raw artifacts intact, and
writes a normalized sibling directory under `experiments/perf/`.

## Quick Start

```bash
extract latest
extract sample-20260515_141407 --force
extract experiments/prof/sample-20260515_141407 --skip-ncu
```

`latest` selects the newest directory matching:

```text
<instance>-YYYYMMDD_HHMMSS
```

## Output Layout

For an input such as:

```text
experiments/prof/sample-20260515_141407/
```

the extractor writes:

```text
experiments/perf/sample-20260515_141407/
├── manifest.json
├── runs.jsonl
├── summary.csv
├── summary.json
├── report.md
├── nsys/
│   └── <case>_run_<N>_<report>.json
└── ncu/
    ├── <case>_<mode>_run_<N>_raw.csv
    └── <case>_<mode>_run_<N>_metrics.json
```

## Nsight Systems

For `nsys` runs, `extract` prefers the generated SQLite file:

```text
<case>/nsys/run_N_nsys.sqlite
```

If the SQLite file is missing, it falls back to:

```text
<case>/nsys/run_N_nsys.nsys-rep
```

It invokes `nsys stats` and stores JSON output for these reports by default:

```text
cuda_api_sum
cuda_gpu_kern_sum
cuda_gpu_mem_time_sum
cuda_gpu_mem_size_sum
osrt_sum
```

You can select reports explicitly:

```bash
extract latest --nsys-report cuda_api_sum --nsys-report osrt_sum
```

## Nsight Compute

For `ncu-basic` and `ncu-deep`, `extract` reads `.ncu-rep` files and exports CSV
through the official importer:

```bash
ncu --import <report.ncu-rep> --csv --page raw --print-units base --print-fp
```

The raw CSV is preserved, and a smaller `metrics.json` extracts stable columns
when present, including kernel name, launch shape, GPU duration, selected
throughput metrics, register count, shared memory, and replay passes.

Use the details page instead of raw if needed:

```bash
extract latest --ncu-page details
```

## Log Information

The extractor also parses each `run_N.log` for useful control-plane data:

```text
$ ...                         command line
==PROF== Profiling "..."      profiled kernel names and pass counts
==ERROR== ...                 profiler errors
SKIPPED: ...                  missing profiler data warnings
==PROF== Report: ...          generated report path
Generated: ...                generated nsys artifacts
```

Errors in a profiler log mark the record as `partial` if a report artifact still
exists, otherwise `failed`.

## Common Options

```text
--force        overwrite an existing experiments/perf/<run> directory
--skip-nsys    skip nsys JSON extraction
--skip-ncu     skip ncu CSV extraction
--prof-dir     profiling input root, default experiments/prof
--perf-dir     extraction output root, default experiments/perf
```

`summary.csv` is the easiest file for scripts. `runs.jsonl` keeps richer
per-run records, including warnings, errors, artifacts, profiler summaries, and
selected NCU metrics.
