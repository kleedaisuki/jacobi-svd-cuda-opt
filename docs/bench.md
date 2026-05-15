# Bench Pipeline

`bench.sh` is the root benchmark/profiling pipeline for this project. It builds the CUDA executable, loads an experiment instance, runs all configured cases, and writes timing/profiler outputs under `experiments/prof/`.

## Quick Start

```bash
./bench.sh sample --dry-run
./bench.sh sample --modes timing --runs 3
./bench.sh sample --modes timing,nsys --case-jobs 2
```

The first positional argument is an instance name or a direct path to an instance script. For `sample`, the script resolves:

```text
experiments/instances/sample.sh
```

## Instance Format

An instance script is a Bash script that declares global settings and registers cases with `add_case`.

```bash
RUNS=3
CASE_JOBS=2
MODES=(timing nsys ncu-basic ncu-deep)

APP_ARGS=(
    --format mat
    --max-sweeps 128
    --threads-per-block 256
)

add_case small experiments/cases/mat/small.mat --layout-transpose-mode auto
add_case large experiments/cases/mat/large.mat --layout-transpose-mode on
```

`APP_ARGS` are passed to every program invocation. Arguments after each `add_case` are appended only for that case.

## Command Options

```text
--runs N
```

Override `RUNS` from the instance script.

```text
--modes timing,nsys,ncu-basic,ncu-deep
```

Override `MODES`. Use a comma-separated list.

```text
--case-jobs N
```

Override `CASE_JOBS`. Cases run concurrently up to this limit. Each case still runs its own modes and repetitions sequentially.

```text
--skip-build
```

Skip CMake configure/build and use the existing executable.

```text
--dry-run
```

Print commands without executing them. This is useful for checking paths and instance expansion.

## Output Layout

Each run writes one timestamped directory:

```text
experiments/prof/<instance>-<timestamp>/
├── manifest.env
├── build.log
└── <case-name>/
    ├── timing/
    │   ├── run_1.log
    │   └── run_1_output.mat
    ├── nsys/
    │   ├── run_1.log
    │   ├── run_1_output.mat
    │   ├── run_1_nsys.nsys-rep
    │   └── run_1_nsys.sqlite
    ├── ncu-basic/
    │   ├── run_1.log
    │   ├── run_1_output.mat
    │   └── run_1_ncu_basic.ncu-rep
    └── ncu-deep/
        ├── run_1.log
        ├── run_1_output.mat
        └── run_1_ncu_deep.ncu-rep
```

For `.txt` input streams, output files use `.txt`; for `.mat` input streams, they use `.mat`.

## Modes

`timing` runs the executable through `/usr/bin/time` and asks the application to print its JSON report.

`nsys` runs:

```bash
nsys profile --trace=cuda,nvtx,osrt --stats=true
```

`ncu-basic` runs:

```bash
ncu --set basic
```

`ncu-deep` runs:

```bash
ncu --set full
```

Nsight Compute (`ncu`) needs permission to access NVIDIA GPU performance counters. If it fails with `ERR_NVGPUCTRPERM`, fix the system driver permission first or run only `timing,nsys`.

## Parallelism

`CASE_JOBS` controls case-level parallelism:

```bash
CASE_JOBS=2
```

The pipeline starts up to two cases at once. Within each case, modes and repetitions remain sequential. This avoids mixing profiler outputs for the same case while still letting independent cases run in parallel.

Be conservative with `ncu-basic` and `ncu-deep`: Nsight Compute can be very slow for workloads that launch thousands of kernels, and concurrent `ncu` sessions can compete heavily for the same GPU.

## Implementation Files

The root script is intentionally small:

```text
bench.sh
scripts/bench/core.sh
scripts/bench/instances.sh
scripts/bench/runners.sh
```

`bench.sh` owns the main flow. The helper scripts hold common utilities, instance handling, CMake build logic, and per-mode runners.
