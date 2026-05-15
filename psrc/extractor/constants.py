"""Extractor constants."""

from __future__ import annotations

import re


RUN_DIR_RE = re.compile(r"^(?P<instance>.+)-(?P<timestamp>\d{8}_\d{6})$")
RUN_LOG_RE = re.compile(r"^run_(?P<index>\d+)\.log$")
NVIDIA_PROGRESS_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\r")
PROFILING_RE = re.compile(r'==PROF== Profiling "([^"]+)"\s+-\s+(\d+):.*?-\s+(\d+)\s+passes')
REPORT_RE = re.compile(r"==PROF== Report:\s*(?P<path>.+)")
TIME_RE = re.compile(
    r"elapsed_seconds=(?P<elapsed>[0-9.]+)\s+"
    r"user_seconds=(?P<user>[0-9.]+)\s+"
    r"sys_seconds=(?P<sys>[0-9.]+)\s+"
    r"max_rss_kb=(?P<rss>\d+)"
)

NSYS_REPORTS = (
    "cuda_api_sum",
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "osrt_sum",
)

NCU_METRICS = (
    "ID",
    "Process ID",
    "Process Name",
    "Kernel Name",
    "Context",
    "Stream",
    "Block Size",
    "Grid Size",
    "Device",
    "CC",
    "gpu__time_duration.sum",
    "gpu__time_duration.avg",
    "gpu__time_duration.max",
    "gpu__time_duration.min",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "profiler__replayer_passes",
)
