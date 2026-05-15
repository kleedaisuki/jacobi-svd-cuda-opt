# Baseline 性能剖析报告

## 1. 观察起点

本报告分析 2026-05-15 生成并提取的 baseline 性能数据，目标是回答两个问题：

1. 当前 baseline 有哪些值得注意的性能现象？
2. 主要性能瓶颈更像是 GPU 计算、内存带宽、I/O，还是 CUDA 调度与同步开销？

报告只基于已经落盘的机器可读数据，不重新运行实验，也不修改实验结果。

## 2. 数据来源

### 2.1 实验产物

| 类型 | 路径 | 内容 |
|---|---|---|
| timing 提取结果 | `experiments/perf/baseline_timing-20260515_155426/` | `timing` 模式，14 个 case，每个 case 3 次运行 |
| profile 提取结果 | `experiments/perf/baseline_profile-20260515_155701/` | `nsys`、`ncu-basic`、`ncu-deep`，10 个代表 case，每个模式 1 次 |
| timing 实例 | `scripts/instances/baseline_timing.sh` | baseline timing 实验矩阵 |
| profile 实例 | `scripts/instances/baseline_profile.sh` | baseline Nsight 实验矩阵 |
| 输入数据 | `experiments/cases/baseline/` | 由项目内 `generate` 命令生成的 `.mat` 与 `.txt` 用例 |

### 2.2 数据完整性

| 项目 | 数量 |
|---|---:|
| timing `runs.jsonl` 记录 | 42 |
| profile `runs.jsonl` 记录 | 30 |
| timing `summary.csv` 行数，含表头 | 43 |
| profile `summary.csv` 行数，含表头 | 31 |
| profile `nsys` JSON 文件 | 20 |
| profile `ncu` CSV/metrics 文件 | 40 |
| `failed` / `partial` 记录 | 0 |

`nsys` 记录中存在 5 条 warning/record，主要是没有 NVTX 数据、没有 GPU memory summary JSON、没有 CUDA GPU kernel summary JSON。这不影响本报告使用的 CUDA API summary 与 OS runtime summary；所有 profile 记录状态仍为 `ok`。

### 2.3 术语说明

- 控制面开销（control-plane overhead）：CPU 侧提交、同步、调度 GPU work 的开销，不等同于 GPU kernel 自身计算时间。
- 内核发射（kernel launch）：CPU 侧向 CUDA runtime/driver 提交一个 GPU kernel 的动作。
- 主机到设备拷贝（host-to-device copy, H2D）与设备到主机拷贝（device-to-host copy, D2H）：CPU memory 与 GPU memory 间的数据传输。
- 流式多处理器（Streaming Multiprocessor, SM）：NVIDIA GPU 上执行线程块的基本计算单元。
- 吞吐率（throughput）：Nsight Compute 中相对于硬件峰值的利用率百分比。

## 3. Timing 层观察

### 3.1 全量 timing 汇总

| case | runs | elapsed mean s | elapsed median s | app mean ms | app ms/testcase | app ms/sweep | sweeps/testcase | testcases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mat_grid_small_auto` | 3 | 2.710 | 1.790 | 2280.7 | 23.76 | 3.33 | 7.12 | 96 |
| `mat_ill_conditioned_auto` | 3 | 1.363 | 1.370 | 1342.3 | 33.56 | 2.97 | 11.30 | 40 |
| `mat_low_rank_auto` | 3 | 1.197 | 1.210 | 1173.5 | 29.34 | 2.91 | 10.07 | 40 |
| `mat_sparse_auto` | 3 | 0.780 | 0.780 | 758.9 | 18.97 | 2.91 | 6.53 | 40 |
| `mat_square_medium_auto` | 3 | 7.133 | 7.000 | 7108.4 | 148.09 | 13.88 | 10.67 | 48 |
| `mat_square_medium_off` | 3 | 7.600 | 7.280 | 7230.8 | 150.64 | 14.12 | 10.67 | 48 |
| `mat_square_medium_on` | 3 | 6.933 | 6.970 | 6913.9 | 144.04 | 13.50 | 10.67 | 48 |
| `mat_tall_skinny_medium_auto` | 3 | 1.453 | 1.450 | 1430.9 | 22.36 | 3.11 | 7.19 | 64 |
| `mat_tall_skinny_medium_off` | 3 | 1.470 | 1.470 | 1446.6 | 22.60 | 3.14 | 7.19 | 64 |
| `mat_tall_skinny_medium_on` | 3 | 1.853 | 1.440 | 1396.4 | 21.82 | 3.04 | 7.19 | 64 |
| `mat_zero_columns_auto` | 3 | 0.840 | 0.850 | 819.4 | 20.48 | 3.19 | 6.42 | 40 |
| `txt_grid_small_auto` | 3 | 0.987 | 1.010 | 962.7 | 15.04 | 2.21 | 6.81 | 64 |
| `txt_ill_conditioned_small_auto` | 3 | 0.783 | 0.780 | 761.6 | 23.80 | 2.27 | 10.47 | 32 |
| `txt_sparse_small_auto` | 3 | 0.887 | 0.830 | 862.0 | 26.94 | 3.61 | 7.47 | 32 |

### 3.2 从 timing 看出的第一层事实

`mat_square_medium_*` 是最重的一组。它的 `app ms/testcase` 约为 `144-151 ms`，明显高于 `mat_tall_skinny_medium_*` 的 `21.8-22.6 ms/testcase`。二者 testcase 数量不同，但按 testcase 归一化以后，差距仍然约为 `6.4x-6.9x`。

这说明瓶颈与矩阵形状强相关，尤其与列数和 round-robin schedule 的 round 数强相关。Jacobi SVD 的核心循环按列对轮次推进，列数越大，round 数越多；即使总元素数量不是唯一最大，较大的列数也会制造更多细粒度 CUDA work。

`mat_ill_conditioned_auto` 和 `mat_low_rank_auto` 的 `sweeps/testcase` 分别约为 `11.30` 与 `10.07`，比 sparse/zero-columns 更高。这符合数值压力用例需要更多 sweep 才收敛的直觉。但它们的总耗时仍低于 `mat_square_medium_*`，说明当前最强的性能放大器不是“数值难度”本身，而是“列数导致的 round 粒度爆炸”。

`mat_grid_small_auto` 第一轮 wall time 明显偏高，`mat_tall_skinny_medium_on` 也有一次 wall-time outlier。后续判断优先使用 app report 与多次运行均值/中位数，不把单次 wall-time outlier 当成核心结论。

## 4. Nsight Systems 层观察

### 4.1 CUDA API 汇总

表格中的格式为：

`总 API 时间秒 / 调用次数 / 平均每次调用微秒`

注意：Nsight Systems 的 CUDA API 总时间是 API 调用时长求和，不是程序 wall-clock。pipeline 使用多线程提交任务，所以不同 CPU 线程上的 CUDA API 时间可能重叠。它不能直接当作运行时间，但可以可靠揭示“开销在哪里堆积”。

| case | `cudaLaunchKernel` | `cudaMemcpy` | `cudaMemset` | `cudaMalloc+Free` | round-like `cudaMemset` |
|---|---:|---:|---:|---:|---:|
| `mat_grid_small_auto` | 2.755 / 56,502 / 48.76 | 2.300 / 37,920 / 60.65 | 1.102 / 18,768 / 58.69 | 0.168 / 2,118 | 18,768 |
| `mat_ill_conditioned_auto` | 1.613 / 43,444 / 37.14 | 1.538 / 29,032 / 52.97 | 0.594 / 14,436 / 41.17 | 0.105 / 936 | 14,436 |
| `mat_sparse_auto` | 0.883 / 24,207 / 36.48 | 0.890 / 16,214 / 54.91 | 0.352 / 8,027 / 43.80 | 0.113 / 926 | 8,027 |
| `mat_square_medium_auto` | 13.332 / 232,692 / 57.29 | 10.630 / 155,200 / 68.49 | 5.548 / 77,504 / 71.58 | 0.141 / 1,140 | 77,504 |
| `mat_square_medium_off` | 13.317 / 232,608 / 57.25 | 11.202 / 155,200 / 72.18 | 5.818 / 77,504 / 75.07 | 0.159 / 1,056 | 77,504 |
| `mat_square_medium_on` | 12.488 / 232,704 / 53.66 | 10.431 / 155,200 / 67.21 | 5.059 / 77,504 / 65.28 | 0.155 / 1,152 | 77,504 |
| `mat_tall_skinny_medium_auto` | 1.369 / 44,268 / 30.93 | 1.468 / 29,624 / 49.56 | 0.505 / 14,684 / 34.40 | 0.132 / 1,496 | 14,684 |
| `mat_tall_skinny_medium_off` | 1.423 / 44,180 / 32.20 | 1.557 / 29,624 / 52.55 | 0.515 / 14,684 / 35.07 | 0.110 / 1,408 | 14,684 |
| `mat_tall_skinny_medium_on` | 1.449 / 44,308 / 32.69 | 1.548 / 29,624 / 52.26 | 0.497 / 14,684 / 33.85 | 0.155 / 1,536 | 14,684 |
| `txt_grid_small_auto` | 1.569 / 29,216 / 53.70 | 1.304 / 19,648 / 66.35 | 0.606 / 9,696 / 62.52 | 0.525 / 1,408 | 9,696 |

### 4.2 从 Nsight Systems 看出的第二层事实

`mat_square_medium_*` 的 CUDA API 调用次数极大：

- `cudaLaunchKernel` 约 `232k` 次；
- `cudaMemcpy` 约 `155k` 次；
- `cudaMemset` 约 `77.5k` 次。

这些数字与代码中的 round 级模式高度吻合。核心循环位于 `src/domain/jacobi_svd.cu`：

- 每个 round 拷贝当前列对 `round` 到 device：`cudaMemcpy(d_pairs.data(), round.data(), ...)`；
- 每个 round 清 `d_any_rotation`：`cudaMemset(d_any_rotation.data(), 0, sizeof(int))`；
- 每个 round 发射 `pair_stats_kernel`；
- 每个 round 发射 `compute_rotation_params_kernel`；
- 每个 round 发射 `apply_rotation_kernel`；
- 每个 round 从 device 拷回 `any_rotation`，用于判断本 sweep 是否收敛。

也就是说，当前结构近似为：

```text
每个 Jacobi round:
  H2D memcpy pairs
  memset convergence flag
  launch pair_stats_kernel
  launch compute_rotation_params_kernel
  launch apply_rotation_kernel
  D2H memcpy convergence flag
```

这正好解释了表中的比例关系：

- `cudaMemset` 次数近似等于 round-like 单元数；
- `cudaLaunchKernel` 次数大约是 `3 * round-like 单元数`，再加少量初始化、转置、构建 U/Sigma 等 kernel；
- `cudaMemcpy` 次数大约是 `2 * round-like 单元数`，再加输入输出矩阵传输。

这是一种非常典型的“单次 GPU work 太小，调度与同步开销吞掉收益”的形态。这里的瓶颈不优雅，但很真实，呜。

## 5. Nsight Compute 层观察

### 5.1 Kernel 采样汇总

Nsight Compute 只按 instance 中配置的 filter 采样了少量 launch：`ncu-basic` 采样 `pair_stats_kernel`，`ncu-deep` 采样 `apply_rotation_kernel`。因此这些数据代表“被采样 kernel 的局部特征”，不是所有 kernel launch 的完整分布。

| case | mode | kernel variant | samples | avg duration us | avg SM % | avg memory % | avg DRAM % | regs/thread | shared B/block |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mat_grid_small_auto` | `ncu-basic` | `pair_stats_kernel<0>` | 2 | 6.480 | 0.62 | 0.67 | 0.20 | 40.0 | 7168 |
| `mat_grid_small_auto` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 4.352 | 0.24 | 1.39 | 0.40 | 32.0 | 1024 |
| `mat_ill_conditioned_auto` | `ncu-basic` | `pair_stats_kernel<1>` | 1 | 6.528 | 7.99 | 3.29 | 2.45 | 40.0 | 7168 |
| `mat_ill_conditioned_auto` | `ncu-basic` | `pair_stats_kernel<0>` | 1 | 6.816 | 1.95 | 1.65 | 0.71 | 40.0 | 7168 |
| `mat_ill_conditioned_auto` | `ncu-deep` | `apply_rotation_kernel<1>` | 1 | 5.056 | 5.19 | 9.50 | 4.69 | 30.0 | 1024 |
| `mat_sparse_auto` | `ncu-basic` | `pair_stats_kernel<1>` | 2 | 6.880 | 8.71 | 3.83 | 3.39 | 40.0 | 7168 |
| `mat_sparse_auto` | `ncu-deep` | `apply_rotation_kernel<1>` | 1 | 5.216 | 8.35 | 11.31 | 7.39 | 30.0 | 1024 |
| `mat_square_medium_auto` | `ncu-basic` | `pair_stats_kernel<1>` | 1 | 6.848 | 10.77 | 4.64 | 2.62 | 40.0 | 7168 |
| `mat_square_medium_auto` | `ncu-basic` | `pair_stats_kernel<0>` | 1 | 6.752 | 3.25 | 1.70 | 0.44 | 40.0 | 7168 |
| `mat_square_medium_auto` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 4.704 | 0.92 | 4.09 | 1.32 | 32.0 | 1024 |
| `mat_square_medium_off` | `ncu-basic` | `pair_stats_kernel<0>` | 2 | 6.592 | 3.25 | 1.70 | 0.46 | 40.0 | 7168 |
| `mat_square_medium_off` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 4.608 | 0.95 | 4.30 | 1.12 | 32.0 | 1024 |
| `mat_square_medium_on` | `ncu-basic` | `pair_stats_kernel<1>` | 2 | 6.496 | 7.02 | 3.17 | 1.32 | 40.0 | 7168 |
| `mat_square_medium_on` | `ncu-deep` | `apply_rotation_kernel<1>` | 1 | 4.672 | 0.96 | 2.94 | 1.10 | 30.0 | 1024 |
| `mat_tall_skinny_medium_auto` | `ncu-basic` | `pair_stats_kernel<0>` | 2 | 6.640 | 0.98 | 1.07 | 0.45 | 40.0 | 7168 |
| `mat_tall_skinny_medium_auto` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 4.928 | 1.11 | 4.09 | 1.12 | 32.0 | 1024 |
| `mat_tall_skinny_medium_off` | `ncu-basic` | `pair_stats_kernel<0>` | 2 | 6.864 | 5.67 | 3.93 | 1.78 | 40.0 | 7168 |
| `mat_tall_skinny_medium_off` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 5.024 | 2.12 | 8.96 | 2.13 | 32.0 | 1024 |
| `mat_tall_skinny_medium_on` | `ncu-basic` | `pair_stats_kernel<1>` | 2 | 6.528 | 1.51 | 0.86 | 0.60 | 40.0 | 7168 |
| `mat_tall_skinny_medium_on` | `ncu-deep` | `apply_rotation_kernel<1>` | 1 | 4.416 | 0.62 | 1.54 | 0.75 | 30.0 | 1024 |
| `txt_grid_small_auto` | `ncu-basic` | `pair_stats_kernel<0>` | 2 | 6.432 | 0.62 | 0.69 | 0.20 | 40.0 | 7168 |
| `txt_grid_small_auto` | `ncu-deep` | `apply_rotation_kernel<0>` | 1 | 4.448 | 0.25 | 1.38 | 0.39 | 32.0 | 1024 |

### 5.2 从 Nsight Compute 看出的第三层事实

被采样 kernel 的时长普遍只有几微秒：

- `pair_stats_kernel` 约 `6.4-7.0 us`；
- `apply_rotation_kernel` 约 `4.3-5.2 us`。

SM、memory、DRAM 吞吐率普遍很低，很多记录低于 `5%`，最高的局部样本也只是十几个百分点。这不像“GPU 算力打满”或“DRAM 带宽打满”。更像 GPU 每次只拿到一小块 work，刚启动就结束，硬件吞吐率还没爬起来，CPU 侧又要提交下一批 work。

这与 Nsight Systems 的结论一致：当前最大瓶颈不是 kernel 内部指令效率，而是工作粒度太细。

## 6. 代码路径映射

### 6.1 主循环

核心路径在 `src/domain/jacobi_svd.cu` 的 `run_one_sided_jacobi_svd_internal`：

- 分配 `d_a`、`d_v`、`d_u`、`d_sigma`：`src/domain/jacobi_svd.cu:133-137`
- 输入矩阵 H2D：`src/domain/jacobi_svd.cu:139`
- 构造 round-robin schedule：`src/domain/jacobi_svd.cu:151`
- sweep/round 双层循环：`src/domain/jacobi_svd.cu:168-253`
- round 内 H2D pairs：`src/domain/jacobi_svd.cu:180-183`
- round 内 reset convergence flag：`src/domain/jacobi_svd.cu:184`
- round 内三个 kernel：`src/domain/jacobi_svd.cu:189-243`
- round 内 D2H convergence flag：`src/domain/jacobi_svd.cu:246-248`
- 构建 U/Sigma 与输出 D2H：`src/domain/jacobi_svd.cu:269-280`

### 6.2 分配与释放

`DeviceMatrix::reset` 直接调用 `cudaMalloc`，析构或 reset 调用 `cudaFree`。相关路径在 `src/domain/device_matrix.cu:55-63`。profile 中 `cudaMalloc+cudaFree` 不是最大头，但每个 testcase 都重新构造多块 device matrix 与 buffer，会造成稳定的次级开销。

### 6.3 pipeline 并发

`src/application/pipeline.cu` 使用全局线程池提交 testcase：`src/application/pipeline.cu:70-79` 与 `src/application/pipeline.cu:95-103`。这能隐藏部分 wall-clock 延迟，但也会让多个 CPU worker 同时向同一 GPU 发送大量细碎 CUDA work。对于当前这种 micro-kernel-heavy 模式，并发不一定总是收益，可能把 CUDA runtime/driver 调度压力推高。

## 7. MECE 瓶颈归因

### 7.1 计算瓶颈：目前证据弱

如果主要是计算瓶颈，预期会看到较高 SM throughput，或者至少某些核心 kernel 有长时间运行与高占用特征。但 NCU 采样显示核心 kernel 通常只有 `4-7 us`，SM throughput 大多很低。因此“单个 kernel 算得太慢”不是当前最强解释。

### 7.2 内存带宽瓶颈：目前证据弱到中等

如果主要是 DRAM bandwidth 瓶颈，预期会看到高 DRAM throughput。但 NCU 中 `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` 普遍较低。部分 sparse/ill-conditioned 样本 memory throughput 稍高，但远未接近饱和。因此“设备端全局内存带宽打满”不是第一瓶颈。

### 7.3 CUDA 调度与同步瓶颈：证据强

这是目前最强解释：

- `mat_square_medium_*` 有约 `232k` 次 kernel launch；
- 同组有约 `155k` 次 `cudaMemcpy`；
- 同组有约 `77.5k` 次 `cudaMemset`；
- 每个 round 的代码结构确实包含 `3 launch + 2 memcpy + 1 memset`；
- 单个 sampled kernel 只有几微秒；
- SM/DRAM 利用率低。

这些证据互相咬合，形成清晰链条：round 级控制面开销主导。

### 7.4 内存分配瓶颈：证据中等，优先级次于调度

`cudaMalloc+cudaFree` 在各 profile case 中有数百到上千次调用，总 API 时间通常低于 launch/memcpy/memset，但不是零。workspace 复用会有收益，但如果先不处理 round 级 launch/memcpy，同样会被更大的开销掩盖。

### 7.5 I/O 与格式解析瓶颈：当前不是主瓶颈

`txt_grid_small_auto` 的 wall time 与 app time 不高，`.txt` 路径没有显示出压倒性的解析成本。文本输入会影响端到端吞吐，但本轮 baseline 的最大性能异常仍然来自 CUDA 控制面。

## 8. Layout Transpose 观察

layout transpose 对 square medium 有小幅收益：

| case | app mean ms | app ms/testcase |
|---|---:|---:|
| `mat_square_medium_auto` | 7108.4 | 148.09 |
| `mat_square_medium_off` | 7230.8 | 150.64 |
| `mat_square_medium_on` | 6913.9 | 144.04 |

`on` 比 `off` 约快 `4.4%`，比 `auto` 约快 `2.7%`。这说明 layout transpose 方向是有效的，但收益量级远小于 round 级控制面开销带来的损耗。

对于 tall-skinny medium：

| case | app mean ms | app ms/testcase |
|---|---:|---:|
| `mat_tall_skinny_medium_auto` | 1430.9 | 22.36 |
| `mat_tall_skinny_medium_off` | 1446.6 | 22.60 |
| `mat_tall_skinny_medium_on` | 1396.4 | 21.82 |

三者差距很小，`on` 仍略好，但没有形成压倒性差异。后续调优 layout threshold 可以做，但不应优先于控制面开销治理。

## 9. 优化优先级

### P0：消除 per-round host/device 往返

当前最该砍的是每 round 的 `d_pairs` H2D 与 `any_rotation` D2H。

可选方向：

1. 将 round-robin schedule 预先展开并常驻 device memory，避免每 round `cudaMemcpy(d_pairs, round.data(), ...)`。
2. 将 `any_rotation` 从 per-round D2H 改为 per-sweep 汇总；一个 sweep 结束后再拷回 host 判断是否收敛。
3. 更激进地，把收敛判断尽量留在 device 侧，host 只在必要边界同步。

预计收益：对 square medium 这种 `77.5k` round-like 单元的 case，收益应非常明显。

### P1：减少 kernel launch 数量

当前每 round 三个 kernel：

1. `pair_stats_kernel`
2. `compute_rotation_params_kernel`
3. `apply_rotation_kernel`

可选方向：

1. 尝试融合 `compute_rotation_params_kernel` 与 `apply_rotation_kernel`。
2. 对固定 schedule 捕获 CUDA Graphs（CUDA Graphs），降低大量重复 launch 的 CPU 侧提交成本。
3. 对较小列数或小 pair_count 的 case，考虑一个 kernel 处理多个 round 或多个 pair group。

这里要谨慎：完全融合 `pair_stats` 与 `apply_rotation` 可能受到 reduction 与同步边界限制，不一定简单。但先减少一个 launch/round 就很有价值。

### P2：复用 device workspace

当前每 testcase 都构造 `DeviceMatrix` 与 `DeviceBuffer`，内部走 `cudaMalloc/cudaFree`。建议引入 worker-local 或 pipeline-local workspace：

1. 按最大 rows/columns 预分配 buffer；
2. testcase 间复用；
3. 必要时按容量增长，而不是每次释放重分配。

这能降低 allocator 开销，也能让后续 profile 更干净。

### P3：重新审视 pipeline 并发策略

当前 CPU 线程池会并发提交多个 testcase 到同一 GPU。对 coarse-grained kernel 这通常有利，但当前 workload 是大量 micro-kernel 与同步。建议后续单独实验：

1. 限制 kernel worker 并发为 1；
2. 保留 I/O/output 并发；
3. 比较 CUDA API 总时长、wall time、GPU 利用率。

目标不是简单减少并发，而是避免多个 CPU 线程同时制造 driver 队列压力。

### P4：继续 layout transpose threshold 调优

layout transpose 在 square medium 上有几个百分点收益，但不是首要瓶颈。建议在 P0/P1 后重新测 threshold，因为一旦控制面开销下降，layout 对 kernel 内存访问模式的影响会更清晰。

## 10. 局限性

1. `ncu` 只采样了少量 launch，不能代表所有 kernel 的完整分布。
2. `nsys` 的 CUDA API summed time 不能等同于 wall-clock，因为 pipeline 中存在 CPU 多线程并发。
3. 本轮 baseline 使用 `--max-sweeps 32`，不代表更高 sweep 上限下的全部行为。
4. 输入矩阵由 deterministic generator 生成，覆盖了多种形状和数值压力，但仍不是所有真实 workload。
5. 没有采集更细的 GPU timeline kernel summary JSON；当前 `nsys` 对 `cuda_gpu_kern_sum` 未产生 JSON，但 CUDA API 与 NCU 层已经足以支持主瓶颈判断。

## 11. 最终归纳

当前 baseline 的主要性能瓶颈不是单个 CUDA kernel 的计算效率，也不是设备端 DRAM 带宽饱和，而是 Jacobi round 粒度过细导致的控制面开销：每个 round 都有小 H2D 拷贝、小 memset、三次 kernel launch、一次小 D2H 拷贝。`mat_square_medium_*` 上约 `77.5k` 个 round-like 单元把这个问题放大到秒级。

下一步最值得做的是先重写 round 级数据流：让 schedule 常驻 device，减少 per-round host/device 同步，再考虑 kernel fusion 或 CUDA Graphs。workspace 复用与 layout threshold 调优也有价值，但应排在控制面治理之后。这个方向最符合当前数据，也最可能带来肉眼可见的 baseline 改善。喵。
