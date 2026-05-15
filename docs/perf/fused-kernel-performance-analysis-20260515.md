# 融合 cooperative kernel 性能剖析报告

## 1. 观察起点

本文分析 2026-05-15 生成并提取的融合算子性能数据，目标是回答两个问题：

1. 融合 cooperative kernel 相对融合前 baseline 带来了多大性能收益？
2. 融合后主要瓶颈从哪里显现出来？

报告只基于已经落盘的机器可读数据，不重新运行实验，也不修改实验结果。这里的“融合”指当前实现优先使用 `jacobi_sweep_kernel`，把原来每个 Jacobi round 的多次 host/device 往返与多个 kernel launch 合并为每个 sweep 一次 cooperative kernel launch。若设备不支持 cooperative launch，代码仍会回退到旧 round 级路径。

需要先明确一个分析边界：本报告评估的是当前数据集与当前 RTX 3070 Ti Laptop GPU 上的观测结果。不同 GPU、驱动版本、矩阵规模、输出格式和 `queue-capacity` 配置可能改变绝对时间，但不太可能改变“融合打掉控制面开销”这一主结论。

## 2. 数据来源

### 2.1 实验产物

| 类型 | 路径 | 内容 |
|---|---|---|
| baseline timing | `experiments/perf/baseline_timing-20260515_155426/` | 融合前 timing，14 个 case，每个 case 3 次 |
| baseline profile | `experiments/perf/baseline_profile-20260515_155701/` | 融合前 `nsys`、`ncu-basic`、`ncu-deep`，10 个代表 case |
| fused timing | `experiments/perf/fused_timing-20260515_193153/` | 融合后 timing，复用 baseline 14 个 case，每个 case 3 次 |
| fused profile | `experiments/perf/fused_profile-20260515_193220/` | 融合后 `nsys`、`ncu-basic`、`ncu-deep`，复用 baseline 10 个代表 case |
| fused thread-pool | `experiments/perf/fused_thread_pool-20260515_193413/` | 融合后 `queue-capacity=1,2,4,8` 扫描，含 timing、nsys 与 SQLite 并发指标 |
| 实例脚本 | `scripts/instances/fused_timing.sh`、`scripts/instances/fused_profile.sh`、`scripts/instances/fused_thread_pool.sh` | 可复跑实验配置 |
| 输入数据 | `experiments/cases/baseline/` | 复用 baseline `.mat` 与 `.txt` cases |

### 2.2 数据完整性

| 项目 | 数量 |
|---|---:|
| baseline timing `runs.jsonl` 记录 | 42 |
| baseline profile `runs.jsonl` 记录 | 30 |
| fused timing `runs.jsonl` 记录 | 42 |
| fused profile `runs.jsonl` 记录 | 30 |
| fused thread-pool `runs.jsonl` 记录 | 32 |
| fused thread-pool concurrency CSV 行数，含表头 | 17 |
| `failed` / `partial` 记录 | 0 |

`nsys` 记录中仍存在 warning，主要来自没有 NVTX 数据或部分 stats report 为空；本报告使用的 CUDA API summary、OS runtime summary、NCU metrics 与 timing JSON 均可用。所有参与分析的记录状态均为 `ok`。

### 2.3 术语说明

- 算子融合（kernel fusion）：把多个小 kernel 或多个调度阶段合并成更粗粒度的 GPU work，减少 CPU 侧提交与同步。
- cooperative kernel（协作内核）：CUDA 支持 grid 级同步的特殊 kernel launch 形式，代码中通过 `cudaLaunchCooperativeKernel` 启动。
- 控制面开销（control-plane overhead）：CPU 侧提交、同步、调度 GPU work 的开销，不等同于 GPU kernel 自身计算时间。
- kernel launch（内核发射）：CPU 侧向 CUDA runtime/driver 提交 GPU kernel 的动作。
- H2D/D2H copy（host-to-device / device-to-host copy）：主机和设备之间的数据传输。
- SM（Streaming Multiprocessor，流式多处理器）：NVIDIA GPU 上执行线程块的基本计算单元。
- throughput（吞吐率）：Nsight Compute 中相对于硬件峰值的利用率百分比。
- thread pool（线程池）：pipeline 侧并发执行 testcase 的 host worker 机制。

## 3. 代码路径观察

### 3.1 融合后路径

融合后主路径位于 `src/domain/jacobi_svd.cu` 的 `try_run_cooperative_sweeps`。核心结构是：

```text
每个 testcase:
  初始化 device matrices
  初始化 V
  对每个 sweep:
    cudaMemset(d_any_rotation)
    cudaLaunchCooperativeKernel(jacobi_sweep_kernel)
    cudaMemcpy(any_rotation D2H)
  构建 U/Sigma
  拷回输出
```

`jacobi_sweep_kernel` 位于 `src/domain/jacobi_rotation_kernels.cuh`，在 device 侧重算 round-robin schedule，并在每个 round 后执行 `grid.sync()`。这把旧路径的 round 级 host/device 控制流推进到了 device 内部。

### 3.2 融合前回退路径

旧路径仍在 `src/domain/jacobi_svd.cu` 的 `if (!ran_cooperative)` 分支中。它的 round 内结构近似为：

```text
每个 Jacobi round:
  H2D memcpy pairs
  cudaMemset convergence flag
  launch pair_stats_kernel
  launch compute_and_apply_rotation_kernel
  D2H memcpy convergence flag
```

也就是说，融合前的主要问题不是单个 kernel 太慢，而是每个 round 都把控制权交回 host，导致 kernel launch、small memcpy、memset 大量堆积。

## 4. Timing 层：融合前后端到端对比

### 4.1 全量 timing 对比

下表使用 `app_report.elapsed_ms` 的 3 次均值，并同时列出 `app ms/testcase`。融合前后所有 case 的 `total_sweeps` 完全一致，因此加速来自执行结构变化，而不是少做迭代。

| case | baseline app ms | fused app ms | app 加速 | baseline ms/testcase | fused ms/testcase | sweeps |
|---|---:|---:|---:|---:|---:|---:|
| `mat_grid_small_auto` | 2280.7 | 648.2 | 3.52x | 23.76 | 6.75 | 684 |
| `mat_ill_conditioned_auto` | 1342.3 | 291.9 | 4.60x | 33.56 | 7.30 | 452 |
| `mat_low_rank_auto` | 1173.5 | 314.9 | 3.73x | 29.34 | 7.87 | 403 |
| `mat_sparse_auto` | 758.9 | 245.2 | 3.09x | 18.97 | 6.13 | 261 |
| `mat_square_medium_auto` | 7108.4 | 905.1 | 7.85x | 148.09 | 18.86 | 512 |
| `mat_square_medium_off` | 7230.8 | 1069.8 | 6.76x | 150.64 | 22.29 | 512 |
| `mat_square_medium_on` | 6913.9 | 898.2 | 7.70x | 144.04 | 18.71 | 512 |
| `mat_tall_skinny_medium_auto` | 1430.9 | 311.3 | 4.60x | 22.36 | 4.86 | 460 |
| `mat_tall_skinny_medium_off` | 1446.6 | 346.7 | 4.17x | 22.60 | 5.42 | 460 |
| `mat_tall_skinny_medium_on` | 1396.4 | 360.5 | 3.87x | 21.82 | 5.63 | 460 |
| `mat_zero_columns_auto` | 819.4 | 260.8 | 3.14x | 20.48 | 6.52 | 257 |
| `txt_grid_small_auto` | 962.7 | 231.5 | 4.16x | 15.04 | 3.62 | 436 |
| `txt_ill_conditioned_small_auto` | 761.6 | 247.8 | 3.07x | 23.80 | 7.74 | 335 |
| `txt_sparse_small_auto` | 862.0 | 221.5 | 3.89x | 26.94 | 6.92 | 239 |

汇总：

| 指标 | 数值 |
|---|---:|
| app elapsed 几何平均加速 | 4.36x |
| wall-clock median 几何平均加速 | 4.31x |
| app elapsed 算术平均加速 | 4.58x |

### 4.2 Timing 观察

`mat_square_medium_*` 是最大受益组，app 加速约 `6.76x-7.85x`。这是符合预期的：该组列数更大，round-robin schedule 产生更多 round，融合前会制造最多的 round 级控制面开销。融合把控制粒度从 round 提升到 sweep 后，最重 case 的收益最大。

`mat_sparse_auto` 和小型 txt case 的加速也明显，但幅度较小，主要因为这些 case 的原始控制面压力较低，或者输入输出、分配释放等固定成本占比更高。

一个重要事实是：融合前后 sweep 数完全一致。当前数据不支持“融合改变收敛行为”这个解释；更合理的解释是，融合保留了数值迭代数量，同时显著减少 host 参与频率。

## 5. Nsight Systems 层：控制面调用规模

### 5.1 CUDA API 调用数对比

下表列出 10 个 profile case 中 `cudaLaunchKernel` 与 `cudaMemcpy` 的调用数变化。注意 fused 路径的主要 fused kernel 是 `cudaLaunchCooperativeKernel`，这里的 `cudaLaunchKernel` 主要来自初始化、转置、构建 U/Sigma 等普通 kernel；因此 launch 调用减少要结合 5.2 的 API 总表理解。

| case | baseline launch | fused launch | launch 减少 | baseline memcpy | fused memcpy | memcpy 减少 |
|---|---:|---:|---:|---:|---:|---:|
| `mat_grid_small_auto` | 56502 | 198 | 285.4x | 37920 | 1068 | 35.5x |
| `mat_ill_conditioned_auto` | 43444 | 136 | 319.4x | 29032 | 612 | 47.4x |
| `mat_sparse_auto` | 24207 | 126 | 192.1x | 16214 | 421 | 38.5x |
| `mat_square_medium_auto` | 232692 | 180 | 1292.7x | 155200 | 704 | 220.5x |
| `mat_square_medium_off` | 232608 | 96 | 2423.0x | 155200 | 704 | 220.5x |
| `mat_square_medium_on` | 232704 | 192 | 1212.0x | 155200 | 704 | 220.5x |
| `mat_tall_skinny_medium_auto` | 44268 | 216 | 204.9x | 29624 | 716 | 41.4x |
| `mat_tall_skinny_medium_off` | 44180 | 128 | 345.2x | 29624 | 716 | 41.4x |
| `mat_tall_skinny_medium_on` | 44308 | 256 | 173.1x | 29624 | 716 | 41.4x |
| `txt_grid_small_auto` | 29216 | 128 | 228.2x | 19648 | 692 | 28.4x |

10 个 profile case 的几何平均：

| 指标 | 减少倍数 |
|---|---:|
| `cudaLaunchKernel` 调用数 | 423.5x |
| `cudaMemcpy` 调用数 | 65.2x |

### 5.2 CUDA API 时间组成

Nsight Systems 的 CUDA API summed time 是 API 调用耗时求和，不等同于 wall-clock，因为 pipeline 有多线程并发。不过它能可靠揭示 runtime/driver 开销堆积位置。

融合前 10 个 profile case 合计：

| API | 调用数 | summed time s | 占比 |
|---|---:|---:|---:|
| `cudaLaunchKernel` | 984129 | 50.198 | 43.0% |
| `cudaMemcpy` | 657286 | 42.868 | 36.7% |
| `cudaMemset` | 327491 | 20.597 | 17.6% |
| `cudaMalloc` | 6588 | 1.246 | 1.1% |
| `cudaMallocHost` | 512 | 1.234 | 1.1% |
| `cudaFree` | 6588 | 0.517 | 0.4% |
| `cudaFreeHost` | 512 | 0.104 | 0.1% |

融合后 10 个 profile case 合计：

| API | 调用数 | summed time s | 占比 |
|---|---:|---:|---:|
| `cudaMemcpy` | 7053 | 5.871 | 45.2% |
| `cudaMalloc` | 3132 | 1.888 | 14.5% |
| `cudaMallocHost` | 512 | 1.795 | 13.8% |
| `cudaLaunchCooperativeKernel` | 4749 | 1.222 | 9.4% |
| `cudaMemset` | 4749 | 0.681 | 5.2% |
| `cudaFree` | 3132 | 0.644 | 5.0% |
| `cudaFreeHost` | 512 | 0.532 | 4.1% |
| `cudaLaunchKernel` | 1656 | 0.332 | 2.6% |
| `cudaDeviceSynchronize` | 576 | 0.015 | 0.1% |

### 5.3 Nsight Systems 观察

融合前的主导项是 `cudaLaunchKernel + cudaMemcpy + cudaMemset` 的 round 级爆炸。融合后，这三者的总量大幅下降，尤其 `cudaLaunchKernel` 从近百万次降到一千多次普通 launch，加上四千多次 cooperative launch。

更关键的是瓶颈结构变了：

1. `cudaMemcpy` 成为 summed time 最大项，占比约 `45.2%`。
2. `cudaMalloc/cudaFree` 与 `cudaMallocHost/cudaFreeHost` 变得显眼，合计约 `37.4%`。
3. `cudaLaunchCooperativeKernel + cudaMemset` 仍存在，但已经不是最大头。

这说明融合已经把原来的控制面大头打掉。下一阶段如果继续优化，优先级应从“减少 round 级 launch”转向“减少 per-testcase 分配释放、减少/隐藏 H2D/D2H copy、进一步减少 per-sweep host 交互”。

## 6. Nsight Compute 层：kernel 形态变化

### 6.1 NCU 分布对比

| 指标 | baseline NCU | fused NCU |
|---|---:|---:|
| 样本数 | 30 | 30 |
| kernel duration min | 4.352 us | 31.392 us |
| kernel duration median | 6.496 us | 280.048 us |
| kernel duration mean | 6.006 us | 389.374 us |
| kernel duration max | 7.008 us | 1605.792 us |
| SM throughput median | 1.985% | 7.075% |
| SM throughput mean | 3.380% | 9.138% |
| SM throughput max | 10.770% | 24.500% |
| compute/memory throughput median | 2.230% | 2.160% |
| compute/memory throughput mean | 3.087% | 4.016% |
| DRAM throughput median | 0.925% | 0.040% |

### 6.2 代表性 fused kernel 样本

| case | mode | kernel | duration us | SM % | memory % | DRAM % | grid | regs/thread | shared B/block |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mat_square_medium_auto` | `ncu-basic` | `jacobi_sweep_kernel<1>` | 976.19 | 20.84 | 9.66 | 0.04 | 48 | 36 | 7184 |
| `mat_square_medium_auto` | `ncu-deep` | `jacobi_sweep_kernel<1>` | 571.62 | 14.41 | 4.70 | 0.04 | 32 | 36 | 7184 |
| `mat_square_medium_off` | `ncu-basic` | `jacobi_sweep_kernel<0>` | 1605.79 | 24.50 | 25.40 | 0.04 | 64 | 36 | 7184 |
| `mat_sparse_auto` | `ncu-deep` | `jacobi_sweep_kernel<1>` | 583.04 | 15.74 | 4.64 | 0.08 | 32 | 36 | 7184 |
| `mat_grid_small_auto` | `ncu-deep` | `jacobi_sweep_kernel<0>` | 65.95 | 1.58 | 0.34 | 0.06 | 4 | 36 | 7184 |
| `txt_grid_small_auto` | `ncu-deep` | `jacobi_sweep_kernel<0>` | 31.39 | 0.71 | 0.40 | 0.12 | 2 | 36 | 7184 |

### 6.3 NCU 观察

融合后单个 kernel 的持续时间从几微秒提升到几十到一千多微秒。这正是想要的方向：GPU work 粒度变粗，launch overhead 被摊薄。

但 fused kernel 仍没有让 GPU 饱和：

- SM throughput 中位数约 `7.1%`，均值约 `9.1%`。
- 最好的局部样本也只有约 `24.5%` SM throughput。
- DRAM throughput 很低，中位数约 `0.04%`。

因此融合后的主瓶颈不像是设备端 DRAM 带宽打满，也不像是纯计算峰值打满。更像是算法并行度、同步结构和 host/pipeline 固定成本共同限制了吞吐。

## 7. 融合后瓶颈定位

### 7.1 第一层：数据搬运成为最大 CUDA API 项

融合后 `cudaMemcpy` 调用数从几十万级下降到几千级，但 summed time 仍占 `45.2%`。这说明 round 级小拷贝被消掉以后，剩下的大块输入输出拷贝和收敛标志拷贝变成主要可见成本。

在 `mat_square_medium_auto` 的 fused profile 中，`cudaMemcpy` 有 `704` 次，summed time 约 `1.317 s`，是该 case CUDA API summary 的最大项。由于输出包括 U、Sigma、V，且 pipeline 对每个 testcase 都生成结果，输出侧 D2H 不可能完全消失。优化方向更像是减少不必要中间拷贝、批量化输出或重审输出格式成本，而不是继续微调 round 级 pairs 拷贝。

### 7.2 第二层：分配释放成本显著抬头

融合后 10 个 profile case 中：

- `cudaMalloc + cudaFree` 合计约 `2.532 s`；
- `cudaMallocHost + cudaFreeHost` 合计约 `2.327 s`；
- 二者合计占 CUDA API summed time 约 `37.4%`。

这在融合前被 launch/memcpy/memset 洪水掩盖了。融合后它变成下一层真实瓶颈。当前每个 testcase 构造 `DeviceMatrix`、`DeviceBuffer`，并在输入输出路径中使用 pinned host memory。后续可考虑 worker-local 或 pipeline-local workspace，尤其是复用固定尺寸的 device buffer、V/U/Sigma buffer 和 pinned host staging buffer。

### 7.3 第三层：per-sweep host 交互仍存在

融合后每个 sweep 仍有：

```text
cudaMemset(d_any_rotation)
cudaLaunchCooperativeKernel(jacobi_sweep_kernel)
cudaMemcpy(any_rotation D2H)
```

这比旧路径每 round 都 host 交互好很多，但还不是“整个 testcase 一次 launch”。对于 `mat_square_medium_auto`，profile 中 `cudaLaunchCooperativeKernel` 与 `cudaMemset` 均为 `512` 次，正好对应该 case 的 total sweeps。

下一步如果要继续打控制面，方向不是恢复 round 级拆分，而是把 sweep 级 convergence check 尽量留在 device 端，或者批量执行多个 sweep 后再回 host 检查。代价是可能多做少量 sweep，或者需要更复杂的 device-side termination 机制。这个 trade-off 需要实测，不应只凭直觉决定。

### 7.4 第四层：cooperative kernel 内部并行度有限

`jacobi_sweep_kernel` 内部每个 round 处理 round-robin column pairs，并在 round 后 `grid.sync()`。这消除了 host round-trip，但引入了 device-side global synchronization。

对于列数不大的 case，grid size 很小：

- `txt_grid_small_auto` 的 NCU 样本可低至 `grid=2`；
- `mat_grid_small_auto` 可为 `grid=4`；
- 较重 case 常见 `grid=32/48/64`。

这些 grid size 很难填满 GPU。换句话说，融合后不是“每个 kernel 太短”，而是“每个 kernel 内部有很多同步阶段，每阶段可并行列对数由列数限制”。这是 Jacobi round-robin schedule 本身的结构性上限。

### 7.5 第五层：线程池收益开始分化

融合后 `queue-capacity` 扫描结果如下：

| case | q=1 app ms | q=2 app ms | q=4 app ms | q=8 app ms | 观察 |
|---|---:|---:|---:|---:|---|
| `mat_square_medium_auto` | 1720.6 | 992.0 | 812.3 | 788.5 | 并发仍明显有用 |
| `mat_sparse_auto` | 258.4 | 250.6 | 229.5 | 237.8 | q=4 略好 |
| `mat_ill_conditioned_auto` | 316.4 | 328.5 | 325.6 | 316.1 | 基本持平 |
| `mat_tall_skinny_medium_auto` | 307.0 | 349.4 | 347.4 | 333.3 | q=1 最好 |

对应的 SQLite 并发指标显示，`queue-capacity` 增大确实会提高 host-side CUDA Runtime active thread 数。例如 `mat_square_medium_auto`：

| queue | runtime span s | runtime total s | max active | avg active | active>=4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.883 | 2.076 | 3 | 2.35 | 0.0% |
| 2 | 0.865 | 2.688 | 4 | 3.11 | 60.0% |
| 4 | 0.825 | 3.427 | 6 | 4.15 | 73.1% |
| 8 | 0.837 | 4.764 | 10 | 5.69 | 76.6% |

但并发越高不一定越好。融合后 launch pressure 已经大幅降低，线程池不再是全局单调收益。重 case 仍能通过 overlap 获益；轻 case 可能因为 allocator、runtime、output queue、CPU 同步竞争而回落。

## 8. MECE 瓶颈归因

### 8.1 控制面开销：已从 P0 降为 P1/P2

融合前最强瓶颈是 round 级控制面开销。融合后它被显著缓解：kernel launch 和 memcpy 调用数下降一个到三个数量级。现在仍有 per-sweep `cudaLaunchCooperativeKernel + cudaMemset + D2H flag copy`，但不再是压倒性主因。

### 8.2 数据搬运：融合后最显眼

`cudaMemcpy` 现在是 CUDA API summary 最大项。它既包含输入输出，也包含 sweep convergence flag。由于输出矩阵本身不可避免，优化空间主要在减少中间拷贝、压缩输出路径、批量化小 D2H flag 或推迟 host 检查。

### 8.3 分配释放：被融合暴露出来的次级瓶颈

`cudaMalloc/cudaFree` 与 `cudaMallocHost/cudaFreeHost` 现在占比很高。workspace 复用是高优先级优化点，因为它不改变数值算法，风险比重写 kernel 低，且对所有 case 都可能有收益。

### 8.4 设备端计算：不是满载瓶颈

NCU 显示 fused kernel 的 SM throughput 提高，但仍偏低；DRAM throughput 很低。这说明继续优化单条 load/store 或 DRAM coalescing 不是当前最高杠杆。更关键的是增加有效并行度、减少同步阶段、让每次 cooperative kernel 内的工作更饱满。

### 8.5 Pipeline 并发：需要 workload-aware 调参

融合后 `queue-capacity` 的最优点随 workload 分化。简单把并发窗口调大不是好品味；更合理的是按输入规模或 case 形状决定并发窗口，或者把 output queue 背压与 GPU work 并发分开建模。

## 9. 工程建议

### P0：引入 workspace 复用

优先考虑 worker-local 或 pipeline-local workspace：

- 复用 `DeviceMatrix` / `DeviceBuffer`；
- 复用 `d_any_rotation`、`d_v`、`d_u`、`d_sigma`；
- 复用 pinned host staging buffer；
- 对相同或相近 shape 做容量保留，而不是每 testcase 重新 malloc/free。

这是当前最稳的下一步，因为它直接对应 fused profile 中暴露出的 allocation/free 成本，且不改变核心数值路径。

### P1：减少 per-sweep host convergence check

当前每 sweep 回 host 检查 `any_rotation`。可以评估：

1. 每 K 个 sweep 检查一次；
2. device-side loop + cooperative groups termination；
3. CUDA Graphs（CUDA 图）封装 sweep 重复结构；
4. 使用异步 copy 与 stream/event 隐藏 flag D2H。

风险是可能改变停止时机或增加多余 sweep，所以需要严格验证数值输出与 sweep count 变化。

### P1：重新审视输出与 copy 路径

融合后 `cudaMemcpy` 是最大 summed time 项。应拆分输入 H2D、输出 D2H、flag D2H 的比例，避免把所有 copy 混成一个桶。若输出格式允许，可评估：

- 批量写出；
- 减少 host 中间格式转换；
- 对 `.mat` 输出使用更直接的 staging；
- 避免不必要的 U/V/Sigma 临时复制。

### P2：改进 cooperative kernel 的并行度

`jacobi_sweep_kernel` 当前受列对数和 `grid.sync()` 限制。可探索：

- 多 testcase batching，让一个 cooperative kernel 同时处理多个小矩阵；
- 对小列数 case 走非 cooperative 或 batched 专用路径；
- 让一个 block 处理多个小 pair，减少 idle；
- 针对 tall-skinny/square 分别选择 grid 策略；
- 重审 round-robin schedule 是否能预生成到 device 常驻结构，减少 device 侧重复计算。

这类优化复杂度更高，应排在 workspace 和 copy 拆账之后。

### P2：重新调 queue-capacity

融合后 `queue-capacity=8` 对 `mat_square_medium_auto` 仍略好，但对 `mat_tall_skinny_medium_auto` 反而较差。建议不要设一个全局常数解决所有 workload。可以先做两个低风险方向：

1. 以输入 case 规模估计并发窗口；
2. 把 output queue capacity 与 GPU compute in-flight 数拆成两个参数。

## 10. 结论

融合 cooperative kernel 是一次成功的结构性优化。它保持相同 sweep 数，同时让 14 个 timing case 的 app elapsed 几何平均加速达到约 `4.36x`，`mat_square_medium_*` 达到约 `6.76x-7.85x`。Nsight Systems 进一步证明，融合把 `cudaLaunchKernel` 调用数几何平均减少约 `423.5x`，把 `cudaMemcpy` 调用数几何平均减少约 `65.2x`。

融合前瓶颈是 CPU 控制面把 GPU work 切得太碎：每个 Jacobi round 都发生 memcpy、memset 和 kernel launch。融合后这个问题被推进到下一层：数据搬运、device/host 分配释放、per-sweep host convergence check、cooperative kernel 内部同步和有限并行度开始成为主导。

下一步最值得做的是 workspace 复用与 copy 路径拆账，然后再评估 per-sweep convergence check 的 device 化或批量化。继续微调旧的 round 级 kernel 已经不是主线；现在应该围绕 fused path 把固定成本和同步结构继续压下去。喵。
