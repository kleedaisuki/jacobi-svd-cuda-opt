# 线程池并发效果分析

## 1. 分析目标

本文独立记录 baseline 实验后的线程池（Thread Pool）分析，回答三个问题：

1. 当前 pipeline 的线程池是否真的制造了并发？
2. 这种并发是否提高了端到端吞吐率（throughput）？
3. 线程池并发与当前主要性能瓶颈之间是什么关系？

本文不替代 `baseline-performance-analysis-20260515.md`，而是作为其补充。结论基于已经落盘的 baseline profile 数据，以及一次针对 `mat_square_medium_auto` 的小型 queue-capacity 对照补测。

## 2. 数据来源

| 类型 | 路径或命令 | 说明 |
|---|---|---|
| Nsight Systems 原始 SQLite | `experiments/prof/baseline_profile-20260515_155701/*/nsys/run_1_nsys.sqlite` | 用于统计 host thread 的 CUDA Runtime API 并发 |
| profile 提取结果 | `experiments/perf/baseline_profile-20260515_155701/` | 用于核对 CUDA API 调用规模 |
| timing 提取结果 | `experiments/perf/baseline_timing-20260515_155426/` | 用于参考 baseline wall time 与 app elapsed |
| queue-capacity 补测 | 直接运行 `build/jacobi-svd-cuda` | 只改变 `--queue-capacity`，输入为 `mat_square_medium.mat` |

补测命令模板：

```bash
/usr/bin/time -f 'wall=%e user=%U sys=%S rss=%M' \
  build/jacobi-svd-cuda \
  experiments/cases/baseline/mat/square_medium.mat \
  /tmp/jacobi_square_q${q}.mat \
  --force \
  --json-report \
  --max-sweeps 32 \
  --threads-per-block 256 \
  --queue-capacity "${q}" \
  --format mat \
  --layout-transpose-mode auto
```

## 3. 代码路径观察

pipeline 在 `src/application/pipeline.cu` 中把每个 testcase 提交到全局线程池：

- `.mat` 路径提交点：`src/application/pipeline.cu:70-79`
- `.txt` 路径提交点：`src/application/pipeline.cu:95-103`
- 全局线程池实现：`src/application/thread_pool.cuh`
- 异步输出队列：`src/application/output_stage.cuh`

需要注意一个容易误解的点：`--queue-capacity` 限制的是 future/output 队列中允许排队的结果数量，也间接限制前端可提交的 in-flight testcase 数。它不是 GPU stream 数，也不是显式的 CUDA kernel worker 数。

因此，`queue-capacity` 是一个端到端 pipeline 背压参数，而不是纯粹的 GPU 并发参数。

## 4. Nsight Systems 并发证据

对 `mat_square_medium_auto` 的 Nsight Systems SQLite 统计 CUDA Runtime API 的 host thread 活动，得到：

| 指标 | 数值 |
|---|---:|
| CUDA Runtime API 活动跨度 | `8.91 s` |
| CUDA Runtime API 累计耗时 | `29.79 s` |
| 参与 CUDA Runtime 调用的 host thread 数 | `22` |
| 同时处于 CUDA Runtime API 的最大 active thread 数 | `4` |
| 平均 active thread 数 | `3.34` |
| 至少 2 个 active thread 的时间占比 | `95.8%` |
| 至少 4 个 active thread 的时间占比 | `59.9%` |

这说明线程池不是摆设。它确实让多个 host thread 并发进入 CUDA Runtime，也确实让多个 testcase 在 host 侧并发推进。

另外两个代表 case：

| case | runtime span s | runtime total s | host threads | max active | avg active | time active>=2 | time active>=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mat_square_medium_auto` | 8.91 | 29.79 | 22 | 4 | 3.34 | 95.8% | 59.9% |
| `mat_tall_skinny_medium_auto` | 1.67 | 3.64 | 22 | 4 | 2.18 | 66.2% | 17.3% |
| `mat_grid_small_auto` | 2.02 | 6.51 | 22 | 4 | 3.23 | 89.9% | 55.2% |

这里的 active thread 只表示 CPU thread 正在 CUDA Runtime API 中执行，不等价于 GPU kernel 同时执行，也不等价于更高的 SM occupancy。它证明的是 host-side concurrency，而不是 device-side saturation。

## 5. queue-capacity 对照

为了确认并发是否带来端到端收益，对 `mat_square_medium_auto` 做了单 case 补测：

| queue-capacity | wall s | app elapsed ms | user s | sys s | max RSS KB |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.34 | 8308.679 | 5.53 | 3.56 | 132952 |
| 2 | 6.78 | 6764.825 | 5.70 | 3.81 | 132612 |
| 4 | 6.38 | 6355.138 | 5.80 | 3.74 | 130448 |
| 8 | 6.84 | 6792.245 | 5.91 | 3.72 | 135212 |

相对 `queue-capacity=1`：

| queue-capacity | wall 改善 |
|---:|---:|
| 2 | 约 `18.7%` |
| 4 | 约 `23.5%` |
| 8 | 约 `18.0%` |

这说明线程池并发确实提高了吞吐率，但存在饱和点。对这个代表 case，`queue-capacity=4` 最好，`queue-capacity=8` 开始回落。

## 6. 为什么并发有效

当前每个 testcase 内部包含大量短小 CUDA work。线程池让多个 testcase 交错推进，可以隐藏一部分：

- kernel launch 固定开销；
- 小 `cudaMemcpy` 的同步等待；
- output 写出等待；
- CPU 侧读取和 decode 的延迟；
- driver/runtime 内部等待。

从端到端 wall time 看，`queue-capacity=2/4` 明显好于 `queue-capacity=1`，说明这种 overlap 是真实收益，不是测量幻觉。

## 7. 为什么并发不是根治

线程池没有改变单个 testcase 的 Jacobi round 结构。核心路径仍然近似为：

```text
每个 Jacobi round:
  1 x H2D memcpy pairs
  1 x cudaMemset convergence flag
  1 x pair_stats_kernel launch
  1 x compute_rotation_params_kernel launch
  1 x apply_rotation_kernel launch
  1 x D2H memcpy convergence flag
```

在 `mat_square_medium_*` 上，profile 中约有：

- `232k` 次 `cudaLaunchKernel`
- `155k` 次 `cudaMemcpy`
- `77.5k` 次 `cudaMemset`

这些调用来自 round 级粒度本身。线程池可以把多个 testcase 的这些操作交错起来，但不能消灭这些操作。

换句话说，线程池是在提高流水线吞吐，而不是降低单 testcase 的结构性控制面开销（control-plane overhead）。当 in-flight testcase 太多时，额外并发还可能增加 CUDA Runtime、driver 调度、allocator、输出队列和 CPU 线程竞争，所以 `queue-capacity=8` 回落并不意外。

## 8. 工程判断

### 8.1 线程池是否有效

有效。Nsight Systems 证明多个 host thread 并发进入 CUDA Runtime；queue-capacity 对照证明这种并发能降低 wall time。

### 8.2 是否应该继续增大并发

不应该盲目增大。`queue-capacity=8` 已经比 `4` 慢。当前更合理的判断是：并发窗口需要调优，而不是越大越好。

### 8.3 当前最佳猜测

对 `mat_square_medium_auto` 这个代表 case，`queue-capacity=4` 暂时最好。但这个结论不应直接推广到所有 workload。不同 shape、不同 sweep 数、不同 output size 下，最佳点可能变化。

### 8.4 与主瓶颈的关系

线程池是有效的缓解手段，但主瓶颈仍然是 round 级控制面开销。优化优先级不应从“减少 per-round host/device 往返和 kernel launch”转移到“继续堆线程”。

## 9. 后续实验建议

1. 系统扫描 `queue-capacity=1,2,3,4,6,8`，覆盖 square、tall-skinny、ill-conditioned、sparse 四类 case。
2. 区分 output queue capacity 与 kernel worker 并发，避免一个参数同时承担背压和计算并发两种语义。
3. 在 P0/P1 优化后重新扫描 queue-capacity，因为 round 粒度变粗以后，最优并发点可能下降。
4. 如果引入 CUDA Graphs 或 device-resident schedule，应重新评估线程池价值，因为 host-side launch pressure 会显著变化。
5. 对多 case 并发进行 GPU timeline 层验证，而不仅看 CUDA Runtime API active thread。

## 10. 结论

线程池确实提高了并发，也确实提升了当前 baseline 的端到端吞吐。对 `mat_square_medium_auto`，`queue-capacity=2/4` 明显快于 `1`，其中 `4` 在单次补测中最好。

但线程池不是根治。它没有改变每个 Jacobi round 的 `memcpy + memset + 3 kernel launch + memcpy` 结构，也不能消除由几十万次 CUDA API 调用带来的控制面开销。下一步最值得做的仍然是重写 round 级数据流，让 schedule 常驻 device、减少 per-round D2H/H2D，同步考虑 kernel fusion 或 CUDA Graphs。线程池应保留并调优，但它不是这轮性能优化的主战场。喵。
