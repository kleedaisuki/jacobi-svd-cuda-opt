# 项目结构

本项目使用 C++20 与 CUDA，实现单边雅可比奇异值分解（One-sided Jacobi SVD）。当前代码按领域驱动设计（Domain-Driven Design, DDD）的边界组织：命令行接口层负责参数解析与报告输出，应用层负责把输入、核函数计算和输出串成 pipeline，领域层负责 CUDA SVD 算法与 GPU 资源封装，基础设施层负责矩阵文件格式、内存映射（Memory-Mapped File）与页锁定主机内存（Pinned Host Memory）。

## ASCII Tree

```text
jacobi-svd-cuda-opt/
├── CMakeLists.txt
├── LICENSE
├── STRUCTURE.md
├── include/
│   └── jacobi/
│       └── svd/
│           ├── application/
│           │   └── pipeline.hpp
│           ├── domain/
│           │   ├── cuda_error.hpp
│           │   ├── device_matrix.hpp
│           │   ├── jacobi_svd.hpp
│           │   ├── jacobi_svd_config.hpp
│           │   ├── jacobi_svd_result.hpp
│           │   ├── kernels.hpp
│           │   └── layout_transpose.hpp
│           └── io/
│               ├── files.hpp
│               ├── io.hpp
│               ├── mat_dispatch.hpp
│               ├── mat_file.hpp
│               ├── mat_metadata.hpp
│               ├── matrix.hpp
│               ├── matrix_stream.hpp
│               ├── pinned_host_task_buffer.hpp
│               └── txt_file.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── application/
│   │   ├── kernel_stage.hpp
│   │   ├── output_stage.hpp
│   │   ├── pipeline.cu
│   │   ├── pipeline_detail.hpp
│   │   ├── pipeline_helpers.cu
│   │   ├── result_writer.hpp
│   │   ├── text_testcase_source.hpp
│   │   └── thread_pool.hpp
│   ├── domain/
│   │   ├── cuda_check.cuh
│   │   ├── cuda_error.cu
│   │   ├── device_buffer.cuh
│   │   ├── device_matrix.cu
│   │   ├── jacobi_rotation_kernels.cuh
│   │   ├── jacobi_schedule.cuh
│   │   ├── jacobi_svd.cu
│   │   ├── jacobi_svd_detail.hpp
│   │   ├── layout_transpose_autotune.cu
│   │   ├── layout_transpose_kernels.cuh
│   │   └── matrix_index.cuh
│   ├── infrastructure/
│   │   ├── append_mapped_output_file.hpp
│   │   ├── files.cu
│   │   ├── io_detail.hpp
│   │   ├── mat_dispatch_reader.cu
│   │   ├── mat_file.cu
│   │   ├── memory_mapped_input_file.hpp
│   │   ├── pinned_host_task_buffer.cu
│   │   └── txt_file.cu
│   └── interfaces/
│       ├── arg_parser.hpp
│       ├── cli.hpp
│       ├── cli_actions.hpp
│       ├── cli_parser.cu
│       ├── cli_support.cu
│       ├── cli_types.hpp
│       └── main.cu
└── tests/
    ├── CMakeLists.txt
    └── cli_parser_tests.cu
```

`build/` 是 CMake/NVCC 生成目录，不属于源码结构；`.git/`、`.vscode/` 等本地工具目录也不纳入架构说明。

## 构建结构

顶层 `CMakeLists.txt` 只承担项目装配职责：声明项目、设置项目级选项、加入 `src/`，并在显式开启 `JACOBI_BUILD_TESTS` 时加入 `tests/`。源码清单、主程序目标、NVCC driver mode/native CUDA mode 的构建细节都下沉到 `src/CMakeLists.txt`。

这种拆分让根目录不需要知道接口层、应用层、领域层和基础设施层各自有哪些 `.cu` 文件；后续新增算法源文件或调整 executable 构建方式时，改动主要局限在 `src/` 内。

## 测试结构

`tests/` 存放 CTest 驱动的单元测试（Unit Test）。顶层 `CMakeLists.txt` 只保留 `JACOBI_BUILD_TESTS` 项目级开关，以及让根构建目录可被 `ctest` 发现测试所需的最小 CTest 初始化；测试目标、测试标签和 `check` 目标都放在 `tests/` 子目录内，避免测试细节污染根构建脚本。

当前测试基础设施兼容两条构建路径：

- 默认 NVCC driver mode：测试目标通过与主程序一致的 `nvcc` 参数构建，产物写入 `build/`。
- CMake native CUDA language mode：测试目标使用普通 `add_executable()`，便于以后接入更细粒度的 target 属性、sanitizer 或 IDE 工具链。

日常验证可以使用：

```bash
cmake -S . -B build -DJACOBI_BUILD_TESTS=ON
cmake --build build --target check
```

其中 `check` 会先构建测试可执行文件，再调用 `ctest --output-on-failure`。这避免了手动运行 `ctest` 时测试二进制尚未构建的问题。

## 分层说明

### 1. Interface Layer

`src/interfaces/` 是 CLI 表示层。`main.cu` 是进程入口，`ArgParser` 解析命令行参数，`cli_support.cu` 负责把用户输入转换为 `PipelineConfig`，并输出文本报告、JSON 报告、dry-run 配置、帮助信息和版本信息。

当前 CLI 支持输入/输出路径、输入/输出格式、统一格式选项、收敛阈值 `epsilon`、最大 sweep 次数、CUDA block 线程数、布局转置策略、布局转置阈值、阈值自动调优、输出队列容量、覆盖输出、dry-run、打印配置、JSON 报告和 quiet 模式。

### 2. Application Layer

`include/jacobi/svd/application/pipeline.hpp` 暴露应用层公共接口：`PipelineConfig`、`PipelineReport`、`JacobiSvdPipeline` 与 `run_pipeline()`。

`src/application/` 实现 `testcases -> kernel -> output` 的执行流：

- `pipeline.cu` 是聚合根实现，负责解析输入/输出格式、可选执行布局转置阈值自动调优、提交计算任务、关闭输出阶段并生成运行报告。
- `kernel_stage.hpp` 把单个输入矩阵转换为 `U`、`Sigma`、`V` 三个输出矩阵。
- `output_stage.hpp` 使用 future 队列（Future Queue）和消费者线程写出结果，队列容量由 `max_queued_results` 控制。
- `result_writer.hpp` 根据输出格式选择 `MatOutputStream` 或 `TxtOutputStream`，每个 testcase 固定写出三张矩阵：`U`、`Sigma(1 x n)`、`V`。
- `text_testcase_source.hpp` 封装文本输入流；二进制 `.mat` 输入在 `pipeline.cu` 中使用 `MatDispatchReader` 单游标派发。
- `thread_pool.hpp` 提供全局线程池（Thread Pool），使多个 testcase 的 CPU 解析与 GPU 调用可以通过 future 编排。
- `pipeline_helpers.cu` 和 `pipeline_detail.hpp` 提供格式解析、输出目录检查、矩阵尺寸校验、溢出检查和内部 `OutputPacket`。

这里的设计重点不是“多线程写同一个文件”，而是让计算任务并行提交，写线程按提交顺序消费 future。这样结果顺序稳定，同时避免把全部结果留在内存中。

### 3. Domain Layer

`include/jacobi/svd/domain/` 是算法公共 API：

- `jacobi_svd.hpp` 暴露 `one_sided_jacobi_svd()` 与 `auto_tune_layout_transpose_threshold()`。
- `jacobi_svd_config.hpp` 定义收敛阈值、最大 sweep、线程数、布局转置策略与自动调优参数。
- `jacobi_svd_result.hpp` 定义主机侧结果容器：`U`、`Sigma`、`V` 和实际 sweep 数。
- `device_matrix.hpp` 封装行主序设备矩阵（Device Matrix）的生命周期，kernel 侧只接收裸指针。
- `layout_transpose.hpp` 定义布局转置策略（Layout Transpose Policy）与自动调优报告。
- `kernels.hpp` 是领域层聚合头文件。

`src/domain/` 是 CUDA 实现细节：

- `jacobi_svd.cu` 校验输入，按配置选择直接行主序路径或布局转置路径，分配设备矩阵，初始化 `V`，执行 round-robin 列对调度，最后构造 `U` 与 `Sigma`。
- `jacobi_rotation_kernels.cuh` 包含核心 CUDA kernels：初始化单位矩阵、计算列对统计量、计算 Givens 旋转参数、应用旋转、从收敛后的 `A` 构造 `U/Sigma`。
- `jacobi_schedule.cuh` 构建巡回赛调度（Round-Robin Schedule），保证同一 round 内列对互不冲突；奇数列通过 dummy 列处理。
- `layout_transpose_kernels.cuh` 用共享内存 tile 实现行主序和列布局之间的转置转换，提升列操作的访存连续性。
- `layout_transpose_autotune.cu` 扫描一组矩阵尺寸，比较直接路径和转置路径平均耗时，并推荐阈值。
- `device_matrix.cu` 和 `device_buffer.cuh` 管理 GPU 内存的 RAII（Resource Acquisition Is Initialization）生命周期。
- `matrix_index.cuh` 集中提供行主序/列主序索引映射。
- `cuda_check.cuh` 和 `cuda_error.cu` 统一 CUDA runtime 错误处理。

当前实现假设输入矩阵满足 `rows >= columns`。这不是数学上的 SVD 限制，而是当前单边雅可比实现的工程前提；调用方如果要支持宽矩阵，需要在外层增加转置或另一条算法路径。

### 4. Infrastructure Layer

`include/jacobi/svd/io/` 是矩阵输入输出公共 API：

- `matrix.hpp` 定义行主序矩阵容器 `Matrix`。
- `mat_metadata.hpp` 定义 `.mat` 文件头 `MatMetaData { rows, columns }`，磁盘中使用网络字节序（Network Byte Order）。
- `matrix_stream.hpp` 定义矩阵输入/输出 policy 概念（Policy Concept），并用 `MatrixInputStream`、`MatrixOutputStream` 解耦流控制与文件格式。
- `mat_file.hpp` 和 `txt_file.hpp` 定义 `.mat` 与 `.txt` 的 policy、reader、writer。
- `mat_dispatch.hpp` 定义 `.mat` 单游标派发读取器 `MatDispatchReader` 与 `MatDispatchTask`，用于 pipeline 的低驻留内存读取。
- `pinned_host_task_buffer.hpp` 暴露页锁定任务缓冲区。
- `files.hpp` 保留批量读写兼容接口，`io.hpp` 是 IO 聚合头。

`src/infrastructure/` 是这些 API 的实现：

- `mat_file.cu` 使用内存映射输入和追加式映射输出处理 `.mat` 矩阵流。
- `txt_file.cu` 使用文本流处理空格分隔、换行分隔的矩阵；矩阵之间以空行分隔。
- `mat_dispatch_reader.cu` 逐条读取 `.mat` 元数据与 payload，把原始网络字节序 payload 放入 `PinnedHostTaskBuffer`，再由工作线程解码。
- `pinned_host_task_buffer.cu` 使用 `cudaMallocHost/cudaFreeHost` 管理一块连续的输入区加工作区。
- `memory_mapped_input_file.hpp` 和 `append_mapped_output_file.hpp` 封装平台相关的映射文件操作。
- `io_detail.hpp` 提供字节序转换、payload 尺寸检查、矩阵序列化/反序列化等共用细节。
- `files.cu` 实现 `read_mat_file/write_mat_file/read_txt_file/write_txt_file` 兼容函数。

## 文件格式

二进制 `.mat` 文件是矩阵流，由若干条记录顺序组成。每条记录先写入元数据，再写入矩阵 payload：

```cpp
struct MatMetaData
{
    std::uint64_t rows;
    std::uint64_t columns;
};
```

元数据字段在磁盘上使用网络字节序，矩阵元素使用 `double`。文本 `.txt` 格式以空格分隔同一行的元素，以换行分隔矩阵行，以空行分隔不同矩阵。

输入输出格式可以显式指定为 `mat` 或 `txt`，也可以使用 `auto` 按扩展名 `.mat`/`.txt` 推断。

## 执行模型

一次 pipeline 运行的主要路径如下：

```text
CLI
  -> PipelineConfig
  -> input format resolution
  -> optional layout-transpose auto-tune
  -> testcase source
  -> GlobalThreadPool futures
  -> KernelStage / one_sided_jacobi_svd
  -> FutureQueueOutputStage
  -> ResultWriter
  -> PipelineReport
```

`.mat` 输入走单游标派发：主线程顺序读取每条记录并把 payload 放入页锁定缓冲区，工作线程负责解码和计算。`.txt` 输入走 `TxtInputStream`，逐张矩阵读出后提交给线程池。

输出阶段只由一个消费者线程写文件，但它消费的是 future 队列。这个选择把“有序输出”作为默认行为，而不是额外的特殊分支：提交顺序就是写出顺序，计算慢的 testcase 会在对应 future 上阻塞，后续结果不会越序写入。

## 算法实现

核心算法是单边雅可比 SVD。对输入矩阵 `A(m x n)`，算法维护右奇异矩阵 `V(n x n)`，对列对 `(p, q)` 计算：

- `a_pp = dot(A_p, A_p)`
- `a_qq = dot(A_q, A_q)`
- `a_pq = dot(A_p, A_q)`

若 `abs(a_pq) > epsilon * sqrt(a_pp * a_qq)`，则用稳定的 Rutishauser 公式计算 Givens 旋转（Givens Rotation），同时更新 `A` 的两列和 `V` 的两列。所有 sweep 结束后，`A` 的列范数成为奇异值 `Sigma`，归一化后的列成为 `U`。

CUDA 并行化的关键是 `build_round_robin_schedule()`：同一 round 内每个列号最多出现一次，因此多个 block 可以同时处理不同列对而不写同一列。每个列对的统计量在 block 内用共享内存归约（Shared-Memory Reduction）得到，旋转参数单独计算，再应用到 `A` 与 `V`。

由于算法天然按列访问，直接行主序矩阵会产生跨步访问（Strided Access）。当前实现提供两条路径：

- 直接路径：`A` 保持行主序，索引映射处理列访问。
- 布局转置路径：先把逻辑 `A(rows x columns)` 转为列连续布局，在旋转阶段改善访存，结束后再转回行主序以构造 `U/Sigma`。

`LayoutTransposeMode::auto_select` 会根据 `layout_transpose_min_columns` 和 `layout_transpose_min_elements` 决定是否使用转置路径；也可以通过 CLI 强制开启、强制关闭，或在运行前执行微基准自动调优。
