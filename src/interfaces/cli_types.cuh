#pragma once

#include "jacobi/svd/application/pipeline.cuh"

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace jacobi::svd::cli
{
    /**
     * @brief CLI 参数错误异常；CLI argument error exception.
     */
    class CliArgumentError final : public std::invalid_argument
    {
    public:
        /**
         * @brief 构造参数错误异常；Construct argument error exception.
         * @param message 错误消息；Error message.
         */
        explicit CliArgumentError(const std::string &message)
            : std::invalid_argument(message)
        {
        }
    };

    /**
     * @brief CLI 解析动作；CLI parse action.
     */
    enum class ParseAction
    {
        /**
         * @brief 正常执行 pipeline；Execute pipeline.
         */
        run,

        /**
         * @brief 输出帮助信息；Print help text.
         */
        help,

        /**
         * @brief 输出版本信息；Print version.
         */
        version
    };

    /**
     * @brief CLI 运行选项；CLI runtime options.
     */
    struct CliOptions final
    {
        /**
         * @brief 输入文件路径；Input file path.
         */
        std::filesystem::path input_path;

        /**
         * @brief 输出文件路径；Output file path.
         */
        std::filesystem::path output_path;

        /**
         * @brief 输入格式；Input format.
         */
        pipeline::MatrixFileFormat input_format = pipeline::MatrixFileFormat::auto_detect;

        /**
         * @brief 输出格式；Output format.
         */
        pipeline::MatrixFileFormat output_format = pipeline::MatrixFileFormat::auto_detect;

        /**
         * @brief 收敛阈值 epsilon；Convergence tolerance epsilon.
         */
        double epsilon = 1.0e-9;

        /**
         * @brief 最大 sweep 次数；Maximum sweep count.
         */
        int max_sweeps = 128;

        /**
         * @brief 每个 block 的线程数；Threads per CUDA block.
         */
        int threads_per_block = 256;

        /**
         * @brief 布局转置策略；Layout-transpose policy.
         */
        LayoutTransposeMode layout_transpose_mode = LayoutTransposeMode::auto_select;

        /**
         * @brief 自动策略下布局转置最小列数阈值；Auto-mode minimum-column threshold for layout transpose.
         */
        int layout_transpose_min_columns = 16;

        /**
         * @brief 自动策略下布局转置最小元素阈值；Auto-mode minimum-element threshold for layout transpose.
         */
        std::size_t layout_transpose_min_elements = 4096;

        /**
         * @brief 是否执行布局转置阈值自动调优；Whether to run layout-transpose threshold auto-tuning.
         */
        bool layout_transpose_auto_tune = false;

        /**
         * @brief 自动调优时每个尺寸的重复次数；Per-size repetitions during auto-tuning.
         */
        int layout_transpose_benchmark_repetitions = 2;

        /**
         * @brief 自动调优时每次基准的 sweep 上限；Sweep cap per benchmark run during auto-tuning.
         */
        int layout_transpose_benchmark_sweeps = 8;

        /**
         * @brief 输出队列容量；Output queue capacity.
         */
        std::size_t queue_capacity = 4;

        /**
         * @brief 鏄惁寮哄埗瑕嗙洊宸叉湁杈撳嚭锛沇hether to force overwriting existing output.
         */
        bool force_overwrite = false;

        /**
         * @brief 仅展示配置，不执行；Print configuration only without execution.
         */
        bool dry_run = false;

        /**
         * @brief 鏄惁杈撳嚭鐢熸晥閰嶇疆锛沇hether to print effective configuration before execution.
         */
        bool print_config = false;

        /**
         * @brief 是否输出 JSON 报告；Whether to emit JSON report.
         */
        bool json_report = false;

        /**
         * @brief 是否静默文本摘要；Whether to suppress text summary.
         */
        bool quiet = false;
    };

    /**
     * @brief CLI 解析结果；CLI parse result.
     */
    struct ParseResult final
    {
        /**
         * @brief 解析动作；Requested action.
         */
        ParseAction action = ParseAction::run;

        /**
         * @brief 解析得到的选项；Parsed options.
         */
        CliOptions options{};
    };

    /**
     * @brief 命令行选项标识；Identifier of one command-line option.
     */
    enum class OptionId
    {
        /**
         * @brief 输入路径选项；Input-path option.
         */
        input,

        /**
         * @brief 输出路径选项；Output-path option.
         */
        output,

        /**
         * @brief 输入格式选项；Input-format option.
         */
        input_format,

        /**
         * @brief 输出格式选项；Output-format option.
         */
        output_format,

        /**
         * @brief 统一格式选项（同时作用输入/输出）；Unified format option for both input/output.
         */
        format,

        /**
         * @brief epsilon 选项；epsilon option.
         */
        epsilon,

        /**
         * @brief 最大 sweep 选项；Maximum sweep option.
         */
        max_sweeps,

        /**
         * @brief 线程数选项；Threads-per-block option.
         */
        threads_per_block,

        /**
         * @brief 队列容量选项；Queue-capacity option.
         */
        queue_capacity,

        /**
         * @brief 布局转置策略选项；Layout-transpose policy option.
         */
        layout_transpose_mode,

        /**
         * @brief 布局转置最小列数阈值选项；Layout-transpose minimum-column threshold option.
         */
        layout_transpose_min_columns,

        /**
         * @brief 布局转置最小元素阈值选项；Layout-transpose minimum-element threshold option.
         */
        layout_transpose_min_elements,

        /**
         * @brief 布局转置阈值自动调优选项；Layout-transpose threshold auto-tuning option.
         */
        layout_transpose_auto_tune,

        /**
         * @brief 自动调优重复次数选项；Auto-tuning repetitions option.
         */
        layout_transpose_benchmark_repetitions,

        /**
         * @brief 自动调优 sweep 上限选项；Auto-tuning sweep-cap option.
         */
        layout_transpose_benchmark_sweeps,

        /**
         * @brief force 閫夐」锛沠orce-overwrite option.
         */
        force,

        /**
         * @brief dry-run 选项；dry-run option.
         */
        dry_run,

        /**
         * @brief print-config 閫夐」锛沜rint-config option.
         */
        print_config,

        /**
         * @brief JSON 报告选项；JSON-report option.
         */
        json_report,

        /**
         * @brief 静默选项；Quiet option.
         */
        quiet,

        /**
         * @brief 帮助选项；Help option.
         */
        help,

        /**
         * @brief 版本选项；Version option.
         */
        version
    };

    /**
     * @brief 单个选项定义；Single option definition.
     */
    struct OptionDefinition final
    {
        /**
         * @brief 选项标识；Option identifier.
         */
        OptionId id = OptionId::help;

        /**
         * @brief 长选项名（不含 `--`）；Long name without `--`.
         */
        std::string_view long_name{};

        /**
         * @brief 短选项名（不含 `-`，`\0` 表示无）；Short name without `-`, `\0` if absent.
         */
        char short_name = '\0';

        /**
         * @brief 是否需要参数值；Whether this option requires a value.
         */
        bool requires_value = false;

        /**
         * @brief 参数占位文本；Value placeholder text.
         */
        std::string_view value_hint{};

        /**
         * @brief 帮助说明；Help description.
         */
        std::string_view description{};
    };
} // namespace jacobi::svd::cli
