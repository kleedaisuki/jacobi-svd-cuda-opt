#include "src/interfaces/cli.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace jacobi::svd::cli
{
    /**
     * @brief 统一小写化（ASCII）；Convert text to lowercase (ASCII).
     * @param text 输入文本；Input text.
     * @return 小写文本；Lowercased text.
     */
    [[nodiscard]] std::string to_lower_ascii(std::string_view text)
    {
        std::string lowered(text);
        std::transform(lowered.begin(),
                       lowered.end(),
                       lowered.begin(),
                       [](const unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        return lowered;
    }

    /**
     * @brief 解析矩阵格式文本；Parse matrix format string.
     * @param raw 文本值；Raw text value.
     * @param option_name 选项名；Option name.
     * @return 文件格式枚举；Matrix file format enum.
     */
    [[nodiscard]] pipeline::MatrixFileFormat parse_matrix_format(std::string_view raw, std::string_view option_name)
    {
        const std::string lowered = to_lower_ascii(raw);
        if (lowered == "auto")
        {
            return pipeline::MatrixFileFormat::auto_detect;
        }
        if (lowered == "mat")
        {
            return pipeline::MatrixFileFormat::mat;
        }
        if (lowered == "txt")
        {
            return pipeline::MatrixFileFormat::txt;
        }

        throw CliArgumentError("Invalid value for --" + std::string(option_name) +
                               ": expected one of {auto, mat, txt}.");
    }

    /**
     * @brief 解析布局转置策略文本；Parse layout-transpose mode string.
     * @param raw 文本值；Raw text value.
     * @param option_name 选项名；Option name.
     * @return 布局转置策略；Layout-transpose mode.
     */
    [[nodiscard]] LayoutTransposeMode parse_layout_transpose_mode(std::string_view raw, std::string_view option_name)
    {
        const std::string lowered = to_lower_ascii(raw);
        if (lowered == "auto")
        {
            return LayoutTransposeMode::auto_select;
        }
        if (lowered == "on")
        {
            return LayoutTransposeMode::force_enable;
        }
        if (lowered == "off")
        {
            return LayoutTransposeMode::force_disable;
        }

        throw CliArgumentError("Invalid value for --" + std::string(option_name) +
                               ": expected one of {auto, on, off}.");
    }

    /**
     * @brief 将文件格式转为字符串；Convert file format to string.
     * @param format 格式枚举；Format enum.
     * @return 字符串表示；String representation.
     */
    [[nodiscard]] std::string_view matrix_format_to_string(pipeline::MatrixFileFormat format)
    {
        switch (format)
        {
        case pipeline::MatrixFileFormat::auto_detect:
            return "auto";
        case pipeline::MatrixFileFormat::mat:
            return "mat";
        case pipeline::MatrixFileFormat::txt:
            return "txt";
        }

        return "unknown";
    }

    /**
     * @brief 将布局转置策略转为字符串；Convert layout-transpose mode to string.
     * @param mode 布局转置策略；Layout-transpose mode.
     * @return 字符串表示；String representation.
     */
    [[nodiscard]] std::string_view layout_transpose_mode_to_string(LayoutTransposeMode mode)
    {
        switch (mode)
        {
        case LayoutTransposeMode::auto_select:
            return "auto";
        case LayoutTransposeMode::force_enable:
            return "on";
        case LayoutTransposeMode::force_disable:
            return "off";
        }
        return "unknown";
    }

    /**
     * @brief 由文件扩展名推断矩阵格式；Infer matrix format from file extension.
     * @param path 文件路径；File path.
     * @return 若可识别返回格式，否则返回空；Returns format when recognized, otherwise empty.
     */
    [[nodiscard]] std::optional<pipeline::MatrixFileFormat> detect_matrix_format_from_extension(
        const std::filesystem::path &path)
    {
        const std::string lowered = to_lower_ascii(path.extension().string());
        if (lowered == ".mat")
        {
            return pipeline::MatrixFileFormat::mat;
        }
        if (lowered == ".txt")
        {
            return pipeline::MatrixFileFormat::txt;
        }
        return std::nullopt;
    }

    /**
     * @brief 获取格式对应默认扩展名；Get canonical extension of one format.
     * @param format 目标格式；Target format.
     * @return 扩展名字符串；File extension string.
     */
    [[nodiscard]] std::string canonical_extension_for_format(pipeline::MatrixFileFormat format)
    {
        if (format == pipeline::MatrixFileFormat::txt)
        {
            return ".txt";
        }
        return ".mat";
    }

    /**
     * @brief 解析正整数；Parse positive integer.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed integer.
     */
    [[nodiscard]] int parse_positive_int(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        int value = 0;
        try
        {
            value = std::stoi(text, &parsed, 10);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }
        if (value <= 0)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be positive.");
        }
        return value;
    }

    /**
     * @brief 解析正整数（size_t）；Parse positive size_t integer.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed size.
     */
    [[nodiscard]] std::size_t parse_positive_size(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        unsigned long long value = 0ULL;
        try
        {
            value = std::stoull(text, &parsed, 10);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid integer value for --" + std::string(option_name) + ".");
        }
        if (value == 0ULL)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be positive.");
        }
        return static_cast<std::size_t>(value);
    }

    /**
     * @brief 解析正浮点数；Parse positive floating-point value.
     * @param raw 文本值；Raw value text.
     * @param option_name 选项名；Option name.
     * @return 解析结果；Parsed floating-point value.
     */
    [[nodiscard]] double parse_positive_double(std::string_view raw, std::string_view option_name)
    {
        const std::string text(raw);
        std::size_t parsed = 0;
        double value = 0.0;
        try
        {
            value = std::stod(text, &parsed);
        }
        catch (const std::exception &)
        {
            throw CliArgumentError("Invalid floating-point value for --" + std::string(option_name) + ".");
        }

        if (parsed != text.size())
        {
            throw CliArgumentError("Invalid floating-point value for --" + std::string(option_name) + ".");
        }
        if (!std::isfinite(value) || value <= 0.0)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " must be a positive finite number.");
        }
        return value;
    }

    /**
     * @brief 规范化路径用于比较；Normalize path for equality comparison.
     * @param path 输入路径；Input path.
     * @return 规范化路径；Normalized path.
     */
    [[nodiscard]] std::filesystem::path normalized_path_for_compare(const std::filesystem::path &path)
    {
        std::error_code error;
        const std::filesystem::path absolute = std::filesystem::absolute(path, error);
        if (error)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    /**
     * @brief 规范化运行选项；Normalize run-time options.
     * @param options CLI 选项（原地修改）；CLI options (modified in-place).
     */
    void normalize_run_options(CliOptions &options)
    {
        if (options.input_path.empty())
        {
            throw CliArgumentError("Missing input file. Use --input <path>.");
        }

        const std::optional<pipeline::MatrixFileFormat> input_ext_format =
            detect_matrix_format_from_extension(options.input_path);
        if (options.input_format == pipeline::MatrixFileFormat::auto_detect && !input_ext_format.has_value())
        {
            throw CliArgumentError("Cannot infer input format from extension. Use --input-format {mat|txt}.");
        }

        if (options.output_path.empty())
        {
            pipeline::MatrixFileFormat output_format = options.output_format;
            if (output_format == pipeline::MatrixFileFormat::auto_detect)
            {
                if (options.input_format != pipeline::MatrixFileFormat::auto_detect)
                {
                    output_format = options.input_format;
                }
                else if (input_ext_format.has_value())
                {
                    output_format = input_ext_format.value();
                }
                else
                {
                    output_format = pipeline::MatrixFileFormat::mat;
                }
                options.output_format = output_format;
            }

            std::string stem = options.input_path.stem().string();
            if (stem.empty())
            {
                stem = "result";
            }
            options.output_path = options.input_path.parent_path() /
                                  (stem + ".svd" + canonical_extension_for_format(output_format));
        }

        const std::optional<pipeline::MatrixFileFormat> output_ext_format =
            detect_matrix_format_from_extension(options.output_path);
        if (options.output_format == pipeline::MatrixFileFormat::auto_detect && !output_ext_format.has_value())
        {
            pipeline::MatrixFileFormat fallback = pipeline::MatrixFileFormat::mat;
            if (options.input_format != pipeline::MatrixFileFormat::auto_detect)
            {
                fallback = options.input_format;
            }
            else if (input_ext_format.has_value())
            {
                fallback = input_ext_format.value();
            }

            if (options.output_path.extension().empty())
            {
                options.output_path += canonical_extension_for_format(fallback);
            }
            options.output_format = fallback;
        }
    }

    /**
     * @brief 校验运行选项；Validate run-time options.
     * @param options CLI 选项；CLI options.
     */
    void validate_run_options(const CliOptions &options)
    {
        if (!std::filesystem::exists(options.input_path))
        {
            throw CliArgumentError("Input file does not exist: " + options.input_path.string());
        }
        if (std::filesystem::is_directory(options.input_path))
        {
            throw CliArgumentError("Input path is a directory, expected a file: " + options.input_path.string());
        }
        if (std::filesystem::is_directory(options.output_path))
        {
            throw CliArgumentError("Output path points to a directory: " + options.output_path.string());
        }

        const std::filesystem::path lhs = normalized_path_for_compare(options.input_path);
        const std::filesystem::path rhs = normalized_path_for_compare(options.output_path);
        if (lhs == rhs)
        {
            throw CliArgumentError("Input and output paths must be different files.");
        }

        if (std::filesystem::exists(options.output_path) && !options.force_overwrite)
        {
            throw CliArgumentError("Output file already exists. Use --force to overwrite: " +
                                   options.output_path.string());
        }
    }

    /**
     * @brief 将 CLI 选项转换为 Pipeline 配置；Convert CLI options to pipeline config.
     * @param options CLI 选项；CLI options.
     * @return Pipeline 配置；Pipeline configuration.
     */
    [[nodiscard]] pipeline::PipelineConfig make_pipeline_config(const CliOptions &options)
    {
        pipeline::PipelineConfig config{};
        config.input_path = options.input_path;
        config.output_path = options.output_path;
        config.input_format = options.input_format;
        config.output_format = options.output_format;
        config.max_queued_results = options.queue_capacity;
        config.kernel_config = JacobiSvdConfig{
            .epsilon = options.epsilon,
            .max_sweeps = options.max_sweeps,
            .threads_per_block = options.threads_per_block,
            .layout_transpose_mode = options.layout_transpose_mode,
            .layout_transpose_min_columns = options.layout_transpose_min_columns,
            .layout_transpose_min_elements = options.layout_transpose_min_elements,
            .layout_transpose_auto_tune = options.layout_transpose_auto_tune,
            .layout_transpose_benchmark_repetitions = options.layout_transpose_benchmark_repetitions,
            .layout_transpose_benchmark_sweeps = options.layout_transpose_benchmark_sweeps,
        };
        return config;
    }

    /**
     * @brief 输出 dry-run 配置摘要；Print dry-run configuration summary.
     * @param options CLI 选项；CLI options.
     */
    void print_dry_run_config(const CliOptions &options, bool include_dry_run_banner)
    {
        if (include_dry_run_banner)
        {
            std::cout << "Dry run: pipeline was not executed.\n";
        }
        std::cout << "input           : " << options.input_path.string() << '\n';
        std::cout << "output          : " << options.output_path.string() << '\n';
        std::cout << "input-format    : " << matrix_format_to_string(options.input_format) << '\n';
        std::cout << "output-format   : " << matrix_format_to_string(options.output_format) << '\n';
        std::cout << "epsilon         : " << options.epsilon << '\n';
        std::cout << "max-sweeps      : " << options.max_sweeps << '\n';
        std::cout << "threads-per-blk : " << options.threads_per_block << '\n';
        std::cout << "layout-mode     : " << layout_transpose_mode_to_string(options.layout_transpose_mode) << '\n';
        std::cout << "layout-min-cols : " << options.layout_transpose_min_columns << '\n';
        std::cout << "layout-min-elem : " << options.layout_transpose_min_elements << '\n';
        std::cout << "layout-auto-tune: " << (options.layout_transpose_auto_tune ? "true" : "false") << '\n';
        std::cout << "layout-bench-rep: " << options.layout_transpose_benchmark_repetitions << '\n';
        std::cout << "layout-bench-swp: " << options.layout_transpose_benchmark_sweeps << '\n';
        std::cout << "queue-capacity  : " << options.queue_capacity << '\n';
        std::cout << "force-overwrite : " << (options.force_overwrite ? "true" : "false") << '\n';
    }

    /**
     * @brief 输出文本执行报告；Print text execution report.
     * @param report Pipeline 报告；Pipeline report.
     */
    void print_text_report(const pipeline::PipelineReport &report, double elapsed_milliseconds)
    {
        std::cout << "Pipeline completed.\n";
        std::cout << "testcases       : " << report.testcase_count << '\n';
        std::cout << "emitted-matrices: " << report.emitted_matrix_count << '\n';
        std::cout << "total-sweeps    : " << report.total_sweeps << '\n';
        std::cout << "layout-mode     : " << layout_transpose_mode_to_string(report.layout_transpose_mode) << '\n';
        std::cout << "layout-min-cols : " << report.layout_transpose_min_columns << '\n';
        std::cout << "layout-min-elem : " << report.layout_transpose_min_elements << '\n';
        std::cout << "layout-auto-tune: " << (report.layout_transpose_auto_tuned ? "true" : "false") << '\n';
        std::cout << "layout-best-spd : " << std::fixed << std::setprecision(3)
                  << report.layout_transpose_estimated_best_speedup << '\n';
        std::cout << "elapsed-ms      : " << std::fixed << std::setprecision(3)
                  << elapsed_milliseconds << '\n';
    }

    /**
     * @brief 输出 JSON 执行报告；Print JSON execution report.
     * @param report Pipeline 报告；Pipeline report.
     */
    void print_json_report(const pipeline::PipelineReport &report, double elapsed_milliseconds)
    {
        std::cout << "{\n";
        std::cout << "  \"testcase_count\": " << report.testcase_count << ",\n";
        std::cout << "  \"emitted_matrix_count\": " << report.emitted_matrix_count << ",\n";
        std::cout << "  \"total_sweeps\": " << report.total_sweeps << ",\n";
        std::cout << "  \"layout_transpose_mode\": \""
                  << layout_transpose_mode_to_string(report.layout_transpose_mode) << "\",\n";
        std::cout << "  \"layout_transpose_min_columns\": " << report.layout_transpose_min_columns << ",\n";
        std::cout << "  \"layout_transpose_min_elements\": " << report.layout_transpose_min_elements << ",\n";
        std::cout << "  \"layout_transpose_auto_tuned\": "
                  << (report.layout_transpose_auto_tuned ? "true" : "false") << ",\n";
        std::cout << "  \"layout_transpose_estimated_best_speedup\": " << std::fixed << std::setprecision(3)
                  << report.layout_transpose_estimated_best_speedup << ",\n";
        std::cout << "  \"elapsed_ms\": " << std::fixed << std::setprecision(3)
                  << elapsed_milliseconds << '\n';
        std::cout << "}\n";
    }
} // namespace jacobi::svd::cli
