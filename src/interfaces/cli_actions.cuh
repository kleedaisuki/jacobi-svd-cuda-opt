#pragma once

#include "src/interfaces/arg_parser.cuh"

#include <cstddef>
#include <string>
#include <string_view>

namespace jacobi::svd::cli
{
/**
 * @brief 统一小写化（ASCII）；Convert text to lowercase (ASCII).
 * @param text 输入文本；Input text.
 * @return 小写文本；Lowercased text.
 */
[[nodiscard]] std::string to_lower_ascii(std::string_view text);

/**
 * @brief 解析矩阵格式文本；Parse matrix format string.
 * @param raw 文本值；Raw text value.
 * @param option_name 选项名；Option name.
 * @return 文件格式枚举；Matrix file format enum.
 */
[[nodiscard]] pipeline::MatrixFileFormat parse_matrix_format(std::string_view raw, std::string_view option_name);

/**
 * @brief 解析布局转置策略文本；Parse layout-transpose mode string.
 * @param raw 文本值；Raw text value.
 * @param option_name 选项名；Option name.
 * @return 布局转置策略；Layout-transpose mode.
 */
[[nodiscard]] LayoutTransposeMode parse_layout_transpose_mode(std::string_view raw, std::string_view option_name);

/**
 * @brief 将文件格式转为字符串；Convert file format to string.
 * @param format 格式枚举；Format enum.
 * @return 字符串表示；String representation.
 */
[[nodiscard]] std::string_view matrix_format_to_string(pipeline::MatrixFileFormat format);

/**
 * @brief 将布局转置策略转为字符串；Convert layout-transpose mode to string.
 * @param mode 布局转置策略；Layout-transpose mode.
 * @return 字符串表示；String representation.
 */
[[nodiscard]] std::string_view layout_transpose_mode_to_string(LayoutTransposeMode mode);

/**
 * @brief 解析正整数；Parse positive integer.
 * @param raw 原始文本；Raw text.
 * @param option_name 选项名；Option name.
 * @return 正整数；Positive integer.
 */
[[nodiscard]] int parse_positive_int(std::string_view raw, std::string_view option_name);

/**
 * @brief 解析正 size_t；Parse positive size_t.
 * @param raw 原始文本；Raw text.
 * @param option_name 选项名；Option name.
 * @return 正 size_t；Positive size_t.
 */
[[nodiscard]] std::size_t parse_positive_size(std::string_view raw, std::string_view option_name);

/**
 * @brief 解析正 double；Parse positive double.
 * @param raw 原始文本；Raw text.
 * @param option_name 选项名；Option name.
 * @return 正 double；Positive double.
 */
[[nodiscard]] double parse_positive_double(std::string_view raw, std::string_view option_name);

/**
 * @brief 规范化运行选项；Normalize run options.
 * @param options CLI 选项；CLI options.
 */
void normalize_run_options(CliOptions &options);

/**
 * @brief 校验运行选项；Validate run options.
 * @param options CLI 选项；CLI options.
 */
void validate_run_options(const CliOptions &options);

/**
 * @brief 将 CLI 选项转换为 Pipeline 配置；Convert CLI options to pipeline config.
 * @param options CLI 选项；CLI options.
 * @return Pipeline 配置；Pipeline configuration.
 */
[[nodiscard]] pipeline::PipelineConfig make_pipeline_config(const CliOptions &options);

/**
 * @brief 打印 dry-run 配置；Print dry-run configuration.
 * @param options CLI 选项；CLI options.
 * @param include_dry_run_banner 是否输出 dry-run 标题；Whether to print dry-run banner.
 */
void print_dry_run_config(const CliOptions &options, bool include_dry_run_banner);

/**
 * @brief 打印文本报告；Print text report.
 * @param report Pipeline 报告；Pipeline report.
 * @param elapsed_milliseconds 耗时毫秒；Elapsed milliseconds.
 */
void print_text_report(const pipeline::PipelineReport &report, double elapsed_milliseconds);

/**
 * @brief 打印 JSON 报告；Print JSON report.
 * @param report Pipeline 报告；Pipeline report.
 * @param elapsed_milliseconds 耗时毫秒；Elapsed milliseconds.
 */
void print_json_report(const pipeline::PipelineReport &report, double elapsed_milliseconds);
} // namespace jacobi::svd::cli
