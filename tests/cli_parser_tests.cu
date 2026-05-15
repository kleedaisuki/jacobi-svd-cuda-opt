#include "src/interfaces/cli.cuh"

#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    /**
     * @brief 测试失败异常；Test failure exception.
     */
    class TestFailure final : public std::runtime_error
    {
    public:
        /**
         * @brief 构造测试失败异常；Construct test failure exception.
         * @param message 失败消息；Failure message.
         */
        explicit TestFailure(const std::string &message)
            : std::runtime_error(message)
        {
        }
    };

    /**
     * @brief 检查条件并在失败时抛出异常；Check condition and throw on failure.
     * @param condition 待检查条件；Condition to check.
     * @param message 失败消息；Failure message.
     */
    void require(bool condition, std::string_view message)
    {
        if (!condition)
        {
            throw TestFailure(std::string(message));
        }
    }

    /**
     * @brief 将字符串参数转为 argv 数组；Convert string arguments to argv array.
     * @param arguments 参数字符串；Argument strings.
     * @return argv 指针数组；argv pointer array.
     */
    [[nodiscard]] std::vector<char *> make_argv(std::vector<std::string> &arguments)
    {
        std::vector<char *> argv;
        argv.reserve(arguments.size());
        for (std::string &argument : arguments)
        {
            argv.push_back(argument.data());
        }
        return argv;
    }

    /**
     * @brief 解析参数集合；Parse one argument set.
     * @param arguments 参数字符串；Argument strings.
     * @return 解析结果；Parse result.
     */
    [[nodiscard]] jacobi::svd::cli::ParseResult parse_arguments(std::vector<std::string> arguments)
    {
        const jacobi::svd::cli::ArgParser parser{};
        std::vector<char *> argv = make_argv(arguments);
        return parser.parse(static_cast<int>(argv.size()), argv.data());
    }

    /**
     * @brief 验证 help 动作不要求输入文件；Verify help action does not require input.
     */
    void test_help_short_circuits_validation()
    {
        const jacobi::svd::cli::ParseResult result = parse_arguments({"jacobi-svd-cuda", "--help"});
        require(result.action == jacobi::svd::cli::ParseAction::help, "help should select help action");
    }

    /**
     * @brief 验证 dry-run 会规范化默认输出路径；Verify dry-run normalizes default output path.
     */
    void test_dry_run_normalizes_default_output()
    {
        const std::filesystem::path input = std::filesystem::temp_directory_path() / "jacobi-cli-parser-input.txt";
        {
            std::ofstream stream(input);
            stream << "1 0\n0 1\n";
        }

        const jacobi::svd::cli::ParseResult result =
            parse_arguments({"jacobi-svd-cuda", "--dry-run", "--input", input.string()});
        std::filesystem::remove(input);

        require(result.options.dry_run, "dry-run flag should be set");
        require(result.options.output_path.filename() == "jacobi-cli-parser-input.svd.txt",
                "default output should preserve input format extension");
        require(result.options.input_format == jacobi::svd::pipeline::MatrixFileFormat::auto_detect,
                "input format should remain auto when extension is known");
        require(result.options.output_format == jacobi::svd::pipeline::MatrixFileFormat::txt,
                "default output format should be inferred from input extension");
    }

    /**
     * @brief 验证短选项组合和值解析；Verify short-option grouping and value parsing.
     */
    void test_short_options_accept_attached_values()
    {
        const std::filesystem::path input = std::filesystem::temp_directory_path() / "jacobi-cli-parser-input.mat";
        {
            std::ofstream stream(input, std::ios::binary);
            stream << "placeholder";
        }

        const jacobi::svd::cli::ParseResult result =
            parse_arguments({"jacobi-svd-cuda", "--dry-run", "-fmat", "-e1e-8", "-s64", input.string()});
        std::filesystem::remove(input);

        require(result.options.input_format == jacobi::svd::pipeline::MatrixFileFormat::mat,
                "attached -f value should set input format");
        require(result.options.output_format == jacobi::svd::pipeline::MatrixFileFormat::mat,
                "attached -f value should set output format");
        require(result.options.epsilon == 1.0e-8, "attached -e value should set epsilon");
        require(result.options.max_sweeps == 64, "attached -s value should set max sweeps");
    }

    /**
     * @brief 验证无效数值会报参数错误；Verify invalid numeric values report argument errors.
     */
    void test_invalid_positive_integer_is_rejected()
    {
        try
        {
            (void)parse_arguments({"jacobi-svd-cuda", "--dry-run", "--max-sweeps", "0", "input.mat"});
        }
        catch (const jacobi::svd::cli::CliArgumentError &)
        {
            return;
        }
        throw TestFailure("zero max-sweeps should be rejected");
    }
} // namespace

/**
 * @brief 测试入口；Test entry point.
 * @return 进程退出码；Process exit code.
 */
int main()
{
    try
    {
        test_help_short_circuits_validation();
        test_dry_run_normalizes_default_output();
        test_short_options_accept_attached_values();
        test_invalid_positive_integer_is_rejected();
    }
    catch (const std::exception &error)
    {
        std::cerr << "cli_parser_tests failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "cli_parser_tests passed\n";
    return 0;
}
