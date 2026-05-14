#include "src/interfaces/cli.hpp"

#include "jacobi/svd/application/pipeline.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

/**
 * @brief 程序入口；Program entry point.
 * @param argc 参数个数；Argument count.
 * @param argv 参数数组；Argument vector.
 * @return 进程退出码；Process exit code.
 */
int main(int argc, char *argv[])
{
    try
    {
        const std::string executable =
            (argc > 0 && argv[0] != nullptr) ? std::filesystem::path(argv[0]).filename().string() : "jacobi-svd";

        const jacobi::svd::cli::ArgParser parser{};
        const jacobi::svd::cli::ParseResult parsed = parser.parse(argc, argv);

        if (parsed.action == jacobi::svd::cli::ParseAction::help)
        {
            std::cout << parser.help_message(executable);
            return 0;
        }
        if (parsed.action == jacobi::svd::cli::ParseAction::version)
        {
            std::cout << "jacobi-svd-cuda CLI v0.1.0\n";
            return 0;
        }

        if (parsed.options.dry_run)
        {
            jacobi::svd::cli::print_dry_run_config(parsed.options, true);
            return 0;
        }

        if (parsed.options.print_config)
        {
            std::cout << "Effective configuration:\n";
            jacobi::svd::cli::print_dry_run_config(parsed.options, false);
            std::cout << '\n';
        }

        const jacobi::svd::pipeline::PipelineConfig config =
            jacobi::svd::cli::make_pipeline_config(parsed.options);
        const auto started_at = std::chrono::steady_clock::now();
        const jacobi::svd::pipeline::PipelineReport report = jacobi::svd::pipeline::run_pipeline(config);
        const auto finished_at = std::chrono::steady_clock::now();
        const double elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(finished_at - started_at).count();

        if (!parsed.options.quiet)
        {
            jacobi::svd::cli::print_text_report(report, elapsed_milliseconds);
        }
        if (parsed.options.json_report)
        {
            jacobi::svd::cli::print_json_report(report, elapsed_milliseconds);
        }

        return 0;
    }
    catch (const jacobi::svd::cli::CliArgumentError &error)
    {
        std::cerr << "Argument error: " << error.what() << '\n';
        std::cerr << "Use --help for usage.\n";
        return 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }
}
