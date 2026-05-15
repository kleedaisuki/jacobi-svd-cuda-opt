#include "src/interfaces/cli.cuh"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace jacobi::svd::cli
{
    ArgParser::ArgParser()
        : definitions_{
              {OptionId::input, "input", 'i', true, "PATH", "Input matrix stream file path."},
              {OptionId::output, "output", 'o', true, "PATH", "Output matrix stream file path."},
              {OptionId::input_format, "input-format", '\0', true, "FMT", "Input format: auto|mat|txt."},
              {OptionId::output_format, "output-format", '\0', true, "FMT", "Output format: auto|mat|txt."},
              {OptionId::format, "format", 'f', true, "FMT", "Set both input/output format: auto|mat|txt."},
              {OptionId::epsilon, "epsilon", 'e', true, "NUM", "Convergence epsilon (>0)."},
              {OptionId::max_sweeps, "max-sweeps", 's', true, "N", "Maximum sweep count (>0)."},
              {OptionId::threads_per_block, "threads-per-block", 't', true, "N", "CUDA threads per block (>0)."},
              {OptionId::layout_transpose_mode, "layout-transpose-mode", '\0', true, "MODE",
               "Layout-transpose mode: auto|on|off."},
              {OptionId::layout_transpose_min_columns, "layout-transpose-min-cols", '\0', true, "N",
               "Auto-mode threshold: minimum columns (>0)."},
              {OptionId::layout_transpose_min_elements, "layout-transpose-min-elems", '\0', true, "N",
               "Auto-mode threshold: minimum elements (>0)."},
              {OptionId::layout_transpose_auto_tune, "layout-transpose-auto-tune", '\0', false, "",
               "Run micro-benchmark to auto-tune layout thresholds before pipeline."},
              {OptionId::layout_transpose_benchmark_repetitions, "layout-transpose-bench-reps", '\0', true, "N",
               "Auto-tune repetitions per scanned matrix size (>0)."},
              {OptionId::layout_transpose_benchmark_sweeps, "layout-transpose-bench-sweeps", '\0', true, "N",
               "Auto-tune sweep cap per benchmark run (>0)."},
              {OptionId::queue_capacity, "queue-capacity", 'c', true, "N", "In-flight/reorder window size (>0)."},
              {OptionId::force, "force", 'y', false, "", "Overwrite existing output file."},
              {OptionId::dry_run, "dry-run", '\0', false, "", "Validate arguments and print config only."},
              {OptionId::print_config, "print-config", '\0', false, "", "Print effective config before execution."},
              {OptionId::json_report, "json-report", '\0', false, "", "Print execution report in JSON."},
              {OptionId::quiet, "quiet", 'q', false, "", "Suppress text report."},
              {OptionId::help, "help", 'h', false, "", "Show this help message."},
              {OptionId::version, "version", 'v', false, "", "Show version information."},
          }
    {
    }

    ParseResult ArgParser::parse(int argc, char *const argv[]) const
    {
        ParseResult result{};
        std::vector<std::string_view> positional_arguments;

        std::size_t index = 1;
        while (index < static_cast<std::size_t>(argc))
        {
            const std::string_view token(argv[index] == nullptr ? "" : argv[index]);
            if (token.empty())
            {
                ++index;
                continue;
            }

            if (token == "--")
            {
                ++index;
                while (index < static_cast<std::size_t>(argc))
                {
                    positional_arguments.emplace_back(argv[index] == nullptr ? "" : argv[index]);
                    ++index;
                }
                break;
            }

            if (token.rfind("--", 0) == 0)
            {
                parse_long_option(token, argv, argc, index, result);
                ++index;
                continue;
            }

            if (token.size() > 1 && token[0] == '-')
            {
                parse_short_options(token, argv, argc, index, result);
                ++index;
                continue;
            }

            positional_arguments.push_back(token);
            ++index;
        }

        std::size_t positional_index = 0;
        if (result.options.input_path.empty() && positional_index < positional_arguments.size())
        {
            result.options.input_path = std::filesystem::path(std::string(positional_arguments[positional_index]));
            ++positional_index;
        }
        if (result.options.output_path.empty() && positional_index < positional_arguments.size())
        {
            result.options.output_path = std::filesystem::path(std::string(positional_arguments[positional_index]));
            ++positional_index;
        }
        if (positional_index < positional_arguments.size())
        {
            throw CliArgumentError("Unexpected positional argument: " +
                                   std::string(positional_arguments[positional_index]));
        }

        if (result.action == ParseAction::run)
        {
            normalize_run_options(result.options);
            validate_run_options(result.options);
        }

        return result;
    }

    std::string ArgParser::help_message(std::string_view executable) const
    {
        std::ostringstream stream;
        stream << "Jacobi SVD CUDA CLI\n\n";
        stream << "Usage:\n";
        stream << "  " << executable << " [OPTIONS] <input> [output]\n\n";
        stream << "Options:\n";

        for (const OptionDefinition &option : definitions_)
        {
            std::ostringstream names;
            if (option.short_name != '\0')
            {
                names << '-' << option.short_name << ", ";
            }
            else
            {
                names << "    ";
            }

            names << "--" << option.long_name;
            if (option.requires_value)
            {
                names << ' ' << option.value_hint;
            }

            stream << "  " << names.str();
            const std::size_t padding = names.str().size() < 32U ? (32U - names.str().size()) : 1U;
            stream << std::string(padding, ' ');
            stream << option.description << '\n';
        }

        stream << "\nExamples:\n";
        stream << "  " << executable << " -i experiments/inputs/a.mat -o experiments/outputs/r.mat\n";
        stream << "  " << executable << " experiments/inputs/a.mat --print-config\n";
        stream << "  " << executable << " input.txt output.txt --format txt --epsilon 1e-10\n";
        stream << "  " << executable << " --input a.mat --output b.txt --output-format txt --json-report --force\n";
        stream << "  " << executable
               << " --input a.mat --layout-transpose-auto-tune --layout-transpose-mode auto\n";
        stream << "\nNotes:\n";
        stream << "  - When [output] is omitted, default output is <input-stem>.svd.{mat|txt}.\n";
        stream << "  - Existing output file requires --force to overwrite.\n";
        return stream.str();
    }

    void ArgParser::parse_long_option(std::string_view token,
                                      char *const argv[],
                                      int argc,
                                      std::size_t &index,
                                      ParseResult &result) const
    {
        const std::string_view body = token.substr(2);
        if (body.empty())
        {
            throw CliArgumentError("Invalid option token '--'.");
        }

        const std::size_t eq_pos = body.find('=');
        const std::string_view option_name = (eq_pos == std::string_view::npos) ? body : body.substr(0, eq_pos);
        const bool has_inline_value = (eq_pos != std::string_view::npos);
        const std::string_view inline_value = has_inline_value ? body.substr(eq_pos + 1) : std::string_view{};

        const OptionDefinition *option = find_long_option(option_name);
        if (option == nullptr)
        {
            throw CliArgumentError("Unknown option: --" + std::string(option_name));
        }

        if (option->requires_value)
        {
            const std::string_view value =
                has_inline_value ? inline_value : consume_next_value(option->long_name, argv, argc, index);
            apply_option(*option, value, result);
            return;
        }

        if (has_inline_value)
        {
            throw CliArgumentError("Option --" + std::string(option->long_name) + " does not take a value.");
        }
        apply_option(*option, std::nullopt, result);
    }

    void ArgParser::parse_short_options(std::string_view token,
                                        char *const argv[],
                                        int argc,
                                        std::size_t &index,
                                        ParseResult &result) const
    {
        if (token.size() <= 1)
        {
            throw CliArgumentError("Invalid short option token.");
        }

        std::size_t offset = 1;
        while (offset < token.size())
        {
            const char short_name = token[offset];
            const OptionDefinition *option = find_short_option(short_name);
            if (option == nullptr)
            {
                throw CliArgumentError(std::string("Unknown short option: -") + short_name);
            }

            if (!option->requires_value)
            {
                apply_option(*option, std::nullopt, result);
                ++offset;
                continue;
            }

            std::string_view value;
            if (offset + 1 < token.size())
            {
                value = token.substr(offset + 1);
            }
            else
            {
                value = consume_next_value(option->long_name, argv, argc, index);
            }

            apply_option(*option, value, result);
            break;
        }
    }

    void ArgParser::apply_option(const OptionDefinition &option,
                                 const std::optional<std::string_view> &value,
                                 ParseResult &result) const
    {
        switch (option.id)
        {
        case OptionId::input:
            result.options.input_path = std::filesystem::path(std::string(value.value()));
            return;
        case OptionId::output:
            result.options.output_path = std::filesystem::path(std::string(value.value()));
            return;
        case OptionId::input_format:
            result.options.input_format = parse_matrix_format(value.value(), option.long_name);
            return;
        case OptionId::output_format:
            result.options.output_format = parse_matrix_format(value.value(), option.long_name);
            return;
        case OptionId::format:
        {
            const pipeline::MatrixFileFormat format = parse_matrix_format(value.value(), option.long_name);
            result.options.input_format = format;
            result.options.output_format = format;
            return;
        }
        case OptionId::epsilon:
            result.options.epsilon = parse_positive_double(value.value(), option.long_name);
            return;
        case OptionId::max_sweeps:
            result.options.max_sweeps = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::threads_per_block:
            result.options.threads_per_block = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_mode:
            result.options.layout_transpose_mode = parse_layout_transpose_mode(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_min_columns:
            result.options.layout_transpose_min_columns = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_min_elements:
            result.options.layout_transpose_min_elements = parse_positive_size(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_auto_tune:
            result.options.layout_transpose_auto_tune = true;
            return;
        case OptionId::layout_transpose_benchmark_repetitions:
            result.options.layout_transpose_benchmark_repetitions = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::layout_transpose_benchmark_sweeps:
            result.options.layout_transpose_benchmark_sweeps = parse_positive_int(value.value(), option.long_name);
            return;
        case OptionId::queue_capacity:
            result.options.queue_capacity = parse_positive_size(value.value(), option.long_name);
            return;
        case OptionId::force:
            result.options.force_overwrite = true;
            return;
        case OptionId::dry_run:
            result.options.dry_run = true;
            return;
        case OptionId::print_config:
            result.options.print_config = true;
            return;
        case OptionId::json_report:
            result.options.json_report = true;
            return;
        case OptionId::quiet:
            result.options.quiet = true;
            return;
        case OptionId::help:
            result.action = ParseAction::help;
            return;
        case OptionId::version:
            result.action = ParseAction::version;
            return;
        }
    }

    const OptionDefinition *ArgParser::find_long_option(std::string_view name) const
    {
        const auto iterator = std::find_if(definitions_.begin(),
                                           definitions_.end(),
                                           [name](const OptionDefinition &option) {
                                               return option.long_name == name;
                                           });
        if (iterator == definitions_.end())
        {
            return nullptr;
        }
        return &(*iterator);
    }

    const OptionDefinition *ArgParser::find_short_option(char name) const
    {
        const auto iterator = std::find_if(definitions_.begin(),
                                           definitions_.end(),
                                           [name](const OptionDefinition &option)
                                           {
                                               return option.short_name == name;
                                           });
        if (iterator == definitions_.end())
        {
            return nullptr;
        }
        return &(*iterator);
    }

    std::string_view ArgParser::consume_next_value(std::string_view option_name,
                                                   char *const argv[],
                                                   int argc,
                                                   std::size_t &index) const
    {
        const std::size_t next_index = index + 1;
        if (next_index >= static_cast<std::size_t>(argc) || argv[next_index] == nullptr)
        {
            throw CliArgumentError("Option --" + std::string(option_name) + " requires a value.");
        }

        index = next_index;
        return std::string_view(argv[index]);
    }
} // namespace jacobi::svd::cli
