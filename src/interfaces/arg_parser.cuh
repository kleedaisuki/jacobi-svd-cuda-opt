#pragma once

#include "src/interfaces/cli_types.cuh"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jacobi::svd::cli
{
    /**
     * @brief CLI 参数解析器；CLI argument parser.
     */
    class ArgParser final
    {
    public:
        /**
         * @brief 构造解析器并注册选项；Construct parser and register options.
         */
        ArgParser();

        /**
         * @brief 解析命令行参数；Parse command-line arguments.
         * @param argc 参数个数；Argument count.
         * @param argv 参数数组；Argument vector.
         * @return 解析结果；Parse result.
         */
        [[nodiscard]] ParseResult parse(int argc, char *const argv[]) const;

        /**
         * @brief 生成帮助文本；Build help message.
         * @param executable 可执行文件名；Executable name.
         * @return 帮助文本；Help text.
         */
        [[nodiscard]] std::string help_message(std::string_view executable) const;

    private:
        /**
         * @brief 解析长选项；Parse one long option token.
         * @param token 当前 token；Current token.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @param result 解析结果；Parse result.
         */
        void parse_long_option(std::string_view token,
                               char *const argv[],
                               int argc,
                               std::size_t &index,
                               ParseResult &result) const;

        /**
         * @brief 解析短选项；Parse one short-option token.
         * @param token 当前 token；Current token.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @param result 解析结果；Parse result.
         */
        void parse_short_options(std::string_view token,
                                 char *const argv[],
                                 int argc,
                                 std::size_t &index,
                                 ParseResult &result) const;

        /**
         * @brief 应用选项到结果对象；Apply one option to parse result.
         * @param option 选项定义；Option definition.
         * @param value 选项值（若有）；Option value if any.
         * @param result 解析结果；Parse result.
         */
        void apply_option(const OptionDefinition &option,
                          const std::optional<std::string_view> &value,
                          ParseResult &result) const;

        /**
         * @brief 查找长选项定义；Find option by long name.
         * @param name 长选项名；Long option name.
         * @return 匹配定义指针；Pointer to matching definition.
         */
        [[nodiscard]] const OptionDefinition *find_long_option(std::string_view name) const;

        /**
         * @brief 查找短选项定义；Find option by short name.
         * @param name 短选项字符；Short option character.
         * @return 匹配定义指针；Pointer to matching definition.
         */
        [[nodiscard]] const OptionDefinition *find_short_option(char name) const;

        /**
         * @brief 从后续参数中取值；Consume next argv token as value.
         * @param option_name 选项名（用于报错）；Option name for diagnostics.
         * @param argv 参数数组；Argument vector.
         * @param argc 参数个数；Argument count.
         * @param index 当前索引（可前移）；Current index (can advance).
         * @return 选项值；Consumed option value.
         */
        [[nodiscard]] std::string_view consume_next_value(std::string_view option_name,
                                                          char *const argv[],
                                                          int argc,
                                                          std::size_t &index) const;

        /**
         * @brief 选项定义列表；Registered option definitions.
         */
        std::vector<OptionDefinition> definitions_;
    };
} // namespace jacobi::svd::cli
