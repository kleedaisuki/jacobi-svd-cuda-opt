#include "jacobi/svd/io/txt_file.hpp"

#include "jacobi/svd/io/matrix_stream.hpp"
#include "src/infrastructure/io_detail.hpp"

#include <memory>

namespace jacobi::svd::io
{
    using namespace detail;
    struct TxtReader::Impl final
    {
        /**
         * @brief 输入文件流；Input file stream.
         */
        std::ifstream input;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : input(path, std::ios::in | std::ios::binary)
        {
            if (!input)
            {
                throw std::runtime_error("Failed to open text matrix file for reading.");
            }
            input.imbue(std::locale::classic());
        }
    };

    /**
     * @brief *.txt 写入器实现体；Implementation of *.txt writer.
     */
    struct TxtWriter::Impl final
    {
        /**
         * @brief 输出文件流；Output file stream.
         */
        std::ofstream output;

        /**
         * @brief 是否已有写入内容；Whether any matrix has been written.
         */
        bool has_written_matrix = false;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : output(path, std::ios::out | std::ios::binary | std::ios::trunc)
        {
            if (!output)
            {
                throw std::runtime_error("Failed to open text matrix file for writing.");
            }
            output.imbue(std::locale::classic());
            output << std::setprecision(std::numeric_limits<double>::max_digits10);
        }
    };

    TxtReader::TxtReader(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    TxtReader::~TxtReader() = default;

    TxtReader::TxtReader(TxtReader &&other) noexcept = default;

    TxtReader &TxtReader::operator=(TxtReader &&other) noexcept = default;

    TxtWriter::TxtWriter(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    TxtWriter::~TxtWriter() = default;

    TxtWriter::TxtWriter(TxtWriter &&other) noexcept = default;

    TxtWriter &TxtWriter::operator=(TxtWriter &&other) noexcept = default;
    TxtFilePolicy::Reader TxtFilePolicy::open_reader(const std::filesystem::path &path)
    {
        return TxtReader(path);
    }

    TxtFilePolicy::Writer TxtFilePolicy::open_writer(const std::filesystem::path &path)
    {
        return TxtWriter(path);
    }

    bool TxtFilePolicy::read_next(Reader &reader, Matrix &matrix)
    {
        if (reader.impl_ == nullptr)
        {
            throw std::runtime_error("TxtReader is not initialized.");
        }

        auto &input = reader.impl_->input;
        std::vector<double> values;
        std::size_t rows = 0;
        std::size_t columns = 0;
        bool in_matrix = false;

        std::string line;
        while (std::getline(input, line))
        {
            if (is_blank_line(line))
            {
                if (!in_matrix)
                {
                    continue;
                }

                matrix.rows = rows;
                matrix.columns = columns;
                matrix.values = std::move(values);
                return true;
            }

            std::vector<double> row_values = parse_txt_row(line);
            if (row_values.empty())
            {
                throw std::invalid_argument("Text matrix row cannot be empty when line is non-blank.");
            }

            if (!in_matrix)
            {
                in_matrix = true;
                columns = row_values.size();
            }
            else if (row_values.size() != columns)
            {
                throw std::invalid_argument("Inconsistent column count in text matrix block.");
            }

            values.insert(values.end(), row_values.begin(), row_values.end());
            rows = checked_add(rows, 1);
        }

        if (!input.eof())
        {
            throw std::runtime_error("Failed while reading text matrix file.");
        }

        if (!in_matrix)
        {
            return false;
        }

        matrix.rows = rows;
        matrix.columns = columns;
        matrix.values = std::move(values);
        return true;
    }

    void TxtFilePolicy::write_next(Writer &writer, const Matrix &matrix)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("TxtWriter is not initialized.");
        }

        validate_matrix_layout(matrix);
        auto &state = *writer.impl_;

        if (state.has_written_matrix)
        {
            state.output << '\n';
        }

        for (std::size_t row = 0; row < matrix.rows; ++row)
        {
            for (std::size_t column = 0; column < matrix.columns; ++column)
            {
                if (column > 0)
                {
                    state.output << ' ';
                }
                const std::size_t index = checked_add(checked_multiply(row, matrix.columns), column);
                state.output << matrix.values[index];
            }
            state.output << '\n';
        }

        if (!state.output)
        {
            throw std::runtime_error("Failed while writing text matrix file.");
        }

        state.has_written_matrix = true;
    }

    void TxtFilePolicy::flush(Writer &writer)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("TxtWriter is not initialized.");
        }

        writer.impl_->output.flush();
        if (!writer.impl_->output)
        {
            throw std::runtime_error("Failed to flush text matrix file.");
        }
    }

    std::vector<Matrix> TxtFilePolicy::read(const std::filesystem::path &path)
    {
        TxtInputStream stream(path);
        return stream.read_all();
    }

    void TxtFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        TxtOutputStream stream(path);
        stream.write_all(matrices);
    }
} // namespace jacobi::svd::io
