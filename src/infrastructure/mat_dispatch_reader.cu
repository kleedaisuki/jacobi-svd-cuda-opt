#include "jacobi/svd/io/mat_dispatch.hpp"

#include "src/infrastructure/io_detail.hpp"

namespace jacobi::svd::io
{
    using namespace detail;
    MatDispatchReader::MatDispatchReader(const std::filesystem::path &path)
        : input_(path, std::ios::in | std::ios::binary)
    {
        if (!input_)
        {
            throw std::runtime_error("Failed to open *.mat file for dispatch reading.");
        }
    }

    bool MatDispatchReader::read_next(MatDispatchTask &task, std::size_t workspace_bytes)
    {
        MatMetaData encoded_meta{};
        input_.read(reinterpret_cast<char *>(&encoded_meta), checked_to_streamsize(kMatHeaderBytes));

        if (input_.gcount() == 0 && input_.eof())
        {
            return false;
        }
        if (input_.gcount() != checked_to_streamsize(kMatHeaderBytes))
        {
            throw std::runtime_error("Truncated *.mat header while dispatch reading.");
        }

        const std::uint64_t rows_u64 = from_network_u64(encoded_meta.rows);
        const std::uint64_t columns_u64 = from_network_u64(encoded_meta.columns);
        if (rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            columns_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed platform size_t range.");
        }

        const std::size_t rows = static_cast<std::size_t>(rows_u64);
        const std::size_t columns = static_cast<std::size_t>(columns_u64);
        const std::size_t element_count = checked_multiply(rows, columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);

        task.buffer.reserve(payload_bytes, workspace_bytes);
        if (payload_bytes > 0)
        {
            std::span<std::byte> input_bytes = task.buffer.mutable_input_bytes();
            input_.read(reinterpret_cast<char *>(input_bytes.data()), checked_to_streamsize(payload_bytes));
            if (input_.gcount() != checked_to_streamsize(payload_bytes))
            {
                throw std::runtime_error("Truncated *.mat payload while dispatch reading.");
            }
        }

        task.sequence_index = next_sequence_index_;
        task.rows = rows;
        task.columns = columns;
        next_sequence_index_ = checked_add(next_sequence_index_, 1);
        return true;
    }

    Matrix decode_dispatch_task_matrix(const MatDispatchTask &task)
    {
        Matrix matrix;
        matrix.rows = task.rows;
        matrix.columns = task.columns;

        const std::size_t element_count = checked_multiply(matrix.rows, matrix.columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);
        const std::span<const std::byte> payload = task.buffer.input_bytes();
        if (payload.size() != payload_bytes)
        {
            throw std::invalid_argument("Dispatch payload size does not match rows * columns.");
        }

        matrix.values.resize(element_count);
        for (std::size_t index = 0; index < element_count; ++index)
        {
            std::uint64_t encoded = 0;
            std::memcpy(&encoded, payload.data() + checked_multiply(index, kMatElementBytes), kMatElementBytes);
            matrix.values[index] = decode_network_double(encoded);
        }
        return matrix;
    }
} // namespace jacobi::svd::io
