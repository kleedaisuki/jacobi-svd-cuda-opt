#include "jacobi/svd/io/mat_file.hpp"

#include "jacobi/svd/io/matrix_stream.hpp"
#include "src/infrastructure/append_mapped_output_file.hpp"
#include "src/infrastructure/io_detail.hpp"
#include "src/infrastructure/memory_mapped_input_file.hpp"

#include <memory>

namespace jacobi::svd::io
{
    using namespace detail;
    struct MatReader::Impl final
    {
        /**
         * @brief 映射文件对象；Mapped file object.
         */
        MemoryMappedInputFile mapped_file;

        /**
         * @brief 映射字节视图；Mapped byte span.
         */
        std::span<const std::byte> bytes;

        /**
         * @brief 当前游标；Current cursor.
         */
        std::size_t cursor = 0;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : mapped_file(path), bytes(mapped_file.bytes())
        {
        }
    };

    /**
     * @brief *.mat 写入器实现体；Implementation of *.mat writer.
     */
    struct MatWriter::Impl final
    {
        /**
         * @brief 追加式映射输出文件；Appendable mapped output file.
         */
        AppendMappedOutputFile mapped_output;

        /**
         * @brief 构造实现体；Construct implementation.
         * @param path 文件路径；File path.
         */
        explicit Impl(const std::filesystem::path &path)
            : mapped_output(path)
        {
        }
    };

    MatReader::MatReader(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    MatReader::~MatReader() = default;

    MatReader::MatReader(MatReader &&other) noexcept = default;

    MatReader &MatReader::operator=(MatReader &&other) noexcept = default;

    MatWriter::MatWriter(const std::filesystem::path &path)
        : impl_(std::make_unique<Impl>(path))
    {
    }

    MatWriter::~MatWriter() = default;

    MatWriter::MatWriter(MatWriter &&other) noexcept = default;

    MatWriter &MatWriter::operator=(MatWriter &&other) noexcept = default;
    MatFilePolicy::Reader MatFilePolicy::open_reader(const std::filesystem::path &path)
    {
        return MatReader(path);
    }

    MatFilePolicy::Writer MatFilePolicy::open_writer(const std::filesystem::path &path)
    {
        return MatWriter(path);
    }

    bool MatFilePolicy::read_next(Reader &reader, Matrix &matrix)
    {
        if (reader.impl_ == nullptr)
        {
            throw std::runtime_error("MatReader is not initialized.");
        }

        auto &state = *reader.impl_;
        if (state.cursor == state.bytes.size())
        {
            return false;
        }

        const std::size_t remaining = state.bytes.size() - state.cursor;
        if (remaining < kMatHeaderBytes)
        {
            throw std::runtime_error("Truncated *.mat header.");
        }

        MatMetaData encoded_meta{};
        std::memcpy(&encoded_meta, state.bytes.data() + state.cursor, kMatHeaderBytes);
        state.cursor = checked_add(state.cursor, kMatHeaderBytes);

        const std::uint64_t rows_u64 = from_network_u64(encoded_meta.rows);
        const std::uint64_t columns_u64 = from_network_u64(encoded_meta.columns);

        if (rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            columns_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed platform size_t range.");
        }

        matrix.rows = static_cast<std::size_t>(rows_u64);
        matrix.columns = static_cast<std::size_t>(columns_u64);

        const std::size_t element_count = checked_multiply(matrix.rows, matrix.columns);
        const std::size_t payload_bytes = checked_multiply(element_count, kMatElementBytes);

        if ((state.bytes.size() - state.cursor) < payload_bytes)
        {
            throw std::runtime_error("Truncated *.mat payload.");
        }

        matrix.values.resize(element_count);

        std::size_t processed = 0;
        while (processed < element_count)
        {
            const std::size_t chunk_count = std::min(kDecodeChunkElements, element_count - processed);
            const std::byte *chunk_base = state.bytes.data() + state.cursor + checked_multiply(processed, kMatElementBytes);

            for (std::size_t index = 0; index < chunk_count; ++index)
            {
                std::uint64_t encoded = 0;
                std::memcpy(&encoded, chunk_base + checked_multiply(index, kMatElementBytes), kMatElementBytes);
                matrix.values[processed + index] = decode_network_double(encoded);
            }

            processed = checked_add(processed, chunk_count);
        }

        state.cursor = checked_add(state.cursor, payload_bytes);
        return true;
    }

    void MatFilePolicy::write_next(Writer &writer, const Matrix &matrix)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("MatWriter is not initialized.");
        }

        validate_matrix_layout(matrix);
        if (matrix.rows > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()) ||
            matrix.columns > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max()))
        {
            throw std::overflow_error("Matrix dimensions exceed *.mat uint64 metadata capacity.");
        }

        MatMetaData encoded_meta{
            .rows = to_network_u64(static_cast<std::uint64_t>(matrix.rows)),
            .columns = to_network_u64(static_cast<std::uint64_t>(matrix.columns)),
        };

        const std::span<const MatMetaData> header_span(&encoded_meta, 1);
        const std::span<const std::byte> header_bytes = std::as_bytes(header_span);
        writer.impl_->mapped_output.append(header_bytes);

        std::vector<std::uint64_t> encoded_chunk;
        encoded_chunk.reserve(kEncodeChunkElements);

        std::size_t processed = 0;
        while (processed < matrix.values.size())
        {
            const std::size_t chunk_count = std::min(kEncodeChunkElements, matrix.values.size() - processed);
            encoded_chunk.resize(chunk_count);
            for (std::size_t index = 0; index < chunk_count; ++index)
            {
                encoded_chunk[index] = encode_network_double(matrix.values[processed + index]);
            }

            const std::span<const std::uint64_t> chunk_span(encoded_chunk.data(), encoded_chunk.size());
            const std::span<const std::byte> chunk_bytes = std::as_bytes(chunk_span);
            writer.impl_->mapped_output.append(chunk_bytes);
            processed = checked_add(processed, chunk_count);
        }
    }

    void MatFilePolicy::flush(Writer &writer)
    {
        if (writer.impl_ == nullptr)
        {
            throw std::runtime_error("MatWriter is not initialized.");
        }
        writer.impl_->mapped_output.flush();
    }

    std::vector<Matrix> MatFilePolicy::read(const std::filesystem::path &path)
    {
        MatInputStream stream(path);
        return stream.read_all();
    }

    void MatFilePolicy::write(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        MatOutputStream stream(path);
        stream.write_all(matrices);
    }
} // namespace jacobi::svd::io
