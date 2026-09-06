#include "ultrahigh_ann/io/importance_probability_io.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ultrahigh_ann::io {
namespace {

constexpr std::array<char, 8> magic{'U', 'H', 'A', 'N', 'N', 'P', 'R', '2'};
constexpr std::size_t reserved_byte_count = 7;

[[nodiscard]] std::runtime_error file_error(const std::filesystem::path& path,
                                            std::string_view message)
{
    return std::runtime_error("invalid importance-probability file " +
                              path.string() + ": " + std::string(message));
}

void write_exact(std::ofstream& output, const char* data,
                 std::size_t byte_count, const std::filesystem::path& path)
{
    if (byte_count >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw file_error(path, "payload is too large to write");
    }
    output.write(data, static_cast<std::streamsize>(byte_count));
    if (!output) {
        throw file_error(path, "failed while writing");
    }
}

void read_exact(std::ifstream& input, char* data, std::size_t byte_count,
                const std::filesystem::path& path, std::string_view description)
{
    if (byte_count >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw file_error(path, "payload is too large to read");
    }
    input.read(data, static_cast<std::streamsize>(byte_count));
    if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
        throw file_error(path,
                         "truncated while reading " + std::string(description));
    }
}

template<class Unsigned>
void write_little_endian(std::ofstream& output, Unsigned value,
                         const std::filesystem::path& path)
{
    static_assert(std::is_unsigned_v<Unsigned>);
    std::array<char, sizeof(Unsigned)> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<char>(value & Unsigned{0xff});
        value >>= 8U;
    }
    write_exact(output, bytes.data(), bytes.size(), path);
}

template<class Unsigned>
[[nodiscard]] Unsigned read_little_endian(std::ifstream& input,
                                          const std::filesystem::path& path,
                                          std::string_view description)
{
    static_assert(std::is_unsigned_v<Unsigned>);
    std::array<unsigned char, sizeof(Unsigned)> bytes{};
    read_exact(input, reinterpret_cast<char*>(bytes.data()), bytes.size(), path,
               description);
    Unsigned value{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<Unsigned>(bytes[index]) << (index * 8U);
    }
    return value;
}

[[nodiscard]] std::uint8_t encoded_kind(ImportanceProbabilityKind kind)
{
    switch (kind) {
    case ImportanceProbabilityKind::l1:
        return 1;
    case ImportanceProbabilityKind::l2:
        return 2;
    }
    throw std::invalid_argument("unknown importance-probability kind");
}

[[nodiscard]] ImportanceProbabilityKind
decode_kind(std::uint8_t value, const std::filesystem::path& path)
{
    if (value == 1) {
        return ImportanceProbabilityKind::l1;
    }
    if (value == 2) {
        return ImportanceProbabilityKind::l2;
    }
    throw file_error(path, "unknown probability kind");
}

void validate_probabilities(std::span<const double> probabilities,
                            const std::filesystem::path& path)
{
    for (const double probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0) {
            throw file_error(path,
                             "probabilities must be finite and in [0, 1]");
        }
    }
}

}  // namespace

void save_importance_probabilities(const std::filesystem::path& path,
                                   ImportanceProbabilityKind kind,
                                   std::size_t representative_count,
                                   const Sha256Digest& representatives_sha256,
                                   std::span<const double> probabilities)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    validate_probabilities(probabilities, path);
    if (representative_count > static_cast<std::size_t>(
                                   std::numeric_limits<std::uint64_t>::max()) ||
        probabilities.size() > static_cast<std::size_t>(
                                   std::numeric_limits<std::uint64_t>::max())) {
        throw file_error(path, "array size cannot be represented on disk");
    }

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw file_error(path, "cannot open for writing");
    }

    write_exact(output, magic.data(), magic.size(), path);
    const char kind_byte = static_cast<char>(encoded_kind(kind));
    write_exact(output, &kind_byte, 1, path);
    constexpr std::array<char, reserved_byte_count> reserved{};
    write_exact(output, reserved.data(), reserved.size(), path);
    write_little_endian(output,
                        static_cast<std::uint64_t>(representative_count), path);
    write_little_endian(output,
                        static_cast<std::uint64_t>(probabilities.size()), path);
    write_exact(output,
                reinterpret_cast<const char*>(representatives_sha256.data()),
                representatives_sha256.size(), path);
    for (const double probability : probabilities) {
        write_little_endian(output, std::bit_cast<std::uint64_t>(probability),
                            path);
    }
}

ImportanceProbabilityFile
load_importance_probabilities(const std::filesystem::path& path)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw file_error(path, "does not exist or cannot be opened");
    }
    std::array<char, magic.size()> file_magic{};
    read_exact(input, file_magic.data(), file_magic.size(), path, "magic");
    if (file_magic != magic) {
        throw file_error(path, "wrong magic or unsupported version");
    }

    char raw_kind{};
    read_exact(input, &raw_kind, 1, path, "probability kind");
    const auto kind = decode_kind(
        static_cast<std::uint8_t>(static_cast<unsigned char>(raw_kind)), path);
    std::array<char, reserved_byte_count> reserved{};
    read_exact(input, reserved.data(), reserved.size(), path,
               "reserved header bytes");
    for (const char value : reserved) {
        if (value != 0) {
            throw file_error(path, "reserved header bytes are not zero");
        }
    }

    const std::uint64_t raw_representative_count =
        read_little_endian<std::uint64_t>(input, path, "representative count");
    const std::uint64_t raw_coordinate_count =
        read_little_endian<std::uint64_t>(input, path, "coordinate count");
    if (raw_representative_count >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()) ||
        raw_coordinate_count >
            static_cast<std::uint64_t>(std::vector<double>{}.max_size())) {
        throw file_error(path, "stored array is too large");
    }

    Sha256Digest representatives_sha256{};
    read_exact(input,
               reinterpret_cast<char*>(representatives_sha256.data()),
               representatives_sha256.size(), path,
               "representative-matrix SHA-256");

    std::vector<double> probabilities(
        static_cast<std::size_t>(raw_coordinate_count));
    for (double& probability : probabilities) {
        probability = std::bit_cast<double>(
            read_little_endian<std::uint64_t>(input, path, "probability"));
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw file_error(path, "unexpected trailing data");
    }
    validate_probabilities(probabilities, path);

    return ImportanceProbabilityFile{
        .kind = kind,
        .representative_count =
            static_cast<std::size_t>(raw_representative_count),
        .representatives_sha256 = representatives_sha256,
        .probabilities = std::move(probabilities),
    };
}

bool importance_probability_source_matches(
    const ImportanceProbabilityFile& file, ImportanceProbabilityKind kind,
    std::size_t representative_count, std::size_t coordinate_count,
    const Sha256Digest& representatives_sha256) noexcept
{
    return file.kind == kind &&
           file.representative_count == representative_count &&
           file.probabilities.size() == coordinate_count &&
           file.representatives_sha256 == representatives_sha256;
}

}  // namespace ultrahigh_ann::io
