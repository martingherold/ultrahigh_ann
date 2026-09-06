#include "ultrahigh_ann/datasets/representative_query_dataset.hpp"

#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ultrahigh_ann {

namespace {

struct NpyHeader {
    std::string descriptor;
    bool fortran_order{};
    std::vector<std::size_t> shape;
};

[[nodiscard]] std::runtime_error npy_error(
    const std::filesystem::path& path,
    std::string_view message)
{
    return std::runtime_error(
        "cannot load " + path.string() + ": " + std::string(message));
}

void read_exact(
    std::ifstream& input,
    char* destination,
    std::size_t byte_count,
    const std::filesystem::path& path,
    std::string_view description)
{
    if (byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::streamsize>::max())) {
        throw npy_error(path, "file section is too large to read");
    }

    input.read(
        destination,
        static_cast<std::streamsize>(byte_count));
    if (input.gcount() != static_cast<std::streamsize>(byte_count)) {
        throw npy_error(
            path,
            "truncated while reading " + std::string(description));
    }
}

template<class Unsigned>
[[nodiscard]] Unsigned read_little_endian(
    std::ifstream& input,
    const std::filesystem::path& path,
    std::string_view description)
{
    static_assert(std::is_unsigned_v<Unsigned>);
    static_assert(sizeof(Unsigned) <= sizeof(std::uint64_t));
    std::array<unsigned char, sizeof(Unsigned)> bytes{};
    read_exact(
        input,
        reinterpret_cast<char*>(bytes.data()),
        bytes.size(),
        path,
        description);

    std::uint64_t value{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return static_cast<Unsigned>(value);
}

[[nodiscard]] std::size_t find_field_colon(
    std::string_view header,
    std::string_view field,
    const std::filesystem::path& path)
{
    const std::size_t field_position = header.find(field);
    if (field_position == std::string_view::npos) {
        throw npy_error(path, "NPY header has no " + std::string(field));
    }
    const std::size_t colon = header.find(':', field_position + field.size());
    if (colon == std::string_view::npos) {
        throw npy_error(path, "malformed NPY field " + std::string(field));
    }
    return colon;
}

[[nodiscard]] std::string parse_descriptor(
    std::string_view header,
    const std::filesystem::path& path)
{
    const std::size_t colon = find_field_colon(header, "descr", path);
    const std::size_t quote = header.find_first_of("'\"", colon + 1);
    if (quote == std::string_view::npos) {
        throw npy_error(path, "malformed NPY dtype descriptor");
    }
    const char delimiter = header[quote];
    const std::size_t end = header.find(delimiter, quote + 1);
    if (end == std::string_view::npos) {
        throw npy_error(path, "unterminated NPY dtype descriptor");
    }
    return std::string(header.substr(quote + 1, end - quote - 1));
}

[[nodiscard]] bool parse_fortran_order(
    std::string_view header,
    const std::filesystem::path& path)
{
    const std::size_t colon =
        find_field_colon(header, "fortran_order", path);
    const std::size_t value = header.find_first_not_of(" \t", colon + 1);
    if (value == std::string_view::npos) {
        throw npy_error(path, "malformed NPY fortran_order field");
    }
    if (header.substr(value, 4) == "True") {
        return true;
    }
    if (header.substr(value, 5) == "False") {
        return false;
    }
    throw npy_error(path, "invalid NPY fortran_order value");
}

[[nodiscard]] std::string_view trim(std::string_view value)
{
    const std::size_t first = value.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

[[nodiscard]] std::vector<std::size_t> parse_shape(
    std::string_view header,
    const std::filesystem::path& path)
{
    const std::size_t colon = find_field_colon(header, "shape", path);
    const std::size_t open = header.find('(', colon + 1);
    const std::size_t close = header.find(')', open + 1);
    if (open == std::string_view::npos || close == std::string_view::npos) {
        throw npy_error(path, "malformed NPY shape field");
    }

    std::vector<std::size_t> shape;
    const std::string_view contents = header.substr(
        open + 1,
        close - open - 1);
    std::size_t start = 0;
    while (start <= contents.size()) {
        const std::size_t comma = contents.find(',', start);
        const std::size_t end =
            comma == std::string_view::npos ? contents.size() : comma;
        const std::string_view token = trim(contents.substr(start, end - start));
        if (!token.empty()) {
            std::size_t dimension{};
            const auto [position, error] = std::from_chars(
                token.data(),
                token.data() + token.size(),
                dimension);
            if (error != std::errc{} || position != token.data() + token.size()) {
                throw npy_error(path, "invalid dimension in NPY shape");
            }
            shape.push_back(dimension);
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
    if (shape.empty()) {
        throw npy_error(path, "scalar NPY arrays are not supported");
    }
    return shape;
}

[[nodiscard]] NpyHeader read_npy_header(
    std::ifstream& input,
    const std::filesystem::path& path)
{
    constexpr std::array<char, 6> expected_magic{
        static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y'};
    std::array<char, expected_magic.size()> magic{};
    read_exact(input, magic.data(), magic.size(), path, "NPY magic");
    if (magic != expected_magic) {
        throw npy_error(path, "invalid NPY magic");
    }

    std::array<unsigned char, 2> version{};
    read_exact(
        input,
        reinterpret_cast<char*>(version.data()),
        version.size(),
        path,
        "NPY version");

    std::uint32_t header_length{};
    if (version[0] == 1) {
        header_length = read_little_endian<std::uint16_t>(
            input, path, "NPY header length");
    } else if (version[0] == 2 || version[0] == 3) {
        header_length = read_little_endian<std::uint32_t>(
            input, path, "NPY header length");
    } else {
        throw npy_error(path, "unsupported NPY format version");
    }

    if (header_length == 0 || header_length > 1024U * 1024U) {
        throw npy_error(path, "unreasonable NPY header length");
    }
    std::string header(header_length, '\0');
    read_exact(input, header.data(), header.size(), path, "NPY header");

    return NpyHeader{
        parse_descriptor(header, path),
        parse_fortran_order(header, path),
        parse_shape(header, path)};
}

[[nodiscard]] std::size_t element_count(
    const std::vector<std::size_t>& shape,
    const std::filesystem::path& path)
{
    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension != 0 &&
            count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw npy_error(path, "NPY shape overflows addressable memory");
        }
        count *= dimension;
    }
    return count;
}

template<class T>
[[nodiscard]] std::pair<std::vector<T>, std::vector<std::size_t>>
load_npy_array(
    const std::filesystem::path& path,
    std::string_view expected_descriptor)
{
    if constexpr (std::endian::native != std::endian::little) {
        throw npy_error(path, "only little-endian hosts are supported");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw npy_error(path, "file does not exist or cannot be opened");
    }
    NpyHeader header = read_npy_header(input, path);
    if (header.descriptor != expected_descriptor) {
        throw npy_error(
            path,
            "expected dtype " + std::string(expected_descriptor) +
                ", found " + header.descriptor);
    }
    if (header.fortran_order) {
        throw npy_error(path, "Fortran-order arrays are not supported");
    }

    const std::size_t count = element_count(header.shape, path);
    if (count > std::vector<T>{}.max_size()) {
        throw npy_error(path, "NPY array is too large");
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw npy_error(path, "NPY byte count overflows addressable memory");
    }

    std::vector<T> values(count);
    read_exact(
        input,
        reinterpret_cast<char*>(values.data()),
        count * sizeof(T),
        path,
        "NPY array data");
    if (input.peek() != std::char_traits<char>::eof()) {
        throw npy_error(path, "unexpected trailing data");
    }
    return {std::move(values), std::move(header.shape)};
}

[[nodiscard]] DenseMatrix load_float_matrix_impl(
    const std::filesystem::path& path)
{
    auto [values, shape] = load_npy_array<float>(path, "<f4");
    if (shape.size() != 2) {
        throw npy_error(path, "expected a two-dimensional matrix");
    }
    for (const float value : values) {
        if (!std::isfinite(value)) {
            throw npy_error(path, "matrix contains a non-finite value");
        }
    }
    return DenseMatrix(std::move(values), shape[0], shape[1]);
}

[[nodiscard]] std::vector<std::size_t> load_labels(
    const std::filesystem::path& path)
{
    auto [raw_labels, shape] =
        load_npy_array<std::uint16_t>(path, "<u2");
    if (shape.size() != 1) {
        throw npy_error(path, "expected a one-dimensional label array");
    }
    return std::vector<std::size_t>(
        raw_labels.begin(),
        raw_labels.end());
}

}  // namespace

DenseMatrix load_float_matrix_npy(const std::filesystem::path& path)
{
    return load_float_matrix_impl(path);
}

std::vector<std::size_t> load_label_vector_npy(
    const std::filesystem::path& path)
{
    return load_labels(path);
}

DenseMatrix load_representative_matrix(
    const std::filesystem::path& directory)
{
    DenseMatrix representatives =
        load_float_matrix_npy(directory / "reference_vectors.npy");
    if (representatives.rows() == 0) {
        throw std::runtime_error("dataset has no reference vectors");
    }
    if (representatives.cols() == 0) {
        throw std::runtime_error("dataset has no features");
    }
    return representatives;
}

RepresentativeQueryDataset load_representative_query_dataset(
    const std::filesystem::path& directory)
{
    DenseMatrix representatives =
        load_representative_matrix(directory);
    DenseMatrix queries =
        load_float_matrix_npy(directory / "queries.npy");
    std::vector<std::size_t> query_labels =
        load_labels(directory / "query_labels.npy");
    const std::filesystem::path representative_labels_path =
        directory / "reference_labels.npy";
    std::vector<std::size_t> representative_labels;
    if (std::filesystem::is_regular_file(representative_labels_path)) {
        representative_labels = load_labels(representative_labels_path);
    } else {
        representative_labels.resize(representatives.rows());
        std::iota(
            representative_labels.begin(),
            representative_labels.end(),
            std::size_t{0});
    }

    if (queries.rows() == 0) {
        throw std::runtime_error("dataset has no queries");
    }
    if (representatives.cols() != queries.cols()) {
        throw std::runtime_error(
            "reference-vector and query dimensions do not match");
    }
    if (query_labels.size() != queries.rows()) {
        throw std::runtime_error(
            "query label count does not match query row count");
    }
    if (representative_labels.size() != representatives.rows()) {
        throw std::runtime_error(
            "reference label count does not match reference-vector row count");
    }
    const std::unordered_set<std::size_t> available_labels{
        representative_labels.begin(),
        representative_labels.end()};
    for (const std::size_t label : query_labels) {
        if (!available_labels.contains(label)) {
            throw std::runtime_error(
                "query label has no representative");
        }
    }

    return RepresentativeQueryDataset{
        std::move(representatives),
        std::move(queries),
        std::move(representative_labels),
        std::move(query_labels)};
}

}  // namespace ultrahigh_ann
