#include "ultrahigh_ann/datasets/representative_query_dataset.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(
              std::filesystem::temp_directory_path() /
              ("ultrahigh-ann-dataset-test-" +
               std::to_string(
                   std::chrono::steady_clock::now()
                       .time_since_epoch()
                       .count())))
    {
        std::filesystem::create_directory(path_);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

bool expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }
    return condition;
}

template<class T>
void write_npy(
    const std::filesystem::path& path,
    std::string_view descriptor,
    std::span<const std::size_t> shape,
    std::span<const T> values)
{
    std::size_t expected_values = 1;
    std::string shape_text{"("};
    for (std::size_t index = 0; index < shape.size(); ++index) {
        expected_values *= shape[index];
        if (index != 0) {
            shape_text += ", ";
        }
        shape_text += std::to_string(shape[index]);
    }
    if (shape.size() == 1) {
        shape_text += ',';
    }
    shape_text += ')';
    if (values.size() != expected_values) {
        throw std::logic_error("test NPY shape does not match its values");
    }

    std::string header =
        "{'descr': '" + std::string(descriptor) +
        "', 'fortran_order': False, 'shape': " + shape_text + ", }";
    constexpr std::size_t preamble_size = 10;
    constexpr std::size_t alignment = 64;
    const std::size_t remainder =
        (preamble_size + header.size() + 1) % alignment;
    const std::size_t padding = alignment - remainder;
    header.append(padding - 1, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::logic_error("test NPY header is too large");
    }

    std::ofstream output(path, std::ios::binary);
    const std::array<char, 8> preamble{
        static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
    output.write(preamble.data(), preamble.size());
    const auto header_size = static_cast<std::uint16_t>(header.size());
    const std::array<char, 2> length{
        static_cast<char>(header_size & 0xffU),
        static_cast<char>((header_size >> 8U) & 0xffU)};
    output.write(length.data(), length.size());
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size_bytes()));
    if (!output) {
        throw std::runtime_error("failed to write test NPY file");
    }
}

}  // namespace

int main()
{
    const TemporaryDirectory directory;
    const std::array<std::size_t, 2> representative_shape{2, 3};
    const std::array<float, 6> representatives{
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F};
    write_npy<float>(
        directory.path() / "representatives.npy",
        "<f4",
        representative_shape,
        representatives);
    const ultrahigh_ann::DenseMatrix representative_matrix =
        ultrahigh_ann::load_representative_matrix(directory.path());

    bool passed = true;
    passed &= expect(
        representative_matrix.rows() == 2 &&
            representative_matrix.cols() == 3 &&
            representative_matrix.row(1)[2] == 6.0F,
        "the representative-only loader must preserve shape and values");

    const std::array<std::size_t, 2> query_shape{3, 3};
    const std::array<float, 9> queries{
        1.0F, 1.0F, 1.0F,
        5.0F, 5.0F, 5.0F,
        2.0F, 2.0F, 2.0F};
    write_npy<float>(
        directory.path() / "queries.npy",
        "<f4",
        query_shape,
        queries);

    const std::array<std::size_t, 1> label_shape{3};
    const std::array<std::uint16_t, 3> labels{0, 1, 0};
    write_npy<std::uint16_t>(
        directory.path() / "query_labels.npy",
        "<u2",
        label_shape,
        labels);

    ultrahigh_ann::RepresentativeQueryDataset dataset =
        ultrahigh_ann::load_representative_query_dataset(
            directory.path());

    passed &= expect(
        dataset.representatives.rows() == 2 &&
            dataset.representatives.cols() == 3,
        "representative shape must be loaded");
    passed &= expect(
        dataset.queries.rows() == 3 && dataset.queries.cols() == 3,
        "query shape must be loaded");
    passed &= expect(
        dataset.representatives.row(1)[2] == 6.0F,
        "representative values must retain row-major order");
    passed &= expect(
        dataset.queries.row(2)[0] == 2.0F,
        "query values must retain row-major order");
    passed &= expect(
        dataset.query_labels == std::vector<std::size_t>({0, 1, 0}),
        "query labels must be loaded");
    passed &= expect(
        dataset.representative_labels ==
            std::vector<std::size_t>({0, 1}),
        "missing representative labels must default to row indices");

    const std::array<std::uint16_t, 2> representative_labels{4, 7};
    const std::array<std::size_t, 1> representative_label_shape{2};
    write_npy<std::uint16_t>(
        directory.path() / "representative_labels.npy",
        "<u2",
        representative_label_shape,
        representative_labels);
    const std::array<std::uint16_t, 3> mapped_query_labels{4, 7, 4};
    write_npy<std::uint16_t>(
        directory.path() / "query_labels.npy",
        "<u2",
        label_shape,
        mapped_query_labels);
    dataset = ultrahigh_ann::load_representative_query_dataset(
        directory.path());
    passed &= expect(
        dataset.representative_labels ==
            std::vector<std::size_t>({4, 7}) &&
            dataset.query_labels ==
                std::vector<std::size_t>({4, 7, 4}),
        "explicit representative labels must map rows to query classes");

    const std::array<std::size_t, 1> short_label_shape{2};
    const std::array<std::uint16_t, 2> short_labels{4, 7};
    write_npy<std::uint16_t>(
        directory.path() / "query_labels.npy",
        "<u2",
        short_label_shape,
        short_labels);
    bool mismatch_rejected = false;
    try {
        static_cast<void>(
            ultrahigh_ann::load_representative_query_dataset(
                directory.path()));
    } catch (const std::runtime_error&) {
        mismatch_rejected = true;
    }
    passed &= expect(
        mismatch_rejected,
        "query/label row mismatches must be rejected");

    write_npy<std::uint16_t>(
        directory.path() / "query_labels.npy",
        "<u2",
        label_shape,
        mapped_query_labels);

    const std::array<std::uint16_t, 1> short_representative_labels{4};
    const std::array<std::size_t, 1> short_representative_label_shape{1};
    write_npy<std::uint16_t>(
        directory.path() / "representative_labels.npy",
        "<u2",
        short_representative_label_shape,
        short_representative_labels);
    bool representative_mismatch_rejected = false;
    try {
        static_cast<void>(
            ultrahigh_ann::load_representative_query_dataset(
                directory.path()));
    } catch (const std::runtime_error&) {
        representative_mismatch_rejected = true;
    }
    passed &= expect(
        representative_mismatch_rejected,
        "representative/label row mismatches must be rejected");

    return passed ? 0 : 1;
}
