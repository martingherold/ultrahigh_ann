#include "benchmark_dataset.hpp"

#include "ultrahigh_ann/exact/l2/exact_l2_index.hpp"

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

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(
              std::filesystem::temp_directory_path() /
              ("ultrahigh-ann-benchmark-dataset-test-" +
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
void write_npy(const std::filesystem::path& path,
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

[[nodiscard]] bool load_rejected(
    const ultrahigh_ann::benchmark::BenchmarkSetup& setup)
{
    try {
        static_cast<void>(
            ultrahigh_ann::benchmark::load_benchmark_dataset(setup));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

}  // namespace

int main()
{
    const TemporaryDirectory directory;
    ultrahigh_ann::benchmark::BenchmarkSetup setup;
    setup.representatives_path = directory.path() / "reference_vectors.npy";
    setup.representative_labels_path =
        directory.path() / "reference_labels.npy";
    setup.queries_path = directory.path() / "queries.npy";
    setup.query_labels_path = directory.path() / "query_labels.npy";

    const std::array<std::size_t, 2> representative_shape{2, 2};
    const std::array<float, 4> representatives{0.0F, 0.0F, 5.0F, 5.0F};
    write_npy<float>(setup.representatives_path, "<f4", representative_shape,
                     representatives);
    const std::array<std::size_t, 2> query_shape{1, 2};
    const std::array<float, 2> queries{4.5F, 5.0F};
    write_npy<float>(setup.queries_path, "<f4", query_shape, queries);
    const std::array<std::size_t, 1> representative_label_shape{2};
    const std::array<std::uint16_t, 2> representative_labels{7, 9};
    write_npy<std::uint16_t>(setup.representative_labels_path, "<u2",
                             representative_label_shape,
                             representative_labels);
    const std::array<std::size_t, 1> query_label_shape{1};
    const std::array<std::uint16_t, 1> query_labels{9};
    write_npy<std::uint16_t>(setup.query_labels_path, "<u2",
                             query_label_shape, query_labels);

    auto dataset =
        ultrahigh_ann::benchmark::load_benchmark_dataset(setup);
    bool passed = true;
    passed &= expect(dataset.labels_available,
                     "a fully labelled dataset must expose quality labels");
    const ultrahigh_ann::ExactL2Index index(dataset.values.representatives);
    const std::size_t prediction = index.query(dataset.values.queries.row(0));
    passed &= expect(
        prediction == 1 &&
            dataset.values.representative_labels[prediction] ==
                dataset.values.query_labels[0],
        "loaded labels must map an end-to-end exact query to its class");

    std::filesystem::remove(setup.representative_labels_path);
    passed &= expect(load_rejected(setup),
                     "query-only labels must be rejected");
    write_npy<std::uint16_t>(setup.representative_labels_path, "<u2",
                             representative_label_shape,
                             representative_labels);
    std::filesystem::remove(setup.query_labels_path);
    passed &= expect(load_rejected(setup),
                     "representative-only labels must be rejected");

    std::filesystem::remove(setup.representative_labels_path);
    dataset = ultrahigh_ann::benchmark::load_benchmark_dataset(setup);
    passed &= expect(
        !dataset.labels_available &&
            dataset.values.representative_labels.size() == 2 &&
            dataset.values.query_labels.size() == 1,
        "a completely unlabelled dataset must remain benchmarkable");

    return passed ? 0 : 1;
}
