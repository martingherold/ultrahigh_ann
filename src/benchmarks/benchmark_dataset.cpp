#include "benchmark_dataset.hpp"

#include <chrono>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ultrahigh_ann::benchmark {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

}  // namespace

LoadedBenchmarkDataset load_benchmark_dataset(const BenchmarkSetup& setup)
{
    const bool representative_labels_exist =
        std::filesystem::is_regular_file(setup.representative_labels_path);
    const bool query_labels_exist =
        std::filesystem::is_regular_file(setup.query_labels_path);
    if (representative_labels_exist != query_labels_exist) {
        throw std::runtime_error(
            "reference-vector and query labels must either both exist or both "
            "be absent");
    }

    const auto hash_start = Clock::now();
    const io::Sha256Digest representatives_sha256 =
        io::sha256_file(setup.representatives_path);
    const io::Sha256Digest queries_sha256 =
        io::sha256_file(setup.queries_path);
    std::optional<io::Sha256Digest> representative_labels_sha256;
    std::optional<io::Sha256Digest> query_labels_sha256;
    if (representative_labels_exist) {
        representative_labels_sha256 =
            io::sha256_file(setup.representative_labels_path);
        query_labels_sha256 = io::sha256_file(setup.query_labels_path);
    }
    const double hash_ms = elapsed_ms(hash_start);

    const auto load_start = Clock::now();
    RepresentativeQueryDataset dataset;
    dataset.representatives =
        load_float_matrix_npy(setup.representatives_path);
    dataset.queries = load_float_matrix_npy(setup.queries_path);
    if (dataset.representatives.rows() == 0 ||
        dataset.representatives.cols() == 0 || dataset.queries.rows() == 0 ||
        dataset.representatives.cols() != dataset.queries.cols()) {
        throw std::runtime_error(
            "reference vectors and queries must be nonempty matrices with the "
            "same dimension");
    }
    if (representative_labels_exist) {
        dataset.representative_labels =
            load_label_vector_npy(setup.representative_labels_path);
        dataset.query_labels =
            load_label_vector_npy(setup.query_labels_path);
        if (dataset.representative_labels.size() !=
                dataset.representatives.rows() ||
            dataset.query_labels.size() != dataset.queries.rows()) {
            throw std::runtime_error(
                "label counts do not match the matrix row counts");
        }
    } else {
        dataset.representative_labels.resize(dataset.representatives.rows());
        std::iota(dataset.representative_labels.begin(),
                  dataset.representative_labels.end(), std::size_t{});
        dataset.query_labels.assign(dataset.queries.rows(), 0);
    }

    return LoadedBenchmarkDataset{
        .values = std::move(dataset),
        .labels_available = representative_labels_exist,
        .load_ms = elapsed_ms(load_start),
        .hash_ms = hash_ms,
        .representatives_sha256 = representatives_sha256,
        .queries_sha256 = queries_sha256,
        .representative_labels_sha256 = representative_labels_sha256,
        .query_labels_sha256 = query_labels_sha256,
    };
}

}  // namespace ultrahigh_ann::benchmark
