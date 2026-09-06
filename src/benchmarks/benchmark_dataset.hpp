#pragma once

#include "benchmark_setup.hpp"

#include "ultrahigh_ann/datasets/representative_query_dataset.hpp"
#include "ultrahigh_ann/io/sha256.hpp"

#include <optional>

namespace ultrahigh_ann::benchmark {

struct LoadedBenchmarkDataset {
    RepresentativeQueryDataset values;
    bool labels_available{};
    double load_ms{};
    double hash_ms{};
    io::Sha256Digest representatives_sha256{};
    io::Sha256Digest queries_sha256{};
    std::optional<io::Sha256Digest> representative_labels_sha256;
    std::optional<io::Sha256Digest> query_labels_sha256;
};

// Benchmark quality metrics are meaningful only when both sides of the label
// mapping are explicit. Completely unlabelled datasets remain valid for
// latency and exact-choice measurements.
[[nodiscard]] LoadedBenchmarkDataset load_benchmark_dataset(
    const BenchmarkSetup& setup);

}  // namespace ultrahigh_ann::benchmark
