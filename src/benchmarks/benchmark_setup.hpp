#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ultrahigh_ann::benchmark {

enum class DistanceMetric {
    l1,
    l2,
};

enum class ApproximateMethod {
    flat,
    uniform,
    hierarchical,
};

struct ApproximateRun {
    std::string name;
    ApproximateMethod method{};
    std::size_t repetitions{};
    std::uint64_t seed{};
    std::size_t projection_dimension{};
};

struct BenchmarkSetup {
    std::filesystem::path setup_path;
    std::filesystem::path dataset_directory;
    std::filesystem::path output_path;
    DistanceMetric distance{DistanceMetric::l1};
    std::size_t maximum_queries{};
    std::vector<ApproximateRun> runs;
};

[[nodiscard]] BenchmarkSetup load_benchmark_setup(
    const std::filesystem::path& path);

[[nodiscard]] std::string_view method_name(ApproximateMethod method) noexcept;

[[nodiscard]] std::string_view distance_name(DistanceMetric distance) noexcept;

}  // namespace ultrahigh_ann::benchmark
