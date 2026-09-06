#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ultrahigh_ann::benchmark {

enum class DistanceMetric {
    l1,
    l2,
};

enum class IndexKind {
    exact,
    flat,
    uniform,
    hierarchical,
};

enum class ExecutionBackend {
    cpu,
    cuda,
};

enum class QueryStrategy {
    sequential,
    parallel_queries,
    parallel_representatives,
    automatic,
    direct,
    gemm,
};

enum class ProbabilityPolicy {
    sequential,
    cpu_parallel,
    gpu_fp32,
    gpu_cublas_fp32,
    load,
};

enum class DiagnosticMode {
    none,
    selected_distances,
    full_distance_table,
};

struct BenchmarkRun {
    std::string name;
    IndexKind index{};
    ExecutionBackend backend{ExecutionBackend::cpu};
    QueryStrategy strategy{QueryStrategy::sequential};
    std::size_t repetitions{};
    std::uint64_t seed{};
    std::size_t projection_dimension{};
    std::vector<std::size_t> batch_sizes{1};
    std::size_t warmups{};
    std::size_t trials{1};
};

struct BenchmarkSetup {
    std::filesystem::path setup_path;
    std::filesystem::path dataset_directory;
    std::filesystem::path representatives_path;
    std::filesystem::path representative_labels_path;
    std::filesystem::path queries_path;
    std::filesystem::path query_labels_path;
    std::filesystem::path json_output_path;
    std::filesystem::path csv_output_path;
    std::optional<std::filesystem::path> probabilities_path;
    DistanceMetric distance{DistanceMetric::l2};
    ProbabilityPolicy probability_policy{ProbabilityPolicy::sequential};
    DiagnosticMode diagnostics{DiagnosticMode::selected_distances};
    std::size_t maximum_queries{};
    std::size_t warmups{};
    std::size_t trials{1};
    std::vector<std::size_t> batch_sizes{1};
    int device{};
    std::string reference_run;
    std::vector<BenchmarkRun> runs;
};

[[nodiscard]] BenchmarkSetup
load_benchmark_setup(const std::filesystem::path& path);

[[nodiscard]] std::string_view distance_name(DistanceMetric value) noexcept;
[[nodiscard]] std::string_view index_name(IndexKind value) noexcept;
[[nodiscard]] std::string_view backend_name(ExecutionBackend value) noexcept;
[[nodiscard]] std::string_view strategy_name(QueryStrategy value) noexcept;
[[nodiscard]] std::string_view
probability_policy_name(ProbabilityPolicy value) noexcept;
[[nodiscard]] std::string_view
diagnostic_mode_name(DiagnosticMode value) noexcept;

}  // namespace ultrahigh_ann::benchmark
