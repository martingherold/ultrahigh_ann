#include "benchmark_dataset.hpp"
#include "benchmark_metrics.hpp"
#include "benchmark_setup.hpp"
#include "ultrahigh_ann_build_config.hpp"

#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"
#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l1_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/uniform_l1_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/uniform_l2_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/uniform_probabilities.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"
#include "ultrahigh_ann/datasets/representative_query_dataset.hpp"
#include "ultrahigh_ann/exact/l1/cuda_exact_l1_index.hpp"
#include "ultrahigh_ann/exact/l1/exact_l1_index.hpp"
#include "ultrahigh_ann/exact/l2/cuda_exact_l2_index.hpp"
#include "ultrahigh_ann/exact/l2/exact_l2_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/flat/cuda_l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/flat/l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/hierarchical/cuda_hierarchical_l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"
#include "ultrahigh_ann/hnsvw25/l2/flat/l2_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l2/hierarchical/cuda_hierarchical_l2_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"
#include "ultrahigh_ann/io/importance_probability_io.hpp"
#include "ultrahigh_ann/io/sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ultrahigh_ann::DenseMatrix;
using ultrahigh_ann::IndexSpaceUsage;
using ultrahigh_ann::RepresentativeQueryDataset;
using ultrahigh_ann::benchmark::ApproximationMetrics;
using ultrahigh_ann::benchmark::BenchmarkRun;
using ultrahigh_ann::benchmark::BenchmarkSetup;
using ultrahigh_ann::benchmark::DiagnosticMode;
using ultrahigh_ann::benchmark::DistanceMetric;
using ultrahigh_ann::benchmark::ExecutionBackend;
using ultrahigh_ann::benchmark::IndexKind;
using ultrahigh_ann::benchmark::LoadedBenchmarkDataset;
using ultrahigh_ann::benchmark::ProbabilityPolicy;
using ultrahigh_ann::benchmark::QueryStrategy;

struct Options {
    std::filesystem::path setup_path;
    bool validate_only{};
};

struct GpuProbabilityExecution {
    std::string device_name;
    int device{};
    std::string distance_backend;
    std::size_t pair_count{};
    std::size_t pair_chunks{};
    std::size_t device_working_set_bytes{};
    double host_to_device_ms{};
    double inverse_distance_ms{};
    double coordinate_maximum_ms{};
    double device_to_host_ms{};
    double total_ms{};
};

struct ProbabilityExecution {
    std::vector<double> probabilities;
    double sampling_mass{};
    double build_ms{};
    double load_ms{};
    std::optional<std::filesystem::path> source_path;
    std::optional<ultrahigh_ann::io::Sha256Digest> source_sha256;
    std::optional<ultrahigh_ann::io::Sha256Digest>
        representatives_sha256;
    std::optional<GpuProbabilityExecution> gpu;
};

struct QueryMeasurement {
    std::size_t batch_size{};
    std::vector<double> trial_ms;
    std::vector<std::size_t> predictions;
    std::size_t workspace_payload_bytes{};
    std::size_t correct{};
    std::size_t exact_choice_agreements{};
    std::size_t exact_label_agreements{};
    std::optional<ApproximationMetrics> approximation;
};

struct RunExecution {
    BenchmarkRun run;
    double build_ms{};
    IndexSpaceUsage space_usage;
    std::vector<QueryMeasurement> measurements;
    std::optional<int> device;
    std::optional<std::string> device_name;
};

struct HostRuntimeInfo {
    std::optional<std::string> processor_model;
    std::optional<std::uint64_t> physical_memory_bytes;
};

[[nodiscard]] double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

[[nodiscard]] HostRuntimeInfo host_runtime_info()
{
    HostRuntimeInfo result;
#if defined(__linux__)
    {
        std::ifstream input("/proc/cpuinfo");
        std::string line;
        while (std::getline(input, line)) {
            if (!line.starts_with("model name")) {
                continue;
            }
            const std::size_t colon = line.find(':');
            if (colon != std::string::npos) {
                const std::size_t value = line.find_first_not_of(" \t", colon + 1);
                if (value != std::string::npos) {
                    result.processor_model = line.substr(value);
                }
            }
            break;
        }
    }
    {
        std::ifstream input("/proc/meminfo");
        std::string line;
        while (std::getline(input, line)) {
            if (!line.starts_with("MemTotal:")) {
                continue;
            }
            std::istringstream fields(line.substr(9));
            std::uint64_t kibibytes{};
            std::string unit;
            if (fields >> kibibytes >> unit && unit == "kB" &&
                kibibytes <=
                    std::numeric_limits<std::uint64_t>::max() / 1024U) {
                result.physical_memory_bytes = kibibytes * 1024U;
            }
            break;
        }
    }
#endif
    return result;
}

void print_usage(std::string_view program)
{
    std::cout
        << "Usage: " << program << " --setup FILE [--validate-only]\n\n"
        << "Run a version-two CPU/CUDA nearest-representative benchmark and "
           "write a complete JSON report plus a trial-level CSV.\n\n"
        << "  --setup FILE      Version-two benchmark configuration\n"
        << "  --validate-only   Validate configuration without loading data\n"
        << "  -h, --help        Show this help\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--validate-only") {
            options.validate_only = true;
            continue;
        }
        if (argument == "--setup") {
            if (index + 1 >= argc || !options.setup_path.empty()) {
                throw std::invalid_argument(
                    "--setup expects exactly one value");
            }
            options.setup_path = argv[++index];
            continue;
        }
        throw std::invalid_argument("unknown option: " + std::string(argument));
    }
    if (options.setup_path.empty()) {
        throw std::invalid_argument("--setup is required");
    }
    return options;
}

void require_file(const std::filesystem::path& path, std::string_view label)
{
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(std::string(label) +
                                 " is not a regular file: " + path.string());
    }
}

[[nodiscard]] ultrahigh_ann::io::ImportanceProbabilityKind
probability_kind(DistanceMetric distance) noexcept
{
    return distance == DistanceMetric::l1
               ? ultrahigh_ann::io::ImportanceProbabilityKind::l1
               : ultrahigh_ann::io::ImportanceProbabilityKind::l2;
}

[[nodiscard]] ultrahigh_ann::ExecutionPolicy
execution_policy(ProbabilityPolicy policy)
{
    switch (policy) {
    case ProbabilityPolicy::sequential:
        return ultrahigh_ann::ExecutionPolicy::sequential;
    case ProbabilityPolicy::cpu_parallel:
        return ultrahigh_ann::ExecutionPolicy::cpu_parallel;
    case ProbabilityPolicy::gpu_fp32:
        return ultrahigh_ann::ExecutionPolicy::gpu_fp32;
    case ProbabilityPolicy::gpu_cublas_fp32:
        return ultrahigh_ann::ExecutionPolicy::gpu_cublas_fp32;
    case ProbabilityPolicy::load:
        break;
    }
    throw std::logic_error("load has no computation execution policy");
}

[[nodiscard]] GpuProbabilityExecution gpu_execution(
    const ultrahigh_ann::L1GpuProbabilityResult& result)
{
    return GpuProbabilityExecution{
        .device_name = result.device_name,
        .device = result.device,
        .distance_backend = "direct",
        .pair_count = result.pair_count,
        .pair_chunks = result.pair_chunks,
        .device_working_set_bytes = result.device_working_set_bytes,
        .host_to_device_ms = result.timings.host_to_device_ms,
        .inverse_distance_ms = result.timings.inverse_distance_ms,
        .coordinate_maximum_ms = result.timings.coordinate_maximum_ms,
        .device_to_host_ms = result.timings.device_to_host_ms,
        .total_ms = result.timings.total_ms,
    };
}

[[nodiscard]] GpuProbabilityExecution gpu_execution(
    const ultrahigh_ann::L2GpuProbabilityResult& result)
{
    return GpuProbabilityExecution{
        .device_name = result.device_name,
        .device = result.device,
        .distance_backend =
            result.distance_backend ==
                    ultrahigh_ann::L2GpuDistanceBackend::cublas
                ? "cublas"
                : "direct",
        .pair_count = result.pair_count,
        .pair_chunks = result.pair_chunks,
        .device_working_set_bytes = result.device_working_set_bytes,
        .host_to_device_ms = result.timings.host_to_device_ms,
        .inverse_distance_ms = result.timings.inverse_distance_ms,
        .coordinate_maximum_ms = result.timings.coordinate_maximum_ms,
        .device_to_host_ms = result.timings.device_to_host_ms,
        .total_ms = result.timings.total_ms,
    };
}

[[nodiscard]] ProbabilityExecution
obtain_probabilities(const BenchmarkSetup& setup,
                     const DenseMatrix& representatives,
                     const ultrahigh_ann::io::Sha256Digest&
                         representatives_sha256)
{
    if (setup.probability_policy == ProbabilityPolicy::load) {
        const auto start = Clock::now();
        const auto source_sha256 = ultrahigh_ann::io::sha256_file(
            *setup.probabilities_path);
        auto file = ultrahigh_ann::io::load_importance_probabilities(
            *setup.probabilities_path);
        if (!ultrahigh_ann::io::importance_probability_source_matches(
                file, probability_kind(setup.distance), representatives.rows(),
                representatives.cols(), representatives_sha256)) {
            throw std::runtime_error(
                "probability metadata or representative-matrix digest does "
                "not match the configured dataset; regenerate the "
                "probability file");
        }
        const double mass =
            ultrahigh_ann::compute_sampling_mass(file.probabilities);
        return ProbabilityExecution{
            .probabilities = std::move(file.probabilities),
            .sampling_mass = mass,
            .build_ms = 0.0,
            .load_ms = elapsed_ms(start),
            .source_path = setup.probabilities_path,
            .source_sha256 = source_sha256,
            .representatives_sha256 = file.representatives_sha256,
            .gpu = std::nullopt,
        };
    }

    const auto start = Clock::now();
    ProbabilityExecution result;
    if (setup.distance == DistanceMetric::l1) {
        if (setup.probability_policy == ProbabilityPolicy::gpu_fp32) {
            auto gpu =
                ultrahigh_ann::compute_l1_importance_probabilities_gpu(
                    representatives, setup.device);
            result.gpu = gpu_execution(gpu);
            result.probabilities = std::move(gpu.probabilities);
        } else {
            result.probabilities =
                ultrahigh_ann::compute_l1_importance_probabilities(
                    representatives,
                    execution_policy(setup.probability_policy));
        }
    } else if (setup.probability_policy == ProbabilityPolicy::gpu_fp32 ||
               setup.probability_policy == ProbabilityPolicy::gpu_cublas_fp32) {
        const auto distance_backend =
            setup.probability_policy == ProbabilityPolicy::gpu_cublas_fp32
                ? ultrahigh_ann::L2GpuDistanceBackend::cublas
                : ultrahigh_ann::L2GpuDistanceBackend::direct;
        auto gpu = ultrahigh_ann::compute_l2_importance_probabilities_gpu(
            representatives, setup.device, 128, distance_backend);
        result.gpu = gpu_execution(gpu);
        result.probabilities = std::move(gpu.probabilities);
    } else {
        result.probabilities =
            ultrahigh_ann::compute_l2_importance_probabilities(
                representatives, execution_policy(setup.probability_policy));
    }
    result.sampling_mass =
        ultrahigh_ann::compute_sampling_mass(result.probabilities);
    result.build_ms = elapsed_ms(start);
    result.representatives_sha256 = representatives_sha256;
    return result;
}

[[nodiscard]] bool requires_probabilities(const BenchmarkSetup& setup)
{
    return std::ranges::any_of(setup.runs, [](const BenchmarkRun& run) {
        return run.index != IndexKind::exact;
    });
}

[[nodiscard]] std::vector<std::size_t>
effective_batches(const BenchmarkRun& run, std::size_t query_count)
{
    std::vector<std::size_t> result;
    std::unordered_set<std::size_t> seen;
    for (const std::size_t requested : run.batch_sizes) {
        const std::size_t batch = std::min(requested, query_count);
        if (seen.insert(batch).second) {
            result.push_back(batch);
        }
    }
    return result;
}

template <class Function>
[[nodiscard]] QueryMeasurement
measure_queries(const BenchmarkRun& run, std::size_t batch_size,
                std::size_t query_count, std::size_t workspace_payload_bytes,
                Function&& function)
{
    std::vector<std::size_t> predictions(query_count);
    for (std::size_t warmup = 0; warmup < run.warmups; ++warmup) {
        function(predictions);
    }
    std::vector<double> times;
    times.reserve(run.trials);
    for (std::size_t trial = 0; trial < run.trials; ++trial) {
        const auto start = Clock::now();
        function(predictions);
        times.push_back(elapsed_ms(start));
    }
    return QueryMeasurement{
        .batch_size = batch_size,
        .trial_ms = std::move(times),
        .predictions = std::move(predictions),
        .workspace_payload_bytes = workspace_payload_bytes,
        .correct = 0,
        .exact_choice_agreements = 0,
        .exact_label_agreements = 0,
        .approximation = std::nullopt,
    };
}

template <class Index>
void run_sequential(const Index& index, const DenseMatrix& queries,
                    std::size_t query_count, std::span<std::size_t> output)
{
    for (std::size_t row = 0; row < query_count; ++row) {
        output[row] = index.query(queries.row(row));
    }
}

template <class Index, class Workspace>
void run_sequential_workspace(const Index& index, const DenseMatrix& queries,
                              std::size_t query_count,
                              std::span<std::size_t> output,
                              Workspace& workspace)
{
    for (std::size_t row = 0; row < query_count; ++row) {
        output[row] = index.query(queries.row(row), workspace);
    }
}

[[nodiscard]] ultrahigh_ann::CpuDenseL2QueryStrategy
cpu_l2_strategy(QueryStrategy strategy)
{
    switch (strategy) {
    case QueryStrategy::sequential:
        return ultrahigh_ann::CpuDenseL2QueryStrategy::sequential;
    case QueryStrategy::parallel_queries:
        return ultrahigh_ann::CpuDenseL2QueryStrategy::parallel_queries;
    case QueryStrategy::parallel_representatives:
        return ultrahigh_ann::CpuDenseL2QueryStrategy::parallel_representatives;
    case QueryStrategy::automatic:
        return ultrahigh_ann::CpuDenseL2QueryStrategy::automatic;
    case QueryStrategy::direct:
    case QueryStrategy::gemm:
        break;
    }
    throw std::logic_error("invalid CPU L2 query strategy");
}

template <class Index>
void run_cpu_l2_batched(const Index& index, const DenseMatrix& queries,
                        std::size_t query_count, std::size_t batch_size,
                        std::span<std::size_t> output, QueryStrategy strategy)
{
    const std::size_t dimension = queries.cols();
    for (std::size_t begin = 0; begin < query_count; begin += batch_size) {
        const std::size_t count = std::min(batch_size, query_count - begin);
        index.query_batch(
            queries.values().subspan(begin * dimension, count * dimension),
            count, output.subspan(begin, count), cpu_l2_strategy(strategy));
    }
}

template <class Index, class Workspace, class Strategy>
void run_cuda_batched(const Index& index, const DenseMatrix& queries,
                      std::size_t query_count, std::size_t batch_size,
                      std::span<std::size_t> output, Workspace& workspace,
                      Strategy strategy)
{
    const std::size_t dimension = queries.cols();
    for (std::size_t begin = 0; begin < query_count; begin += batch_size) {
        const std::size_t count = std::min(batch_size, query_count - begin);
        index.query_batch(
            queries.values().subspan(begin * dimension, count * dimension),
            count, output.subspan(begin, count), workspace, strategy);
    }
}

template <class Index, class Workspace>
void run_cuda_batched(const Index& index, const DenseMatrix& queries,
                      std::size_t query_count, std::size_t batch_size,
                      std::span<std::size_t> output, Workspace& workspace)
{
    const std::size_t dimension = queries.cols();
    for (std::size_t begin = 0; begin < query_count; begin += batch_size) {
        const std::size_t count = std::min(batch_size, query_count - begin);
        index.query_batch(
            queries.values().subspan(begin * dimension, count * dimension),
            count, output.subspan(begin, count), workspace);
    }
}

[[nodiscard]] ultrahigh_ann::CudaExactL2QueryStrategy
cuda_exact_strategy(QueryStrategy strategy)
{
    return strategy == QueryStrategy::direct
               ? ultrahigh_ann::CudaExactL2QueryStrategy::direct
               : ultrahigh_ann::CudaExactL2QueryStrategy::gemm;
}

[[nodiscard]] ultrahigh_ann::CudaSampledCoordinateL2QueryStrategy
cuda_sampled_strategy(QueryStrategy strategy)
{
    return strategy == QueryStrategy::direct
               ? ultrahigh_ann::CudaSampledCoordinateL2QueryStrategy::direct
               : ultrahigh_ann::CudaSampledCoordinateL2QueryStrategy::gemm;
}

[[nodiscard]] ultrahigh_ann::CudaHierarchicalL2QueryStrategy
cuda_hierarchical_strategy(QueryStrategy strategy)
{
    return strategy == QueryStrategy::direct
               ? ultrahigh_ann::CudaHierarchicalL2QueryStrategy::direct
               : ultrahigh_ann::CudaHierarchicalL2QueryStrategy::gemm;
}

[[nodiscard]] ultrahigh_ann::CudaHierarchicalL1QueryStrategy
cuda_hierarchical_l1_strategy(QueryStrategy strategy)
{
    return strategy == QueryStrategy::direct
               ? ultrahigh_ann::CudaHierarchicalL1QueryStrategy::direct
               : ultrahigh_ann::CudaHierarchicalL1QueryStrategy::gemm;
}

template <class Index>
[[nodiscard]] RunExecution execute_cpu_sequential_index(
    const BenchmarkRun& run, const RepresentativeQueryDataset& dataset,
    std::size_t query_count, Index&& index, double build_ms)
{
    RunExecution execution{
        .run = run,
        .build_ms = build_ms,
        .space_usage = index.space_usage(),
        .measurements = {},
        .device = std::nullopt,
        .device_name = std::nullopt,
    };
    for (const std::size_t batch : effective_batches(run, query_count)) {
        execution.measurements.push_back(
            measure_queries(run, batch, query_count,
                            execution.space_usage.query_workspace_payload_bytes,
                            [&](std::span<std::size_t> predictions) {
                                run_sequential(index, dataset.queries,
                                               query_count, predictions);
                            }));
    }
    return execution;
}

template <class Index>
[[nodiscard]] RunExecution execute_cuda_direct_index(
    const BenchmarkRun& run, const RepresentativeQueryDataset& dataset,
    std::size_t query_count, const Index& index, double build_ms)
{
    RunExecution execution{
        .run = run,
        .build_ms = build_ms,
        .space_usage = index.space_usage(),
        .measurements = {},
        .device = index.device(),
        .device_name = index.device_name(),
    };
    for (const std::size_t batch : effective_batches(run, query_count)) {
        auto workspace = index.make_query_workspace(batch);
        execution.measurements.push_back(measure_queries(
            run, batch, query_count, workspace.payload_bytes(),
            [&](std::span<std::size_t> predictions) {
                run_cuda_batched(index, dataset.queries, query_count, batch,
                                 predictions, workspace);
            }));
    }
    return execution;
}

template <class Index>
[[nodiscard]] RunExecution
execute_cpu_l2_index(const BenchmarkRun& run,
                     const RepresentativeQueryDataset& dataset,
                     std::size_t query_count, Index&& index, double build_ms)
{
    RunExecution execution{
        .run = run,
        .build_ms = build_ms,
        .space_usage = index.space_usage(),
        .measurements = {},
        .device = std::nullopt,
        .device_name = std::nullopt,
    };
    for (const std::size_t batch : effective_batches(run, query_count)) {
        execution.measurements.push_back(measure_queries(
            run, batch, query_count,
            execution.space_usage.query_workspace_payload_bytes,
            [&](std::span<std::size_t> predictions) {
                run_cpu_l2_batched(index, dataset.queries, query_count, batch,
                                   predictions, run.strategy);
            }));
    }
    return execution;
}

template <class Index>
[[nodiscard]] RunExecution execute_hierarchical_index(
    const BenchmarkRun& run, const RepresentativeQueryDataset& dataset,
    std::size_t query_count, Index&& index, double build_ms)
{
    RunExecution execution{
        .run = run,
        .build_ms = build_ms,
        .space_usage = index.space_usage(),
        .measurements = {},
        .device = std::nullopt,
        .device_name = std::nullopt,
    };
    for (const std::size_t batch : effective_batches(run, query_count)) {
        auto workspace = index.make_query_workspace();
        execution.measurements.push_back(measure_queries(
            run, batch, query_count,
            execution.space_usage.query_workspace_payload_bytes,
            [&](std::span<std::size_t> predictions) {
                run_sequential_workspace(index, dataset.queries, query_count,
                                         predictions, workspace);
            }));
    }
    return execution;
}

[[nodiscard]] RunExecution
execute_exact_run(const BenchmarkSetup& setup, const BenchmarkRun& run,
                  const RepresentativeQueryDataset& dataset,
                  std::size_t query_count)
{
    if (run.backend == ExecutionBackend::cuda) {
        const auto start = Clock::now();
        if (setup.distance == DistanceMetric::l1) {
            const ultrahigh_ann::CudaExactL1Index index(dataset.representatives,
                                                        setup.device);
            return execute_cuda_direct_index(run, dataset, query_count, index,
                                             elapsed_ms(start));
        }
        const ultrahigh_ann::CudaExactL2Index index(dataset.representatives,
                                                    setup.device);
        const double build_ms = elapsed_ms(start);
        RunExecution execution{
            .run = run,
            .build_ms = build_ms,
            .space_usage = index.space_usage(),
            .measurements = {},
            .device = index.device(),
            .device_name = index.device_name(),
        };
        for (const std::size_t batch : effective_batches(run, query_count)) {
            auto workspace = index.make_query_workspace(batch);
            execution.measurements.push_back(measure_queries(
                run, batch, query_count, workspace.payload_bytes(),
                [&](std::span<std::size_t> predictions) {
                    run_cuda_batched(index, dataset.queries, query_count, batch,
                                     predictions, workspace,
                                     cuda_exact_strategy(run.strategy));
                }));
        }
        return execution;
    }

    const auto start = Clock::now();
    if (setup.distance == DistanceMetric::l1) {
        const ultrahigh_ann::ExactL1Index index(dataset.representatives);
        return execute_cpu_sequential_index(run, dataset, query_count, index,
                                            elapsed_ms(start));
    }
    const ultrahigh_ann::ExactL2Index index(dataset.representatives);
    return execute_cpu_l2_index(run, dataset, query_count, index,
                                elapsed_ms(start));
}

[[nodiscard]] RunExecution
execute_approximate_run(const BenchmarkSetup& setup, const BenchmarkRun& run,
                        const RepresentativeQueryDataset& dataset,
                        const ProbabilityExecution& probabilities,
                        std::span<const double> uniform,
                        std::size_t query_count)
{
    std::mt19937_64 random_engine(run.seed);
    if (run.backend == ExecutionBackend::cuda) {
        const auto start = Clock::now();
        if (setup.distance == DistanceMetric::l1 &&
            run.index == IndexKind::hierarchical) {
            const ultrahigh_ann::CudaHierarchicalL1AnnIndex index(
                dataset.representatives, probabilities.probabilities,
                run.repetitions, run.projection_dimension, random_engine,
                setup.device);
            const double build_ms = elapsed_ms(start);
            RunExecution execution{
                .run = run,
                .build_ms = build_ms,
                .space_usage = index.space_usage(),
                .measurements = {},
                .device = index.device(),
                .device_name = index.device_name(),
            };
            for (const std::size_t batch :
                 effective_batches(run, query_count)) {
                auto workspace = index.make_query_workspace(batch);
                execution.measurements.push_back(measure_queries(
                    run, batch, query_count, workspace.payload_bytes(),
                    [&](std::span<std::size_t> predictions) {
                        run_cuda_batched(
                            index, dataset.queries, query_count, batch,
                            predictions, workspace,
                            cuda_hierarchical_l1_strategy(run.strategy));
                    }));
            }
            return execution;
        }
        if (setup.distance == DistanceMetric::l1 &&
            run.index == IndexKind::flat) {
            const ultrahigh_ann::CudaFlatL1AnnIndex index(
                dataset.representatives, probabilities.probabilities,
                run.repetitions, random_engine, setup.device);
            return execute_cuda_direct_index(run, dataset, query_count, index,
                                             elapsed_ms(start));
        }
        if (setup.distance == DistanceMetric::l2 &&
            run.index == IndexKind::hierarchical) {
            const ultrahigh_ann::CudaHierarchicalL2AnnIndex index(
                dataset.representatives, probabilities.probabilities,
                run.repetitions, run.projection_dimension, random_engine,
                setup.device);
            const double build_ms = elapsed_ms(start);
            RunExecution execution{
                .run = run,
                .build_ms = build_ms,
                .space_usage = index.space_usage(),
                .measurements = {},
                .device = index.device(),
                .device_name = index.device_name(),
            };
            for (const std::size_t batch :
                 effective_batches(run, query_count)) {
                auto workspace = index.make_query_workspace(batch);
                execution.measurements.push_back(measure_queries(
                    run, batch, query_count, workspace.payload_bytes(),
                    [&](std::span<std::size_t> predictions) {
                        run_cuda_batched(
                            index, dataset.queries, query_count, batch,
                            predictions, workspace,
                            cuda_hierarchical_strategy(run.strategy));
                    }));
            }
            return execution;
        }
        const std::span<const double> sampling_probabilities =
            run.index == IndexKind::uniform
                ? uniform
                : std::span<const double>{probabilities.probabilities};
        auto sample = ultrahigh_ann::build_coordinate_sample(
            sampling_probabilities, run.repetitions, random_engine);
        if (setup.distance == DistanceMetric::l1) {
            const ultrahigh_ann::CudaSampledCoordinateL1AnnIndex index(
                dataset.representatives, std::move(sample), setup.device);
            return execute_cuda_direct_index(run, dataset, query_count, index,
                                             elapsed_ms(start));
        }
        const ultrahigh_ann::CudaSampledCoordinateL2AnnIndex index(
            dataset.representatives, std::move(sample), setup.device);
        const double build_ms = elapsed_ms(start);
        RunExecution execution{
            .run = run,
            .build_ms = build_ms,
            .space_usage = index.space_usage(),
            .measurements = {},
            .device = index.device(),
            .device_name = index.device_name(),
        };
        for (const std::size_t batch : effective_batches(run, query_count)) {
            auto workspace = index.make_query_workspace(batch);
            execution.measurements.push_back(measure_queries(
                run, batch, query_count, workspace.payload_bytes(),
                [&](std::span<std::size_t> predictions) {
                    run_cuda_batched(index, dataset.queries, query_count, batch,
                                     predictions, workspace,
                                     cuda_sampled_strategy(run.strategy));
                }));
        }
        return execution;
    }

    const auto start = Clock::now();
    if (run.index == IndexKind::flat) {
        if (setup.distance == DistanceMetric::l1) {
            const ultrahigh_ann::FlatL1AnnIndex index(
                dataset.representatives, probabilities.probabilities,
                run.repetitions, random_engine);
            return execute_cpu_sequential_index(run, dataset, query_count,
                                                index, elapsed_ms(start));
        }
        const ultrahigh_ann::FlatL2AnnIndex index(
            dataset.representatives, probabilities.probabilities,
            run.repetitions, random_engine);
        return execute_cpu_l2_index(run, dataset, query_count, index,
                                    elapsed_ms(start));
    }
    if (run.index == IndexKind::uniform) {
        if (setup.distance == DistanceMetric::l1) {
            const ultrahigh_ann::UniformL1AnnIndex index(
                dataset.representatives, probabilities.sampling_mass,
                run.repetitions, random_engine);
            return execute_cpu_sequential_index(run, dataset, query_count,
                                                index, elapsed_ms(start));
        }
        const ultrahigh_ann::UniformL2AnnIndex index(
            dataset.representatives, probabilities.sampling_mass,
            run.repetitions, random_engine);
        return execute_cpu_l2_index(run, dataset, query_count, index,
                                    elapsed_ms(start));
    }
    if (setup.distance == DistanceMetric::l1) {
        const ultrahigh_ann::HierarchicalL1AnnIndex index(
            dataset.representatives, probabilities.probabilities,
            run.repetitions, run.projection_dimension, random_engine);
        return execute_hierarchical_index(run, dataset, query_count, index,
                                          elapsed_ms(start));
    }
    const ultrahigh_ann::HierarchicalL2AnnIndex index(
        dataset.representatives, probabilities.probabilities, run.repetitions,
        run.projection_dimension, random_engine);
    return execute_hierarchical_index(run, dataset, query_count, index,
                                      elapsed_ms(start));
}

[[nodiscard]] double median(std::vector<double> values)
{
    std::ranges::sort(values);
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                  : values[middle];
}

void add_basic_quality(QueryMeasurement& measurement,
                       const LoadedBenchmarkDataset& dataset,
                       std::size_t query_count,
                       std::span<const std::size_t> reference)
{
    for (std::size_t index = 0; index < query_count; ++index) {
        const std::size_t candidate = measurement.predictions[index];
        const std::size_t exact = reference[index];
        if (candidate >= dataset.values.representatives.rows() ||
            exact >= dataset.values.representatives.rows()) {
            throw std::logic_error("an index returned an invalid row");
        }
        measurement.exact_choice_agreements +=
            static_cast<std::size_t>(candidate == exact);
        if (dataset.labels_available) {
            measurement.correct += static_cast<std::size_t>(
                dataset.values.representative_labels[candidate] ==
                dataset.values.query_labels[index]);
            measurement.exact_label_agreements += static_cast<std::size_t>(
                dataset.values.representative_labels[candidate] ==
                dataset.values.representative_labels[exact]);
        }
    }
}

struct DiagnosticExecution {
    double build_ms{};
    double evaluation_ms{};
    std::size_t payload_bytes{};
    std::optional<ultrahigh_ann::benchmark::QueryGeometryMetrics> geometry;
};

[[nodiscard]] DiagnosticExecution
add_diagnostics(const BenchmarkSetup& setup,
                const LoadedBenchmarkDataset& dataset, std::size_t query_count,
                std::span<const std::size_t> reference,
                std::vector<RunExecution>& executions)
{
    for (RunExecution& execution : executions) {
        for (QueryMeasurement& measurement : execution.measurements) {
            add_basic_quality(measurement, dataset, query_count, reference);
        }
    }
    if (setup.diagnostics == DiagnosticMode::none) {
        return {};
    }
    if (setup.diagnostics == DiagnosticMode::selected_distances) {
        const auto start = Clock::now();
        for (RunExecution& execution : executions) {
            if (execution.run.name == setup.reference_run) {
                continue;
            }
            QueryMeasurement& first = execution.measurements.front();
            first.approximation =
                ultrahigh_ann::benchmark::evaluate_selected_distances(
                    setup.distance, dataset.values.representatives,
                    dataset.values.queries, query_count, reference,
                    first.predictions);
            for (std::size_t index = 1; index < execution.measurements.size();
                 ++index) {
                QueryMeasurement& measurement = execution.measurements[index];
                if (std::ranges::equal(measurement.predictions,
                                       first.predictions)) {
                    measurement.approximation = first.approximation;
                } else {
                    measurement.approximation =
                        ultrahigh_ann::benchmark::evaluate_selected_distances(
                            setup.distance, dataset.values.representatives,
                            dataset.values.queries, query_count, reference,
                            measurement.predictions);
                }
            }
        }
        return DiagnosticExecution{
            .build_ms = 0.0,
            .evaluation_ms = elapsed_ms(start),
            .payload_bytes = 0,
            .geometry = std::nullopt,
        };
    }

    const auto build_start = Clock::now();
    const ultrahigh_ann::benchmark::ExactDistanceTable table(
        setup.distance, dataset.values.representatives, dataset.values.queries,
        query_count);
    DiagnosticExecution result{
        .build_ms = elapsed_ms(build_start),
        .evaluation_ms = 0.0,
        .payload_bytes = table.payload_bytes(),
        .geometry = table.query_geometry(),
    };
    const auto evaluation_start = Clock::now();
    for (RunExecution& execution : executions) {
        if (execution.run.name == setup.reference_run) {
            continue;
        }
        QueryMeasurement& first = execution.measurements.front();
        first.approximation = table.evaluate(first.predictions);
        for (std::size_t index = 1; index < execution.measurements.size();
             ++index) {
            QueryMeasurement& measurement = execution.measurements[index];
            measurement.approximation =
                std::ranges::equal(measurement.predictions, first.predictions)
                    ? first.approximation
                    : std::optional<ApproximationMetrics>{
                          table.evaluate(measurement.predictions)};
        }
    }
    result.evaluation_ms = elapsed_ms(evaluation_start);
    return result;
}

[[nodiscard]] std::string utc_timestamp()
{
    const std::time_t value =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &value) != 0) {
        throw std::runtime_error("failed to create UTC timestamp");
    }
#else
    if (gmtime_r(&value, &utc) == nullptr) {
        throw std::runtime_error("failed to create UTC timestamp");
    }
#endif
    std::array<char, 32> buffer{};
    if (std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ",
                      &utc) == 0) {
        throw std::runtime_error("failed to format UTC timestamp");
    }
    return buffer.data();
}

void write_json_string(std::ostream& output, std::string_view value)
{
    constexpr std::string_view hex{"0123456789abcdef"};
    output.put('"');
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u00" << hex[character >> 4U]
                       << hex[character & 0x0fU];
            } else {
                output.put(static_cast<char>(character));
            }
        }
    }
    output.put('"');
}

void write_optional_string(std::ostream& output,
                           const std::optional<std::string>& value)
{
    if (value.has_value()) {
        write_json_string(output, *value);
    } else {
        output << "null";
    }
}

void write_optional_uint64(std::ostream& output,
                           const std::optional<std::uint64_t>& value)
{
    if (value.has_value()) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_optional_digest(
    std::ostream& output,
    const std::optional<ultrahigh_ann::io::Sha256Digest>& value)
{
    if (value.has_value()) {
        write_json_string(output, ultrahigh_ann::io::sha256_hex(*value));
    } else {
        output << "null";
    }
}

[[nodiscard]] std::string cuda_version_string(int encoded)
{
    if (encoded <= 0) {
        return "unknown";
    }
    const int major = encoded / 1000;
    const int minor = (encoded % 1000) / 10;
    const int patch = encoded % 10;
    std::string result = std::to_string(major) + "." + std::to_string(minor);
    if (patch != 0) {
        result += "." + std::to_string(patch);
    }
    return result;
}

void write_optional_number(std::ostream& output,
                           const std::optional<double>& value)
{
    if (value.has_value()) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_distribution(
    std::ostream& output,
    const ultrahigh_ann::benchmark::DistributionSummary& distribution)
{
    output << "{\"count\":" << distribution.count << ",\"mean\":";
    write_optional_number(output, distribution.mean);
    output << ",\"median\":";
    write_optional_number(output, distribution.median);
    output << ",\"percentile_95\":";
    write_optional_number(output, distribution.percentile_95);
    output << ",\"percentile_99\":";
    write_optional_number(output, distribution.percentile_99);
    output << ",\"maximum\":";
    write_optional_number(output, distribution.maximum);
    output << '}';
}

void write_approximation(std::ostream& output,
                         const std::optional<ApproximationMetrics>& value)
{
    if (!value.has_value()) {
        output << "null";
        return;
    }
    const auto& metrics = *value;
    const double count = static_cast<double>(metrics.query_count);
    output << "{\"distance_optimal_count\":"
           << metrics.optimal_representative_count
           << ",\"distance_optimal_rate\":"
           << static_cast<double>(metrics.optimal_representative_count) / count
           << ",\"non_optimal_count\":" << metrics.non_optimal_count
           << ",\"non_optimal_rate\":"
           << static_cast<double>(metrics.non_optimal_count) / count
           << ",\"reference_improvement_count\":"
           << metrics.reference_improvement_count << ",\"distance_ratio\":";
    write_distribution(output, metrics.distance_ratio);
    output << ",\"zero_optimum_query_count\":"
           << metrics.zero_optimum_query_count
           << ",\"zero_optimum_non_optimal_count\":"
           << metrics.zero_optimum_non_optimal_count
           << ",\"non_optimal_ratio_count\":"
           << metrics.conditional_non_optimal_ratio_count
           << ",\"non_optimal_distance_ratio_mean\":";
    write_optional_number(output, metrics.conditional_mean_distance_ratio);
    output << ",\"non_optimal_relative_excess_mean\":";
    write_optional_number(output, metrics.conditional_mean_relative_excess);
    output << ",\"approximation_guarantee_failures\":[";
    for (std::size_t index = 0; index < metrics.approximation_failures.size();
         ++index) {
        const auto& failure = metrics.approximation_failures[index];
        output << "{\"epsilon\":" << failure.epsilon
               << ",\"violation_count\":" << failure.violation_count
               << ",\"violation_rate\":"
               << static_cast<double>(failure.violation_count) / count << '}'
               << (index + 1 == metrics.approximation_failures.size() ? ""
                                                                      : ",");
    }
    output << "],\"returned_representative_rank\":";
    if (metrics.returned_representative_rank.count == 0) {
        output << "null";
    } else {
        write_distribution(output, metrics.returned_representative_rank);
    }
    output << ",\"by_multiplicative_margin\":[";
    for (std::size_t index = 0; index < metrics.margin_buckets.size();
         ++index) {
        const auto& bucket = metrics.margin_buckets[index];
        output << "{\"lower_inclusive\":" << bucket.lower_inclusive
               << ",\"upper_exclusive\":";
        write_optional_number(output, bucket.upper_exclusive);
        output << ",\"query_count\":" << bucket.query_count
               << ",\"non_optimal_count\":" << bucket.non_optimal_count
               << ",\"ratio_eligible_count\":" << bucket.ratio_eligible_count
               << ",\"mean_distance_ratio\":";
        write_optional_number(output, bucket.mean_distance_ratio);
        output << '}'
               << (index + 1 == metrics.margin_buckets.size() ? "" : ",");
    }
    output << ']';
    output << '}';
}

void write_query_geometry(
    std::ostream& output,
    const std::optional<ultrahigh_ann::benchmark::QueryGeometryMetrics>& value)
{
    if (!value.has_value()) {
        output << "null";
        return;
    }
    const auto& geometry = *value;
    output << "{\"query_count\":" << geometry.query_count
           << ",\"zero_optimum_query_count\":"
           << geometry.zero_optimum_query_count
           << ",\"non_unique_optimum_query_count\":"
           << geometry.non_unique_optimum_query_count
           << ",\"finite_multiplicative_margin\":";
    write_distribution(output, geometry.finite_multiplicative_margin);
    output << ",\"infinite_multiplicative_margin_count\":"
           << geometry.infinite_multiplicative_margin_count
           << ",\"margin_buckets\":[";
    for (std::size_t index = 0; index < geometry.margin_buckets.size();
         ++index) {
        const auto& bucket = geometry.margin_buckets[index];
        output << "{\"lower_inclusive\":" << bucket.lower_inclusive
               << ",\"upper_exclusive\":";
        write_optional_number(output, bucket.upper_exclusive);
        output << ",\"query_count\":" << bucket.query_count << '}'
               << (index + 1 == geometry.margin_buckets.size() ? "" : ",");
    }
    output << "]}";
}

[[nodiscard]] std::filesystem::path
portable_path(const std::filesystem::path& path)
{
    std::error_code error;
    const auto absolute = std::filesystem::absolute(path, error);
    if (error) {
        return path;
    }
    const auto relative =
        absolute.lexically_relative(std::filesystem::current_path(error));
    if (!error && !relative.empty() &&
        relative.begin()->generic_string() != "..") {
        return relative;
    }
    return absolute;
}

void write_provenance(std::ostream& output, const BenchmarkSetup& setup)
{
    const HostRuntimeInfo host = host_runtime_info();
    const auto cuda = ultrahigh_ann::cuda_runtime_info(setup.device);
    const std::string setup_sha256 = ultrahigh_ann::io::sha256_hex(
        ultrahigh_ann::io::sha256_file(setup.setup_path));

    output << ",\n  \"provenance\": {\"source\":{\"project_version\":";
    write_json_string(output, ULTRAHIGH_ANN_VERSION);
    output << ",\"git\":{\"available\":"
           << (ULTRAHIGH_ANN_GIT_AVAILABLE != 0 ? "true" : "false")
           << ",\"commit\":";
    if (ULTRAHIGH_ANN_GIT_AVAILABLE != 0) {
        write_json_string(output, ULTRAHIGH_ANN_GIT_COMMIT);
    } else {
        output << "null";
    }
    output << ",\"dirty\":";
    if (ULTRAHIGH_ANN_GIT_AVAILABLE != 0) {
        output << (ULTRAHIGH_ANN_GIT_DIRTY != 0 ? "true" : "false");
    } else {
        output << "null";
    }
    output << "}},\"build\":{\"cmake_version\":";
    write_json_string(output, ULTRAHIGH_ANN_CMAKE_VERSION);
    output << ",\"build_type\":";
    write_json_string(output, ULTRAHIGH_ANN_BUILD_TYPE);
    output << ",\"cxx\":{\"standard\":20,\"compiler_id\":";
    write_json_string(output, ULTRAHIGH_ANN_CXX_COMPILER_ID);
    output << ",\"compiler_version\":";
    write_json_string(output, ULTRAHIGH_ANN_CXX_COMPILER_VERSION);
    output << ",\"flags\":";
    write_json_string(output, ULTRAHIGH_ANN_CXX_FLAGS);
    output << "},\"cuda\":{\"enabled\":"
           << (ULTRAHIGH_ANN_CUDA_ENABLED != 0 ? "true" : "false")
           << ",\"compiler_id\":";
    if (ULTRAHIGH_ANN_CUDA_ENABLED != 0) {
        write_json_string(output, ULTRAHIGH_ANN_CUDA_COMPILER_ID);
    } else {
        output << "null";
    }
    output << ",\"compiler_version\":";
    if (ULTRAHIGH_ANN_CUDA_ENABLED != 0) {
        write_json_string(output, ULTRAHIGH_ANN_CUDA_COMPILER_VERSION);
    } else {
        output << "null";
    }
    output << ",\"architectures\":";
    if (ULTRAHIGH_ANN_CUDA_ENABLED != 0) {
        write_json_string(output, ULTRAHIGH_ANN_CUDA_ARCHITECTURES);
    } else {
        output << "null";
    }
    output << ",\"flags\":";
    if (ULTRAHIGH_ANN_CUDA_ENABLED != 0) {
        write_json_string(output, ULTRAHIGH_ANN_CUDA_FLAGS);
    } else {
        output << "null";
    }
    output << "}},\"host\":{\"operating_system\":{\"name\":";
    write_json_string(output, ULTRAHIGH_ANN_SYSTEM_NAME);
    output << ",\"version\":";
    write_json_string(output, ULTRAHIGH_ANN_SYSTEM_VERSION);
    output << ",\"architecture\":";
    write_json_string(output, ULTRAHIGH_ANN_SYSTEM_PROCESSOR);
    output << "},\"processor_model\":";
    write_optional_string(output, host.processor_model);
    output << ",\"physical_memory_bytes\":";
    write_optional_uint64(output, host.physical_memory_bytes);
    output << "},\"cuda_runtime\":";
    if (cuda.has_value()) {
        output << "{\"device\":" << cuda->device << ",\"device_name\":";
        write_json_string(output, cuda->device_name);
        output << ",\"compute_capability\":";
        write_json_string(
            output, std::to_string(cuda->compute_capability_major) + "." +
                        std::to_string(cuda->compute_capability_minor));
        output << ",\"total_global_memory_bytes\":"
               << cuda->total_global_memory_bytes
               << ",\"compiled_runtime\":{\"encoded\":"
               << cuda->compiled_runtime_version << ",\"version\":";
        write_json_string(
            output, cuda_version_string(cuda->compiled_runtime_version));
        output << "},\"runtime\":{\"encoded\":" << cuda->runtime_version
               << ",\"version\":";
        write_json_string(output, cuda_version_string(cuda->runtime_version));
        output << "},\"driver\":{\"encoded\":" << cuda->driver_version
               << ",\"version\":";
        write_json_string(output, cuda_version_string(cuda->driver_version));
        output << "}}";
    } else {
        output << "null";
    }
    output << ",\"invocation\":{\"arguments\":[\"--setup\",";
    write_json_string(output, portable_path(setup.setup_path).generic_string());
    output << "],\"setup_sha256\":";
    write_json_string(output, setup_sha256);
    output << "}}";
}

void ensure_parent(const std::filesystem::path& path)
{
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
}

void write_json_report(const BenchmarkSetup& setup,
                       const LoadedBenchmarkDataset& dataset,
                       std::size_t query_count,
                       const ProbabilityExecution& probabilities,
                       const DiagnosticExecution& diagnostics,
                       std::span<const RunExecution> executions)
{
    ensure_parent(setup.json_output_path);
    std::ofstream output(setup.json_output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open JSON output: " +
                                 setup.json_output_path.string());
    }
    output << std::setprecision(17);
    output << "{\n  \"schema_version\": 5,\n  \"generated_at_utc\": ";
    write_json_string(output, utc_timestamp());
    output << ",\n  \"setup_file\": ";
    write_json_string(output, portable_path(setup.setup_path).generic_string());
    write_provenance(output, setup);
    output << ",\n  \"outputs\": {\"json\":";
    write_json_string(output,
                      portable_path(setup.json_output_path).generic_string());
    output << ",\"csv\":";
    write_json_string(output,
                      portable_path(setup.csv_output_path).generic_string());
    output << "},\n  \"dataset\": {\"directory\":";
    write_json_string(output,
                      portable_path(setup.dataset_directory).generic_string());
    output << ",\"representatives_file\":";
    write_json_string(
        output, portable_path(setup.representatives_path).generic_string());
    output << ",\"representative_labels_file\":";
    if (dataset.labels_available) {
        write_json_string(
            output,
            portable_path(setup.representative_labels_path).generic_string());
    } else {
        output << "null";
    }
    output << ",\"queries_file\":";
    write_json_string(output,
                      portable_path(setup.queries_path).generic_string());
    output << ",\"query_labels_file\":";
    if (dataset.labels_available) {
        write_json_string(
            output, portable_path(setup.query_labels_path).generic_string());
    } else {
        output << "null";
    }
    output << ",\"representatives_sha256\":";
    write_json_string(
        output,
        ultrahigh_ann::io::sha256_hex(dataset.representatives_sha256));
    output << ",\"representative_labels_sha256\":";
    write_optional_digest(output, dataset.representative_labels_sha256);
    output << ",\"queries_sha256\":";
    write_json_string(output,
                      ultrahigh_ann::io::sha256_hex(dataset.queries_sha256));
    output << ",\"query_labels_sha256\":";
    write_optional_digest(output, dataset.query_labels_sha256);
    output << ",\"representative_count\":"
           << dataset.values.representatives.rows()
           << ",\"query_count_available\":" << dataset.values.queries.rows()
           << ",\"query_count_run\":" << query_count
           << ",\"dimension\":" << dataset.values.representatives.cols()
           << ",\"labels_available\":"
           << (dataset.labels_available ? "true" : "false")
           << ",\"hash_ms\":" << dataset.hash_ms
           << ",\"load_ms\":" << dataset.load_ms << "},\n"
           << "  \"settings\": {\"distance\":";
    write_json_string(output,
                      ultrahigh_ann::benchmark::distance_name(setup.distance));
    output << ",\"reference_run\":";
    write_json_string(output, setup.reference_run);
    output << ",\"max_queries\":" << setup.maximum_queries
           << ",\"device\":" << setup.device << ",\"diagnostics\":";
    write_json_string(output, ultrahigh_ann::benchmark::diagnostic_mode_name(
                                  setup.diagnostics));
    output << "},\n  \"sampling_probabilities\": {\"required\":"
           << (requires_probabilities(setup) ? "true" : "false")
           << ",\"policy\":";
    write_json_string(output, ultrahigh_ann::benchmark::probability_policy_name(
                                  setup.probability_policy));
    output << ",\"source_file\":";
    if (probabilities.source_path.has_value()) {
        write_json_string(
            output, portable_path(*probabilities.source_path).generic_string());
    } else {
        output << "null";
    }
    output << ",\"source_sha256\":";
    write_optional_digest(output, probabilities.source_sha256);
    output << ",\"representatives_sha256\":";
    write_optional_digest(output, probabilities.representatives_sha256);
    output << ",\"build_ms\":" << probabilities.build_ms
           << ",\"load_ms\":" << probabilities.load_ms
           << ",\"coordinate_count\":" << probabilities.probabilities.size()
           << ",\"sampling_mass\":" << probabilities.sampling_mass;
    if (probabilities.gpu.has_value()) {
        output << ",\"gpu\":{\"device\":" << probabilities.gpu->device
               << ",\"device_name\":";
        write_json_string(output, probabilities.gpu->device_name);
        output << ",\"distance_backend\":";
        write_json_string(output, probabilities.gpu->distance_backend);
        output << ",\"arithmetic\":\"fp32\""
               << ",\"pair_count\":" << probabilities.gpu->pair_count
               << ",\"pair_chunks\":" << probabilities.gpu->pair_chunks
               << ",\"device_working_set_bytes\":"
               << probabilities.gpu->device_working_set_bytes
               << ",\"host_to_device_ms\":"
               << probabilities.gpu->host_to_device_ms
               << ",\"inverse_distance_ms\":"
               << probabilities.gpu->inverse_distance_ms
               << ",\"coordinate_maximum_ms\":"
               << probabilities.gpu->coordinate_maximum_ms
               << ",\"device_to_host_ms\":"
               << probabilities.gpu->device_to_host_ms
               << ",\"total_ms\":" << probabilities.gpu->total_ms
               << '}';
    } else {
        output << ",\"gpu\":null";
    }
    output << "},\n  \"diagnostic_execution\": {\"build_ms\":"
           << diagnostics.build_ms
           << ",\"evaluation_ms\":" << diagnostics.evaluation_ms
           << ",\"payload_bytes\":" << diagnostics.payload_bytes
           << ",\"query_geometry\":";
    write_query_geometry(output, diagnostics.geometry);
    output << "},\n  \"runs\": [\n";

    for (std::size_t run_index = 0; run_index < executions.size();
         ++run_index) {
        const RunExecution& execution = executions[run_index];
        output << "    {\"name\":";
        write_json_string(output, execution.run.name);
        output << ",\"index\":";
        write_json_string(
            output, ultrahigh_ann::benchmark::index_name(execution.run.index));
        output << ",\"backend\":";
        write_json_string(output, ultrahigh_ann::benchmark::backend_name(
                                      execution.run.backend));
        output << ",\"strategy\":";
        write_json_string(output, ultrahigh_ann::benchmark::strategy_name(
                                      execution.run.strategy));
        output << ",\"reference\":"
               << (execution.run.name == setup.reference_run ? "true" : "false")
               << ",\"repetitions\":" << execution.run.repetitions
               << ",\"seed\":" << execution.run.seed
               << ",\"projection_dimension\":"
               << execution.run.projection_dimension
               << ",\"warmups\":" << execution.run.warmups
               << ",\"trials\":" << execution.run.trials
               << ",\"build_ms\":" << execution.build_ms
               << ",\"space\":{\"index_payload_bytes\":"
               << execution.space_usage.index_payload_bytes
               << ",\"query_workspace_payload_bytes\":"
               << execution.space_usage.query_workspace_payload_bytes
               << ",\"unique_coordinates\":"
               << execution.space_usage.unique_query_coordinates
               << ",\"sampled_multiplicity\":"
               << execution.space_usage.sampled_multiplicity << '}';
        output << ",\"device\":";
        if (execution.device.has_value()) {
            output << "{\"index\":" << *execution.device << ",\"name\":";
            write_json_string(output, *execution.device_name);
            output << '}';
        } else {
            output << "null";
        }
        output << ",\"measurements\":[";
        for (std::size_t measurement_index = 0;
             measurement_index < execution.measurements.size();
             ++measurement_index) {
            const QueryMeasurement& measurement =
                execution.measurements[measurement_index];
            const double med = median(measurement.trial_ms);
            output << "{\"batch_size\":" << measurement.batch_size
                   << ",\"workspace_payload_bytes\":"
                   << measurement.workspace_payload_bytes << ",\"trial_ms\":[";
            for (std::size_t trial = 0; trial < measurement.trial_ms.size();
                 ++trial) {
                output << measurement.trial_ms[trial]
                       << (trial + 1 == measurement.trial_ms.size() ? "" : ",");
            }
            output << "],\"median_ms\":" << med
                   << ",\"median_microseconds_per_query\":"
                   << med * 1000.0 / static_cast<double>(query_count)
                   << ",\"median_queries_per_second\":"
                   << static_cast<double>(query_count) * 1000.0 / med
                   << ",\"exact_choice_agreement_count\":"
                   << measurement.exact_choice_agreements
                   << ",\"exact_choice_agreement\":"
                   << static_cast<double>(measurement.exact_choice_agreements) /
                          static_cast<double>(query_count)
                   << ",\"correct\":";
            if (dataset.labels_available) {
                output << measurement.correct;
            } else {
                output << "null";
            }
            output << ",\"accuracy\":";
            if (dataset.labels_available) {
                output << static_cast<double>(measurement.correct) /
                              static_cast<double>(query_count);
            } else {
                output << "null";
            }
            output << ",\"label_agreement_with_exact\":";
            if (dataset.labels_available) {
                output << static_cast<double>(
                              measurement.exact_label_agreements) /
                              static_cast<double>(query_count);
            } else {
                output << "null";
            }
            output << ",\"approximation\":";
            write_approximation(output, measurement.approximation);
            output << '}'
                   << (measurement_index + 1 == execution.measurements.size()
                           ? ""
                           : ",");
        }
        output << "]}" << (run_index + 1 == executions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) {
        throw std::runtime_error("failed while writing JSON output");
    }
}

void write_csv_field(std::ostream& output, std::string_view value)
{
    output.put('"');
    for (const char character : value) {
        if (character == '"') {
            output.put('"');
        }
        output.put(character);
    }
    output.put('"');
}

void write_csv_optional(std::ostream& output,
                        const std::optional<double>& value)
{
    if (value.has_value()) {
        output << *value;
    }
}

void write_csv_report(const BenchmarkSetup& setup,
                      const LoadedBenchmarkDataset& dataset,
                      std::size_t query_count,
                      std::span<const RunExecution> executions)
{
    ensure_parent(setup.csv_output_path);
    std::ofstream output(setup.csv_output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open CSV output: " +
                                 setup.csv_output_path.string());
    }
    output << "run_name,index,backend,strategy,reference,repetitions,seed,"
              "projection_dimension,batch_size,warmups,trial,representatives,"
              "dimension,queries,build_ms,elapsed_ms,microseconds_per_query,"
              "queries_per_second,correct,accuracy,exact_choice_agreement,"
              "label_agreement_with_exact,distance_optimal_rate,"
              "distance_ratio_mean,distance_ratio_median,distance_ratio_p95,"
              "distance_ratio_p99,distance_ratio_max,violation_rate_eps_0_001,"
              "violation_rate_eps_0_005,violation_rate_eps_0_01,"
              "violation_rate_eps_0_02,violation_rate_eps_0_05,"
              "violation_rate_eps_0_10,index_payload_bytes,"
              "workspace_payload_bytes,unique_coordinates,"
              "sampled_multiplicity,device,device_name\n";
    output << std::setprecision(17);
    for (const RunExecution& execution : executions) {
        for (const QueryMeasurement& measurement : execution.measurements) {
            for (std::size_t trial = 0; trial < measurement.trial_ms.size();
                 ++trial) {
                const double time = measurement.trial_ms[trial];
                write_csv_field(output, execution.run.name);
                output << ','
                       << ultrahigh_ann::benchmark::index_name(
                              execution.run.index)
                       << ','
                       << ultrahigh_ann::benchmark::backend_name(
                              execution.run.backend)
                       << ','
                       << ultrahigh_ann::benchmark::strategy_name(
                              execution.run.strategy)
                       << ','
                       << (execution.run.name == setup.reference_run ? 1 : 0)
                       << ',' << execution.run.repetitions << ','
                       << execution.run.seed << ','
                       << execution.run.projection_dimension << ','
                       << measurement.batch_size << ',' << execution.run.warmups
                       << ',' << trial << ','
                       << dataset.values.representatives.rows() << ','
                       << dataset.values.representatives.cols() << ','
                       << query_count << ',' << execution.build_ms << ','
                       << time << ','
                       << time * 1000.0 / static_cast<double>(query_count)
                       << ','
                       << static_cast<double>(query_count) * 1000.0 / time
                       << ',';
                if (dataset.labels_available) {
                    output << measurement.correct << ','
                           << static_cast<double>(measurement.correct) /
                                  static_cast<double>(query_count);
                } else {
                    output << ',';
                }
                output << ','
                       << static_cast<double>(
                              measurement.exact_choice_agreements) /
                              static_cast<double>(query_count)
                       << ',';
                if (dataset.labels_available) {
                    output << static_cast<double>(
                                  measurement.exact_label_agreements) /
                                  static_cast<double>(query_count);
                }
                output << ',';
                if (measurement.approximation.has_value()) {
                    const auto& approximation = *measurement.approximation;
                    output << static_cast<double>(
                                  approximation.optimal_representative_count) /
                                  static_cast<double>(query_count)
                           << ',';
                    write_csv_optional(output,
                                       approximation.distance_ratio.mean);
                    output << ',';
                    write_csv_optional(output,
                                       approximation.distance_ratio.median);
                    output << ',';
                    write_csv_optional(
                        output, approximation.distance_ratio.percentile_95);
                    output << ',';
                    write_csv_optional(
                        output, approximation.distance_ratio.percentile_99);
                    output << ',';
                    write_csv_optional(output,
                                       approximation.distance_ratio.maximum);
                    for (const auto& failure :
                         approximation.approximation_failures) {
                        output << ','
                               << static_cast<double>(failure.violation_count) /
                                      static_cast<double>(query_count);
                    }
                } else {
                    output << ",,,,,,,,,,,";
                }
                output << ',' << execution.space_usage.index_payload_bytes
                       << ',' << measurement.workspace_payload_bytes << ','
                       << execution.space_usage.unique_query_coordinates << ','
                       << execution.space_usage.sampled_multiplicity << ',';
                if (execution.device.has_value()) {
                    output << *execution.device << ',';
                    write_csv_field(output, *execution.device_name);
                } else {
                    output << ',';
                }
                output << '\n';
            }
        }
    }
    if (!output) {
        throw std::runtime_error("failed while writing CSV output");
    }
}

void print_execution(const RunExecution& execution, std::size_t query_count)
{
    for (const QueryMeasurement& measurement : execution.measurements) {
        const double time = median(measurement.trial_ms);
        std::cout
            << "run=" << execution.run.name << " index="
            << ultrahigh_ann::benchmark::index_name(execution.run.index)
            << " backend="
            << ultrahigh_ann::benchmark::backend_name(execution.run.backend)
            << " strategy="
            << ultrahigh_ann::benchmark::strategy_name(execution.run.strategy)
            << " batch_size=" << measurement.batch_size
            << " median_ms=" << std::fixed << std::setprecision(3) << time
            << " us_per_query="
            << time * 1000.0 / static_cast<double>(query_count)
            << " exact_choice_agreement=" << measurement.exact_choice_agreements
            << '/' << query_count << '\n';
    }
}

int run_benchmark(const BenchmarkSetup& setup)
{
    require_file(setup.representatives_path, "representatives");
    require_file(setup.queries_path, "queries");
    if (setup.probability_policy == ProbabilityPolicy::load) {
        require_file(*setup.probabilities_path, "probabilities");
    }
    const bool requests_cuda =
        std::ranges::any_of(setup.runs, [](const BenchmarkRun& run) {
            return run.backend == ExecutionBackend::cuda;
        });
    if (requests_cuda && !(setup.distance == DistanceMetric::l1
                               ? ultrahigh_ann::cuda_exact_l1_available()
                               : ultrahigh_ann::cuda_exact_l2_available())) {
        throw std::runtime_error(
            "the setup requests CUDA, but no CUDA device is available");
    }

    LoadedBenchmarkDataset dataset =
        ultrahigh_ann::benchmark::load_benchmark_dataset(setup);
    const std::size_t query_count =
        setup.maximum_queries == 0
            ? dataset.values.queries.rows()
            : std::min(setup.maximum_queries, dataset.values.queries.rows());
    std::cout << "Loaded " << dataset.values.representatives.rows() << " x "
              << dataset.values.representatives.cols()
              << " representatives and " << dataset.values.queries.rows()
              << " queries; running " << query_count << ".\n";

    ProbabilityExecution probabilities;
    std::vector<double> uniform;
    if (requires_probabilities(setup)) {
        std::cout << "Obtaining shared sampling probabilities with policy="
                  << ultrahigh_ann::benchmark::probability_policy_name(
                         setup.probability_policy)
                  << "..." << std::endl;
        probabilities =
            obtain_probabilities(setup, dataset.values.representatives,
                                 dataset.representatives_sha256);
        uniform = ultrahigh_ann::make_mass_matched_uniform_probabilities(
            dataset.values.representatives.cols(), probabilities.sampling_mass);
    }

    const auto reference_position =
        std::ranges::find_if(setup.runs, [&](const BenchmarkRun& run) {
            return run.name == setup.reference_run;
        });
    std::vector<const BenchmarkRun*> order;
    order.reserve(setup.runs.size());
    order.push_back(&*reference_position);
    for (const BenchmarkRun& run : setup.runs) {
        if (run.name != setup.reference_run) {
            order.push_back(&run);
        }
    }

    std::vector<RunExecution> executions;
    executions.reserve(order.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        const BenchmarkRun& run = *order[index];
        std::cout << '[' << index + 1 << '/' << order.size() << "] Building "
                  << run.name << " ("
                  << ultrahigh_ann::benchmark::backend_name(run.backend) << ' '
                  << ultrahigh_ann::benchmark::index_name(run.index) << ")..."
                  << std::endl;
        if (run.index == IndexKind::exact) {
            executions.push_back(
                execute_exact_run(setup, run, dataset.values, query_count));
        } else {
            executions.push_back(
                execute_approximate_run(setup, run, dataset.values,
                                        probabilities, uniform, query_count));
        }
    }

    const std::span<const std::size_t> reference =
        executions.front().measurements.front().predictions;
    const DiagnosticExecution diagnostic_execution =
        add_diagnostics(setup, dataset, query_count, reference, executions);
    for (const RunExecution& execution : executions) {
        print_execution(execution, query_count);
    }

    write_json_report(setup, dataset, query_count, probabilities,
                      diagnostic_execution, executions);
    write_csv_report(setup, dataset, query_count, executions);
    std::cout << "Wrote JSON report to " << setup.json_output_path << '\n'
              << "Wrote CSV trials to " << setup.csv_output_path << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);
        const BenchmarkSetup setup =
            ultrahigh_ann::benchmark::load_benchmark_setup(options.setup_path);
        if (options.validate_only) {
            std::cout << "valid_setup=" << setup.setup_path << '\n'
                      << "runs=" << setup.runs.size() << '\n'
                      << "reference=" << setup.reference_run << '\n'
                      << "json_output=" << setup.json_output_path << '\n'
                      << "csv_output=" << setup.csv_output_path << '\n';
            return 0;
        }
        return run_benchmark(setup);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
