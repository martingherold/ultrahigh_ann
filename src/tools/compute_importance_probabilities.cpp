#include "compute_importance_probabilities.hpp"

#include "ultrahigh_ann/coordinate_sampling/uniform_probabilities.hpp"
#include "ultrahigh_ann/datasets/representative_query_dataset.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"
#include "ultrahigh_ann/io/importance_probability_io.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace ultrahigh_ann::tools {
namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    ExecutionPolicy policy{ExecutionPolicy::sequential};
    int gpu_device{};
    std::size_t gpu_pair_chunks{128};
};

struct GpuMetadata {
    std::string device_name;
    int device{};
    std::size_t pair_count{};
    std::size_t pair_chunks{};
    std::size_t device_working_set_bytes{};
    std::string_view distance_backend;
    double host_to_device_ms{};
    double inverse_distance_ms{};
    double coordinate_maximum_ms{};
    double device_to_host_ms{};
    double total_ms{};
};

[[nodiscard]] std::string_view distance_name(ProbabilityDistance distance)
{
    return distance == ProbabilityDistance::l1 ? "L1" : "L2";
}

[[nodiscard]] std::string_view policy_name(ExecutionPolicy policy) noexcept
{
    switch (policy) {
    case ExecutionPolicy::sequential:
        return "sequential";
    case ExecutionPolicy::cpu_parallel:
        return "cpu_parallel";
    case ExecutionPolicy::gpu_fp32:
        return "gpu_fp32";
    case ExecutionPolicy::gpu_cublas_fp32:
        return "gpu_cublas_fp32";
    }
    return "unknown";
}

void print_usage(std::string_view program, ProbabilityDistance distance)
{
    std::cout << "Usage: " << program
              << " --input FILE --output FILE --policy POLICY\n\n"
              << "Compute reusable " << distance_name(distance)
              << " importance-sampling probabilities from a float32 NPY "
                 "matrix of reference vectors.\n\n"
              << "Options:\n"
              << "  --input FILE      Float32 NPY matrix; one reference vector per row\n"
              << "  --output FILE     Source-bound UAP v2 output file\n"
              << "  --policy POLICY   sequential, cpu_parallel, gpu_fp32";
    if (distance == ProbabilityDistance::l2) {
        std::cout << ", or gpu_cublas_fp32";
    }
    std::cout << "\n"
              << "  --device N        CUDA device index (default: 0)\n"
              << "  --gpu-pair-chunks N  CUDA pair batches (default: 128)\n"
              << "  -h, --help        Show this help\n\n"
              << "The output embeds the input matrix SHA-256. The program "
                 "reports matrix-load, hashing, probability-computation, "
                 "file-write, and total wall-clock runtimes.\n";
}

[[nodiscard]] ExecutionPolicy parse_policy(std::string_view value,
                                           ProbabilityDistance distance)
{
    if (value == "sequential") {
        return ExecutionPolicy::sequential;
    }
    if (value == "cpu_parallel") {
        return ExecutionPolicy::cpu_parallel;
    }
    if (value == "gpu_fp32") {
        return ExecutionPolicy::gpu_fp32;
    }
    if (value == "gpu_cublas_fp32") {
        if (distance == ProbabilityDistance::l1) {
            throw std::invalid_argument(
                "gpu_cublas_fp32 is unavailable for L1 probability "
                "computation; L1 distance has no matrix-multiplication "
                "identity");
        }
        return ExecutionPolicy::gpu_cublas_fp32;
    }
    std::string expected =
        "--policy expects sequential, cpu_parallel, or gpu_fp32";
    if (distance == ProbabilityDistance::l2) {
        expected += ", or gpu_cublas_fp32";
    }
    throw std::invalid_argument(expected);
}

[[nodiscard]] bool is_gpu_policy(ExecutionPolicy policy) noexcept
{
    return policy == ExecutionPolicy::gpu_fp32 ||
           policy == ExecutionPolicy::gpu_cublas_fp32;
}

template <class Integer>
[[nodiscard]] Integer parse_integer(std::string_view text,
                                    std::string_view option, bool allow_zero)
{
    Integer value{};
    const auto [position, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    bool invalid = error != std::errc{} ||
                   position != text.data() + text.size() ||
                   (!allow_zero && value == 0);
    if constexpr (std::is_signed_v<Integer>) {
        invalid = invalid || value < 0;
    }
    if (invalid) {
        throw std::invalid_argument(
            std::string(option) + (allow_zero ? " expects a nonnegative integer"
                                              : " expects a positive integer"));
    }
    return value;
}

[[nodiscard]] Options parse_options(int argc, char** argv,
                                    ProbabilityDistance distance)
{
    Options options;
    bool input_seen = false;
    bool output_seen = false;
    bool policy_seen = false;
    bool gpu_option_seen = false;
    bool device_seen = false;
    bool pair_chunks_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0], distance);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(std::string(argument) +
                                        " expects a value");
        }
        const std::string_view value{argv[++index]};
        if (argument == "--input") {
            if (input_seen) {
                throw std::invalid_argument(
                    "--input may only be specified once");
            }
            options.input_path = value;
            input_seen = true;
        } else if (argument == "--output") {
            if (output_seen) {
                throw std::invalid_argument(
                    "--output may only be specified once");
            }
            options.output_path = value;
            output_seen = true;
        } else if (argument == "--policy") {
            if (policy_seen) {
                throw std::invalid_argument(
                    "--policy may only be specified once");
            }
            options.policy = parse_policy(value, distance);
            policy_seen = true;
        } else if (argument == "--device") {
            if (device_seen) {
                throw std::invalid_argument(
                    "--device may only be specified once");
            }
            options.gpu_device = parse_integer<int>(value, argument, true);
            gpu_option_seen = true;
            device_seen = true;
        } else if (argument == "--gpu-pair-chunks") {
            if (pair_chunks_seen) {
                throw std::invalid_argument(
                    "--gpu-pair-chunks may only be specified once");
            }
            options.gpu_pair_chunks =
                parse_integer<std::size_t>(value, argument, false);
            gpu_option_seen = true;
            pair_chunks_seen = true;
        } else {
            throw std::invalid_argument("unknown option: " +
                                        std::string(argument));
        }
    }
    if (!input_seen || !output_seen || !policy_seen) {
        throw std::invalid_argument(
            "--input, --output, and --policy are required");
    }
    if (options.input_path == options.output_path) {
        throw std::invalid_argument(
            "--input and --output must refer to different paths");
    }
    if (gpu_option_seen && !is_gpu_policy(options.policy)) {
        throw std::invalid_argument(
            "--device and --gpu-pair-chunks require a GPU policy");
    }
    return options;
}

[[nodiscard]] double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

[[nodiscard]] GpuMetadata gpu_metadata(const L1GpuProbabilityResult& result)
{
    return GpuMetadata{
        .device_name = result.device_name,
        .device = result.device,
        .pair_count = result.pair_count,
        .pair_chunks = result.pair_chunks,
        .device_working_set_bytes = result.device_working_set_bytes,
        .distance_backend = "direct",
        .host_to_device_ms = result.timings.host_to_device_ms,
        .inverse_distance_ms = result.timings.inverse_distance_ms,
        .coordinate_maximum_ms = result.timings.coordinate_maximum_ms,
        .device_to_host_ms = result.timings.device_to_host_ms,
        .total_ms = result.timings.total_ms,
    };
}

[[nodiscard]] GpuMetadata gpu_metadata(const L2GpuProbabilityResult& result)
{
    return GpuMetadata{
        .device_name = result.device_name,
        .device = result.device,
        .pair_count = result.pair_count,
        .pair_chunks = result.pair_chunks,
        .device_working_set_bytes = result.device_working_set_bytes,
        .distance_backend =
            result.distance_backend == L2GpuDistanceBackend::cublas ? "cublas"
                                                                    : "direct",
        .host_to_device_ms = result.timings.host_to_device_ms,
        .inverse_distance_ms = result.timings.inverse_distance_ms,
        .coordinate_maximum_ms = result.timings.coordinate_maximum_ms,
        .device_to_host_ms = result.timings.device_to_host_ms,
        .total_ms = result.timings.total_ms,
    };
}

}  // namespace

int run_compute_importance_probabilities(int argc, char** argv,
                                         ProbabilityDistance distance)
{
    try {
        const Options options = parse_options(argc, argv, distance);
        const auto total_start = Clock::now();

        const auto load_start = Clock::now();
        const DenseMatrix representatives =
            load_float_matrix_npy(options.input_path);
        const double load_ms = elapsed_ms(load_start);
        if (representatives.rows() < 2) {
            throw std::runtime_error(
                "the matrix of reference vectors must contain at least two rows");
        }
        if (representatives.cols() == 0) {
            throw std::runtime_error(
                "the matrix of reference vectors must contain at least one column");
        }

        const auto hash_start = Clock::now();
        const io::Sha256Digest representatives_sha256 =
            io::sha256_file(options.input_path);
        const double hash_ms = elapsed_ms(hash_start);

        const auto compute_start = Clock::now();
        std::vector<double> probabilities;
        std::optional<GpuMetadata> gpu;
        if (distance == ProbabilityDistance::l1) {
            if (options.policy == ExecutionPolicy::gpu_fp32) {
                auto result = compute_l1_importance_probabilities_gpu(
                    representatives, options.gpu_device,
                    options.gpu_pair_chunks);
                gpu = gpu_metadata(result);
                probabilities = std::move(result.probabilities);
            } else {
                probabilities = compute_l1_importance_probabilities(
                    representatives, options.policy);
            }
        } else if (is_gpu_policy(options.policy)) {
            const auto distance_backend =
                options.policy == ExecutionPolicy::gpu_cublas_fp32
                    ? L2GpuDistanceBackend::cublas
                    : L2GpuDistanceBackend::direct;
            auto result = compute_l2_importance_probabilities_gpu(
                representatives, options.gpu_device, options.gpu_pair_chunks,
                distance_backend);
            gpu = gpu_metadata(result);
            probabilities = std::move(result.probabilities);
        } else {
            probabilities = compute_l2_importance_probabilities(representatives,
                                                                options.policy);
        }
        const double compute_ms = elapsed_ms(compute_start);
        const double sampling_mass = compute_sampling_mass(probabilities);

        const auto write_start = Clock::now();
        io::save_importance_probabilities(
            options.output_path,
            distance == ProbabilityDistance::l1
                ? io::ImportanceProbabilityKind::l1
                : io::ImportanceProbabilityKind::l2,
            representatives.rows(), representatives_sha256, probabilities);
        const double write_ms = elapsed_ms(write_start);
        const double total_ms = elapsed_ms(total_start);

        std::cout << std::fixed << std::setprecision(3)
                  << "input=" << options.input_path << '\n'
                  << "output=" << options.output_path << '\n'
                  << "policy=" << policy_name(options.policy) << '\n';
        if (gpu.has_value()) {
            std::cout << "device=" << gpu->device << '\n'
                      << "device_name=" << gpu->device_name << '\n'
                      << "gpu_arithmetic=fp32\n"
                      << "pair_count=" << gpu->pair_count << '\n'
                      << "pair_chunks=" << gpu->pair_chunks << '\n'
                      << "gpu_distance_backend=" << gpu->distance_backend
                      << '\n'
                      << "device_working_set_bytes="
                      << gpu->device_working_set_bytes << '\n'
                      << "gpu_host_to_device_ms=" << gpu->host_to_device_ms
                      << '\n'
                      << "gpu_inverse_distance_ms=" << gpu->inverse_distance_ms
                      << '\n'
                      << "gpu_coordinate_maximum_ms="
                      << gpu->coordinate_maximum_ms << '\n'
                      << "gpu_device_to_host_ms=" << gpu->device_to_host_ms
                      << '\n'
                      << "gpu_total_ms=" << gpu->total_ms << '\n';
        } else {
            const int worker_count =
                options.policy == ExecutionPolicy::cpu_parallel
                    ? static_cast<int>(std::min<std::size_t>(
                          representatives.rows() - 1,
                          static_cast<std::size_t>(omp_get_max_threads())))
                    : 1;
            std::cout << "workers=" << worker_count << '\n';
        }
        std::cout << "reference_vectors=" << representatives.rows() << '\n'
                  << "dimension=" << representatives.cols() << '\n'
                  << "reference_vectors_sha256="
                  << io::sha256_hex(representatives_sha256) << '\n'
                  << "sampling_mass=" << sampling_mass << '\n'
                  << "load_ms=" << load_ms << '\n'
                  << "hash_ms=" << hash_ms << '\n'
                  << "compute_ms=" << compute_ms << '\n'
                  << "write_ms=" << write_ms << '\n'
                  << "total_ms=" << total_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace ultrahigh_ann::tools
