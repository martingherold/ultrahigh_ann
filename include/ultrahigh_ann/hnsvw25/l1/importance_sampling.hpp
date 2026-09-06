#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/hnsvw25/execution_policy.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace ultrahigh_ann {

enum class L1GpuDistanceBackend {
    // L1 distance requires an elementwise absolute-difference reduction and
    // therefore has no equivalent cuBLAS matrix-multiplication backend.
    direct,
};

struct L1GpuProbabilityTimings {
    double host_to_device_ms{};
    double inverse_distance_ms{};
    double coordinate_maximum_ms{};
    double device_to_host_ms{};
    double total_ms{};
};

struct L1GpuProbabilityResult {
    std::vector<double> probabilities;
    std::string device_name;
    int device{};
    L1GpuDistanceBackend distance_backend{L1GpuDistanceBackend::direct};
    std::size_t pair_count{};
    std::size_t pair_chunks{};
    std::size_t device_working_set_bytes{};
    L1GpuProbabilityTimings timings;
};

// True when the library was built with CUDA and at least one usable CUDA
// device is visible to the process.
[[nodiscard]] bool l1_gpu_backend_available() noexcept;

// Compute on one CUDA device. pair_chunks bounds coordinate-maximum kernel
// duration and controls the partial-maximum buffer; it must be positive. All
// CUDA arithmetic is FP32. L1 exposes only the direct distance backend.
[[nodiscard]] L1GpuProbabilityResult compute_l1_importance_probabilities_gpu(
    const DenseMatrix& representatives, int device = 0,
    std::size_t pair_chunks = 128,
    L1GpuDistanceBackend distance_backend = L1GpuDistanceBackend::direct);

// Compute once for a fixed representative matrix and reuse for every L1
// repetition count, seed, and projection dimension.
[[nodiscard]] std::vector<double>
compute_l1_importance_probabilities(const DenseMatrix& representatives);

[[nodiscard]] std::vector<double>
compute_l1_importance_probabilities(const DenseMatrix& representatives,
                                    ExecutionPolicy execution_policy);

[[nodiscard]] CoordinateSample
build_l1_importance_sample(std::span<const double> probabilities,
                           std::size_t repetitions,
                           std::mt19937_64& random_engine);

[[nodiscard]] CoordinateSample build_l1_importance_sample(
    const DenseMatrix& representatives, std::size_t repetitions,
    std::mt19937_64& random_engine,
    ExecutionPolicy execution_policy = ExecutionPolicy::sequential);

// Compatibility spellings retained for existing callers.
[[nodiscard]] CoordinateSample
build_importance_sample(std::span<const double> probabilities,
                        std::size_t repetitions,
                        std::mt19937_64& random_engine);

[[nodiscard]] CoordinateSample
build_importance_sample(const DenseMatrix& representatives,
                        std::size_t repetitions,
                        std::mt19937_64& random_engine);

[[nodiscard]] CoordinateSample
build_importance_sample(const DenseMatrix& representatives,
                        std::size_t repetitions, std::mt19937_64& random_engine,
                        ExecutionPolicy execution_policy);

}  // namespace ultrahigh_ann
