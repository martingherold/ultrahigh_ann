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

enum class L2GpuDistanceBackend {
    // Compute every pair distance directly with a CUDA warp.
    direct,
    // Form an FP32 representative Gram matrix with cuBLAS, then directly
    // refine pair distances susceptible to cancellation.
    cublas,
};

struct L2GpuProbabilityTimings {
    double host_to_device_ms{};
    double inverse_distance_ms{};
    double coordinate_maximum_ms{};
    double device_to_host_ms{};
    double total_ms{};
};

struct L2GpuProbabilityResult {
    std::vector<double> probabilities;
    std::string device_name;
    int device{};
    L2GpuDistanceBackend distance_backend{L2GpuDistanceBackend::direct};
    std::size_t pair_count{};
    std::size_t pair_chunks{};
    std::size_t device_working_set_bytes{};
    L2GpuProbabilityTimings timings;
};

// True when the library was built with CUDA and at least one usable CUDA
// device is visible to the process.
[[nodiscard]] bool l2_gpu_backend_available() noexcept;

// Compute on one CUDA device. pair_chunks bounds coordinate-maximum kernel
// duration and controls the partial-maximum buffer; it must be positive. All
// CUDA arithmetic is FP32. The cuBLAS backend additionally retains a full
// representative_count by representative_count FP32 Gram matrix.
[[nodiscard]] L2GpuProbabilityResult compute_l2_importance_probabilities_gpu(
    const DenseMatrix& representatives, int device = 0,
    std::size_t pair_chunks = 128,
    L2GpuDistanceBackend distance_backend = L2GpuDistanceBackend::direct);

// Compute once for a fixed representative matrix and reuse for every L2
// repetition count, seed, and projection dimension.
[[nodiscard]] std::vector<double> compute_l2_importance_probabilities(
    const DenseMatrix& representatives,
    ExecutionPolicy execution_policy = ExecutionPolicy::sequential);

[[nodiscard]] CoordinateSample
build_l2_importance_sample(std::span<const double> probabilities,
                           std::size_t repetitions,
                           std::mt19937_64& random_engine);

[[nodiscard]] CoordinateSample build_l2_importance_sample(
    const DenseMatrix& representatives, std::size_t repetitions,
    std::mt19937_64& random_engine,
    ExecutionPolicy execution_policy = ExecutionPolicy::sequential);

}  // namespace ultrahigh_ann
