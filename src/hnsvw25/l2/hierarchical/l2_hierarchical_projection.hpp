#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann::detail {

struct L2HierarchicalProjection {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DoubleDenseMatrix jl_matrix;
    DoubleDenseMatrix projected_representatives;
    std::size_t sampled_multiplicity{};
};

struct L2HierarchicalProjectionFp32 {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix jl_matrix;
    DenseMatrix projected_representatives;
    std::size_t sampled_multiplicity{};
};

// Build the CPU hierarchy's sampled Rademacher projection in binary64.
[[nodiscard]] L2HierarchicalProjection build_l2_hierarchical_projection(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine);

// Reuse the same sampling and binary64 random draw as the CPU construction,
// then narrow the transform and project representatives with FP32 arithmetic.
[[nodiscard]] L2HierarchicalProjectionFp32
build_l2_hierarchical_projection_fp32(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine);

void project_l2_hierarchical_query(
    std::span<const float> query, std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    const DoubleDenseMatrix& jl_matrix, std::span<float> sampled_workspace,
    std::span<double> projected_output);

}  // namespace ultrahigh_ann::detail
