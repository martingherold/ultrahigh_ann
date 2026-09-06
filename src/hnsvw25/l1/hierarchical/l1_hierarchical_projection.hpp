#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann::detail {

struct L1HierarchicalProjection {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DoubleDenseMatrix cauchy_matrix;
    DoubleDenseMatrix projected_representatives;
    std::size_t sampled_multiplicity{};
};

struct L1HierarchicalProjectionFp32 {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix cauchy_matrix;
    DenseMatrix projected_representatives;
    std::size_t sampled_multiplicity{};
};

// Construct the CPU hierarchy's sampled Cauchy projection in binary64.
[[nodiscard]] L1HierarchicalProjection build_l1_hierarchical_projection(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine);

// Reuse the same sampling and binary64 random draw as the CPU construction,
// then narrow the transform and project representatives with FP32 arithmetic.
[[nodiscard]] L1HierarchicalProjectionFp32
build_l1_hierarchical_projection_fp32(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine);

void project_l1_hierarchical_query(
    std::span<const float> query,
    std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    const DoubleDenseMatrix& cauchy_matrix,
    std::span<float> sampled_workspace,
    std::span<double> projected_output);

// Gather a row-major batch into the compact sampled representation transferred
// by the CUDA hierarchy. Only sampled coordinates are validated.
void pack_l1_hierarchical_sampled_queries(
    std::span<const float> queries,
    std::size_t query_count,
    std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    std::span<float> output);

}  // namespace ultrahigh_ann::detail
