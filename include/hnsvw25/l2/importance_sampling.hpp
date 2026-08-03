#pragma once

#include "core/dense_matrix.hpp"
#include "coordinate_sampling/coordinate_sample.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace ultrahigh_ann {

// Compute once for a fixed representative matrix and reuse for every L2
// repetition count, seed, and projection dimension.
[[nodiscard]] std::vector<double> compute_l2_importance_probabilities(
    const DenseMatrix& representatives);

[[nodiscard]] CoordinateSample build_l2_importance_sample(
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine);

[[nodiscard]] CoordinateSample build_l2_importance_sample(
    const DenseMatrix& representatives,
    std::size_t repetitions,
    std::mt19937_64& random_engine);

}  // namespace ultrahigh_ann
