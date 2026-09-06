#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l1_ann_index.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Importance-probability facade for the direct coordinate-sampling backend.
class FlatL1AnnIndex {
public:
    FlatL1AnnIndex(
        const DenseMatrix& input,
        std::size_t repetitions,
        std::mt19937_64& random_engine);

    FlatL1AnnIndex(
        const DenseMatrix& input,
        std::span<const double> importance_probabilities,
        std::size_t repetitions,
        std::mt19937_64& random_engine);

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    SampledCoordinateL1AnnIndex index_;
};

}  // namespace ultrahigh_ann
