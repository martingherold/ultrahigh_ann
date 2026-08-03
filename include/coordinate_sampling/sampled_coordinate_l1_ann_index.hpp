#pragma once

#include "core/dense_matrix.hpp"
#include "core/index_space_usage.hpp"
#include "coordinate_sampling/coordinate_sample.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Probability-policy-independent coordinate-subsampling backend for L1.
class SampledCoordinateL1AnnIndex {
public:
    SampledCoordinateL1AnnIndex(
        const DenseMatrix& input,
        std::span<const double> probabilities,
        std::size_t repetitions,
        std::mt19937_64& random_engine);

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    SampledCoordinateL1AnnIndex(
        const DenseMatrix& input,
        CoordinateSample coordinate_sample);

    CoordinateSample coordinate_sample_;
    DenseMatrix sampled_representatives_;
    std::size_t initial_cols_{};
};

}  // namespace ultrahigh_ann
