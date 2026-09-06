#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"

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

    // Construct from an already sampled plan. This permits CPU and CUDA
    // backends to evaluate exactly the same retained coordinates.
    SampledCoordinateL1AnnIndex(
        const DenseMatrix& input,
        CoordinateSample coordinate_sample);

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    CoordinateSample coordinate_sample_;
    DenseMatrix sampled_representatives_;
    std::size_t initial_cols_{};
};

}  // namespace ultrahigh_ann
