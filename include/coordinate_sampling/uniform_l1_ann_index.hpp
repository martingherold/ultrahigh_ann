#pragma once

#include "core/dense_matrix.hpp"
#include "core/index_space_usage.hpp"
#include "coordinate_sampling/sampled_coordinate_l1_ann_index.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Uniform, mass-matched coordinate sampler with direct L1 distance scans.
// It deliberately performs no Cauchy projection.
class UniformL1AnnIndex {
public:
    UniformL1AnnIndex(
        const DenseMatrix& input,
        double sampling_mass,
        std::size_t repetitions,
        std::mt19937_64& random_engine);

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    SampledCoordinateL1AnnIndex index_;
};

}  // namespace ultrahigh_ann
