#pragma once

#include "core/dense_matrix.hpp"
#include "core/index_space_usage.hpp"
#include "coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Uniform, mass-matched coordinate sampler with direct squared-L2 scans.
// It deliberately performs no Johnson-Lindenstrauss projection.
class UniformL2AnnIndex {
public:
    UniformL2AnnIndex(
        const DenseMatrix& input,
        double sampling_mass,
        std::size_t repetitions,
        std::mt19937_64& random_engine);

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    SampledCoordinateL2AnnIndex index_;
};

}  // namespace ultrahigh_ann
