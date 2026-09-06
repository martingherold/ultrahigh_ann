#pragma once

#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Importance-probability facade for the direct coordinate-sampling backend.
class FlatL2AnnIndex {
public:
    FlatL2AnnIndex(const DenseMatrix& input,
                   std::size_t repetitions,
                   std::mt19937_64& random_engine);

    FlatL2AnnIndex(const DenseMatrix& input,
                   std::span<const double> importance_probabilities,
                   std::size_t repetitions,
                   std::mt19937_64& random_engine);

    [[nodiscard]] std::size_t query(
        std::span<const float> query,
        CpuDenseL2QueryStrategy strategy =
            CpuDenseL2QueryStrategy::sequential) const;

    void query_batch(std::span<const float> queries,
                     std::size_t query_count,
                     std::span<std::size_t> output,
                     CpuDenseL2QueryStrategy strategy =
                         CpuDenseL2QueryStrategy::sequential) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    SampledCoordinateL2AnnIndex index_;
};

}  // namespace ultrahigh_ann
