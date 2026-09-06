#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_l2_scan_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Probability-policy-independent coordinate-subsampling backend for L2.
// Squared distances suffice to rank representatives, so query() takes no
// square roots.
class SampledCoordinateL2AnnIndex {
public:
    SampledCoordinateL2AnnIndex(const DenseMatrix& input,
                                std::span<const double> probabilities,
                                std::size_t repetitions,
                                std::mt19937_64& random_engine);

    // Construct from an already sampled plan. This permits CPU and CUDA
    // backends to evaluate exactly the same retained coordinates.
    SampledCoordinateL2AnnIndex(const DenseMatrix& input,
                                CoordinateSample coordinate_sample);

    SampledCoordinateL2AnnIndex(const SampledCoordinateL2AnnIndex&) = delete;
    SampledCoordinateL2AnnIndex& operator=(const SampledCoordinateL2AnnIndex&) =
        delete;
    SampledCoordinateL2AnnIndex(SampledCoordinateL2AnnIndex&&) noexcept;
    SampledCoordinateL2AnnIndex& operator=(
        SampledCoordinateL2AnnIndex&&) noexcept;
    ~SampledCoordinateL2AnnIndex();

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
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace ultrahigh_ann
