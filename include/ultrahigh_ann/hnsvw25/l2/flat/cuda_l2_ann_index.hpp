#pragma once

#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <string>

namespace ultrahigh_ann {

// Importance-probability facade for the CUDA sampled-coordinate backend.
class CudaFlatL2AnnIndex {
public:
    using QueryWorkspace = CudaSampledCoordinateL2AnnIndex::QueryWorkspace;

    CudaFlatL2AnnIndex(
        const DenseMatrix& input,
        std::size_t repetitions,
        std::mt19937_64& random_engine,
        int device = 0,
        ExecutionPolicy probability_execution_policy =
            ExecutionPolicy::gpu_fp32);

    CudaFlatL2AnnIndex(
        const DenseMatrix& input,
        std::span<const double> importance_probabilities,
        std::size_t repetitions,
        std::mt19937_64& random_engine,
        int device = 0);

    [[nodiscard]] QueryWorkspace make_query_workspace(
        std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t query(
        std::span<const float> query,
        QueryWorkspace& workspace,
        CudaSampledCoordinateL2QueryStrategy strategy =
            CudaSampledCoordinateL2QueryStrategy::direct) const;

    void query_batch(
        std::span<const float> queries,
        std::size_t query_count,
        std::span<std::size_t> output,
        QueryWorkspace& workspace,
        CudaSampledCoordinateL2QueryStrategy strategy =
            CudaSampledCoordinateL2QueryStrategy::direct) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

private:
    CudaSampledCoordinateL2AnnIndex index_;
};

}  // namespace ultrahigh_ann
