#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <vector>

namespace ultrahigh_ann {

// Paper-faithful sampled L2 index followed by a Rademacher
// Johnson-Lindenstrauss projection.
class HierarchicalL2AnnIndex {
public:
    class QueryWorkspace {
    public:
        QueryWorkspace() = default;

    private:
        friend class HierarchicalL2AnnIndex;
        std::vector<float> sampled_values_;
        std::vector<double> projected_query_;
    };

    HierarchicalL2AnnIndex(
        const DenseMatrix& input,
        std::size_t repetitions,
        std::size_t projection_dimension,
        std::mt19937_64& random_engine,
        ExecutionPolicy execution_policy = ExecutionPolicy::sequential);

    HierarchicalL2AnnIndex(
        const DenseMatrix& input,
        std::span<const double> importance_probabilities,
        std::size_t repetitions,
        std::size_t projection_dimension,
        std::mt19937_64& random_engine);

    [[nodiscard]] QueryWorkspace make_query_workspace() const;

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] std::size_t query(
        std::span<const float> query,
        QueryWorkspace& workspace) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    HierarchicalL2AnnIndex(
        const DenseMatrix& input,
        CoordinateSample importance_sample,
        std::size_t projection_dimension,
        std::mt19937_64& random_engine);

    CoordinateSample importance_sample_;
    std::size_t initial_cols_{};
    DoubleDenseMatrix jl_matrix_;
    DoubleDenseMatrix reduced_representatives_;
};

}  // namespace ultrahigh_ann
