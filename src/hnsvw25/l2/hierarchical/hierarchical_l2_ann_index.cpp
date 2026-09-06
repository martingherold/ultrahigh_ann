#include "ultrahigh_ann/hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.hpp"

#include "hnsvw25/l2/hierarchical/l2_hierarchical_projection.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ultrahigh_ann {
namespace {

void validate_index_parameters(
    const DenseMatrix& input,
    std::size_t projection_dimension)
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "HierarchicalL2AnnIndex expects at least one representative");
    }
    if (projection_dimension == 0) {
        throw std::invalid_argument(
            "projection_dimension has to be a positive integer");
    }
}

[[nodiscard]] CoordinateSample build_importance_sample_checked(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine,
    ExecutionPolicy execution_policy)
{
    validate_index_parameters(input, projection_dimension);
    return build_l2_importance_sample(
        input,
        repetitions,
        random_engine,
        execution_policy);
}

[[nodiscard]] CoordinateSample build_importance_sample_checked(
    const DenseMatrix& input,
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
{
    validate_index_parameters(input, projection_dimension);
    if (probabilities.size() != input.cols()) {
        throw std::invalid_argument(
            "importance probability dimension does not match input");
    }
    return build_l2_importance_sample(
        probabilities,
        repetitions,
        random_engine);
}

[[nodiscard]] double squared_l2_dist(
    std::span<const double> first,
    std::span<const double> second)
{
    if (first.size() != second.size()) {
        throw std::logic_error(
            "projected dimensions do not match");
    }
    double distance = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        const double difference = first[index] - second[index];
        distance += difference * difference;
    }
    return distance;
}

}  // namespace

HierarchicalL2AnnIndex::HierarchicalL2AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine,
    ExecutionPolicy execution_policy)
    : HierarchicalL2AnnIndex(
          input,
          build_importance_sample_checked(
              input,
              repetitions,
              projection_dimension,
              random_engine,
              execution_policy),
          projection_dimension,
          random_engine)
{
}

HierarchicalL2AnnIndex::HierarchicalL2AnnIndex(
    const DenseMatrix& input,
    std::span<const double> importance_probabilities,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : HierarchicalL2AnnIndex(
          input,
          build_importance_sample_checked(
              input,
              importance_probabilities,
              repetitions,
              projection_dimension,
              random_engine),
          projection_dimension,
          random_engine)
{
}

HierarchicalL2AnnIndex::HierarchicalL2AnnIndex(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : initial_cols_(input.cols())
{
    auto projection = detail::build_l2_hierarchical_projection(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    importance_sample_ = std::move(projection.importance_sample);
    initial_cols_ = projection.original_dimension;
    jl_matrix_ = std::move(projection.jl_matrix);
    reduced_representatives_ =
        std::move(projection.projected_representatives);
}

HierarchicalL2AnnIndex::QueryWorkspace
HierarchicalL2AnnIndex::make_query_workspace() const
{
    QueryWorkspace workspace;
    workspace.sampled_values_.resize(importance_sample_.columns.size());
    workspace.projected_query_.resize(jl_matrix_.rows());
    return workspace;
}

std::size_t HierarchicalL2AnnIndex::query(
    std::span<const float> query) const
{
    QueryWorkspace workspace = make_query_workspace();
    return this->query(query, workspace);
}

std::size_t HierarchicalL2AnnIndex::query(
    std::span<const float> query,
    QueryWorkspace& workspace) const
{
    if (query.size() != initial_cols_) {
        throw std::invalid_argument(
            "query dimension does not match index dimension");
    }
    if (reduced_representatives_.rows() == 1) {
        return 0;
    }

    workspace.sampled_values_.resize(importance_sample_.columns.size());
    workspace.projected_query_.resize(jl_matrix_.rows());
    detail::project_l2_hierarchical_query(
        query, initial_cols_, importance_sample_.columns, jl_matrix_,
        workspace.sampled_values_, workspace.projected_query_);

    const std::span<const double> projected_query{
        workspace.projected_query_};
    std::size_t best_index = 0;
    double minimum_distance = squared_l2_dist(
        reduced_representatives_.row(0),
        projected_query);
    for (std::size_t row = 1;
         row < reduced_representatives_.rows();
         ++row) {
        const double distance = squared_l2_dist(
            reduced_representatives_.row(row),
            projected_query);
        if (distance < minimum_distance) {
            minimum_distance = distance;
            best_index = row;
        }
    }
    return best_index;
}

IndexSpaceUsage HierarchicalL2AnnIndex::space_usage() const noexcept
{
    std::size_t sampled_multiplicity = 0;
    for (const SampledColumn& column : importance_sample_.columns) {
        sampled_multiplicity += column.multiplicity;
    }
    return IndexSpaceUsage{
        .index_payload_bytes =
            importance_sample_.columns.size() * sizeof(SampledColumn) +
            jl_matrix_.values().size_bytes() +
            reduced_representatives_.values().size_bytes(),
        .query_workspace_payload_bytes =
            importance_sample_.columns.size() * sizeof(float) +
            jl_matrix_.rows() * sizeof(double),
        .unique_query_coordinates = importance_sample_.columns.size(),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

}  // namespace ultrahigh_ann
