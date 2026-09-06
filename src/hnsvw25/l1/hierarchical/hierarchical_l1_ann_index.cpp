#include "ultrahigh_ann/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.hpp"

#include "hnsvw25/l1/hierarchical/l1_hierarchical_projection.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <ranges>


namespace ultrahigh_ann {

namespace {

void validate_index_parameters(
    const DenseMatrix& input,
    std::size_t projection_dimension)
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "HierarchicalL1AnnIndex expects at least one representative");
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
    return build_l1_importance_sample(input, repetitions, random_engine,
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
    return build_importance_sample(
        probabilities,
        repetitions,
        random_engine);
}

[[nodiscard]] double median_absolute_difference(
    std::span<const double> first,
    std::span<const double> second,
    std::span<double> differences)
{
    if (first.size() != second.size()) {
        throw std::logic_error(
            "projected dimensions do not match");
    }
    if (differences.size() != first.size()) {
        throw std::logic_error(
            "median scratch dimensions do not match");
    }
    if (first.empty()) {
        throw std::logic_error(
            "cannot compute the median of an empty projection");
    }

    for (std::size_t index = 0; index < first.size(); ++index) {
        differences[index] = std::abs(first[index] - second[index]);
    }

    auto upper_middle = differences.begin() +
                        static_cast<std::ptrdiff_t>(differences.size() / 2);

    std::ranges::nth_element(differences, upper_middle);

    if (differences.size() % 2 != 0) {
        return *upper_middle;
    }

    const auto lower_middle =
        std::ranges::max_element(differences.begin(), upper_middle);
    return *lower_middle + (*upper_middle - *lower_middle) * 0.5;
}

}  // namespace

HierarchicalL1AnnIndex::HierarchicalL1AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : HierarchicalL1AnnIndex(input, repetitions, projection_dimension,
                             random_engine, ExecutionPolicy::sequential)
{
}

HierarchicalL1AnnIndex::HierarchicalL1AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine,
    ExecutionPolicy execution_policy)
    : HierarchicalL1AnnIndex(
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

HierarchicalL1AnnIndex::HierarchicalL1AnnIndex(
    const DenseMatrix& input,
    std::span<const double> importance_probabilities,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : HierarchicalL1AnnIndex(
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

HierarchicalL1AnnIndex::HierarchicalL1AnnIndex(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : initial_cols_(input.cols())
{
    auto projection = detail::build_l1_hierarchical_projection(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    importance_sample_ = std::move(projection.importance_sample);
    initial_cols_ = projection.original_dimension;
    cauchy_matrix_ = std::move(projection.cauchy_matrix);
    reduced_representatives_ =
        std::move(projection.projected_representatives);
}

HierarchicalL1AnnIndex::QueryWorkspace
HierarchicalL1AnnIndex::make_query_workspace() const
{
    QueryWorkspace workspace;
    workspace.sampled_values_.resize(
        importance_sample_.columns.size());
    workspace.projected_query_.resize(
        cauchy_matrix_.rows());
    workspace.median_scratch_.resize(
        cauchy_matrix_.rows());
    return workspace;
}

std::size_t HierarchicalL1AnnIndex::query(
    std::span<const float> query) const
{
    QueryWorkspace workspace =
        make_query_workspace();
    return this->query(query, workspace);
}

std::size_t HierarchicalL1AnnIndex::query(
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
    workspace.projected_query_.resize(
        cauchy_matrix_.rows());
    detail::project_l1_hierarchical_query(
        query, initial_cols_, importance_sample_.columns, cauchy_matrix_,
        workspace.sampled_values_, workspace.projected_query_);

    const std::span<const double> projected_query{
        workspace.projected_query_};
    workspace.median_scratch_.resize(
        projected_query.size());

    std::size_t best_index = 0;
    double minimum_distance =
        median_absolute_difference(
            reduced_representatives_.row(0),
            projected_query,
            workspace.median_scratch_);

    for (std::size_t row = 1;
         row < reduced_representatives_.rows();
         ++row) {
        const double distance =
            median_absolute_difference(
                reduced_representatives_.row(row),
                projected_query,
                workspace.median_scratch_);
        if (distance < minimum_distance) {
            minimum_distance = distance;
            best_index = row;
        }
    }

    return best_index;
}

IndexSpaceUsage HierarchicalL1AnnIndex::space_usage() const noexcept
{
    std::size_t sampled_multiplicity = 0;
    for (const SampledColumn& column : importance_sample_.columns) {
        sampled_multiplicity += column.multiplicity;
    }

    return IndexSpaceUsage{
        .index_payload_bytes =
            importance_sample_.columns.size() * sizeof(SampledColumn) +
            cauchy_matrix_.values().size_bytes() +
            reduced_representatives_.values().size_bytes(),
        .query_workspace_payload_bytes =
            importance_sample_.columns.size() * sizeof(float) +
            2 * cauchy_matrix_.rows() * sizeof(double),
        .unique_query_coordinates = importance_sample_.columns.size(),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

}  // namespace ultrahigh_ann
