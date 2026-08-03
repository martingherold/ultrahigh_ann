#include "hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.hpp"

#include "core/finite_values.hpp"
#include "coordinate_sampling/filter_columns.hpp"
#include "hnsvw25/l2/importance_sampling.hpp"

#include <cmath>
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
    std::mt19937_64& random_engine)
{
    validate_index_parameters(input, projection_dimension);
    return build_l2_importance_sample(
        input,
        repetitions,
        random_engine);
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

[[nodiscard]] DoubleDenseMatrix build_jl_matrix(
    const CoordinateSample& importance_sample,
    std::size_t rows,
    std::mt19937_64& random_engine)
{
    const std::size_t columns = importance_sample.columns.size();
    const std::size_t maximum_size = std::vector<double>{}.max_size();
    if (columns != 0 && rows > maximum_size / columns) {
        throw std::length_error("JL matrix dimensions overflow");
    }

    const double row_scaling = 1.0 / std::sqrt(static_cast<double>(rows));
    std::vector<double> data;
    data.reserve(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (const SampledColumn& sample : importance_sample.columns) {
            std::binomial_distribution<std::size_t> positive_signs(
                sample.multiplicity,
                0.5);
            const std::size_t positives = positive_signs(random_engine);
            const double signed_sum =
                2.0 * static_cast<double>(positives) -
                static_cast<double>(sample.multiplicity);
            const double coefficient =
                signed_sum *
                std::sqrt(sample.inverse_probability) *
                row_scaling;
            detail::validate_finite_result(
                coefficient,
                "JL projection coefficient");
            data.push_back(coefficient);
        }
    }
    return DoubleDenseMatrix(std::move(data), rows, columns);
}

void project_sampled_query(
    std::span<const float> sampled_query,
    const DoubleDenseMatrix& matrix,
    std::span<double> output)
{
    if (sampled_query.size() != matrix.cols()) {
        throw std::logic_error(
            "sampled query and JL dimensions do not match");
    }
    if (output.size() != matrix.rows()) {
        throw std::logic_error(
            "projected query dimensions do not match");
    }
    for (std::size_t projection = 0; projection < matrix.rows(); ++projection) {
        const auto coefficients = matrix.row(projection);
        double sum = 0.0;
        for (std::size_t column = 0; column < sampled_query.size(); ++column) {
            sum += static_cast<double>(sampled_query[column]) *
                   coefficients[column];
        }
        output[projection] = sum;
    }
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
    std::mt19937_64& random_engine)
    : HierarchicalL2AnnIndex(
          input,
          build_importance_sample_checked(
              input,
              repetitions,
              projection_dimension,
              random_engine),
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
    : importance_sample_(std::move(importance_sample)),
      initial_cols_(input.cols())
{
    if (importance_sample_.columns.empty() && input.rows() > 1) {
        throw std::runtime_error(
            "importance sampling selected no coordinates");
    }
    const DenseMatrix sampled_representatives = filter_columns(
        input,
        importance_sample_.columns);
    jl_matrix_ = build_jl_matrix(
        importance_sample_,
        projection_dimension,
        random_engine);
    reduced_representatives_ = DenseMatrix::multiply_right_transposed(
        sampled_representatives,
        jl_matrix_);
    detail::validate_finite_result(
        reduced_representatives_.values(),
        "projected representatives");
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
    for (std::size_t index = 0;
         index < importance_sample_.columns.size();
         ++index) {
        const SampledColumn& column = importance_sample_.columns[index];
        const float value = query[column.source_column];
        detail::validate_finite_value(value, "sampled query coordinate");
        workspace.sampled_values_[index] = value;
    }
    workspace.projected_query_.resize(jl_matrix_.rows());
    project_sampled_query(
        workspace.sampled_values_,
        jl_matrix_,
        workspace.projected_query_);
    detail::validate_finite_result(
        std::span<const double>{workspace.projected_query_},
        "projected query");

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
