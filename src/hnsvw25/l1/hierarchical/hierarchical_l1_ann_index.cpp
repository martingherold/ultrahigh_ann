#include "hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.hpp"

#include "core/dense_matrix.hpp"
#include "core/finite_values.hpp"
#include "coordinate_sampling/filter_columns.hpp"
#include "hnsvw25/l1/importance_sampling.hpp"

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
    std::mt19937_64& random_engine)
{
    validate_index_parameters(input, projection_dimension);
    return build_importance_sample(
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
    return build_importance_sample(
        probabilities,
        repetitions,
        random_engine);
}

[[nodiscard]] DoubleDenseMatrix build_cauchy_matrix(
    const CoordinateSample& importance_sample,
    std::size_t rows,
    std::mt19937_64& random_engine)
{
    const std::size_t columns =
        importance_sample.columns.size();
    const std::size_t maximum_size =
        std::vector<double>{}.max_size();
    if (columns != 0 && rows > maximum_size / columns) {
        throw std::length_error(
            "Cauchy matrix dimensions overflow");
    }

    std::vector<double> result_data;
    result_data.reserve(rows * columns);
    std::cauchy_distribution<double> cauchy(0.0, 1.0);

    for (std::size_t row = 0; row < rows; ++row) {
        for (const SampledColumn& sample :
             importance_sample.columns) {
            const double coefficient =
                cauchy(random_engine) *
                sample.inverse_probability *
                static_cast<double>(sample.multiplicity);

            detail::validate_finite_result(
                coefficient,
                "Cauchy projection coefficient");

            result_data.push_back(coefficient);
        }
    }

    return DoubleDenseMatrix(
        std::move(result_data),
        rows,
        columns);
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
        differences[index] =
        std::abs(first[index] - second[index]);
    }


    auto upper_middle =
        differences.begin() +
        static_cast<std::ptrdiff_t>(
            differences.size() / 2);

    std::ranges::nth_element(differences, upper_middle);

    if (differences.size() % 2 != 0) {
        return *upper_middle;
    }

    const auto lower_middle = std::ranges::max_element(differences.begin(), upper_middle);
    return (*lower_middle + *upper_middle) / 2.0;
}

void project_sampled_query(
    std::span<const float> sampled_query,
    const DoubleDenseMatrix& cauchy_matrix,
    std::span<double> output)
{
    if (sampled_query.size() != cauchy_matrix.cols()) {
        throw std::logic_error(
            "sampled query and Cauchy dimensions do not match");
    }
    if (output.size() != cauchy_matrix.rows()) {
        throw std::logic_error(
            "projected query dimensions do not match");
    }

    for (std::size_t projection = 0;
         projection < cauchy_matrix.rows();
         ++projection) {
        const auto coefficients =
            cauchy_matrix.row(projection);
        double sum = 0.0;

        for (std::size_t column = 0;
             column < sampled_query.size();
             ++column) {
            sum +=
                static_cast<double>(
                    sampled_query[column]) *
                coefficients[column];
        }

        output[projection] = sum;
    }
}

}  // namespace

HierarchicalL1AnnIndex::HierarchicalL1AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
    : HierarchicalL1AnnIndex(
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
    : importance_sample_(std::move(importance_sample)),
      initial_cols_(input.cols())
{
    const std::size_t rows = input.rows();

    const std::size_t columns =
        importance_sample_.columns.size();
    if (columns == 0 && rows > 1) {
        throw std::runtime_error(
            "importance sampling selected no coordinates");
    }

    DenseMatrix sampled_representatives =
        filter_columns(
            input,
            importance_sample_.columns);

    cauchy_matrix_ =
        build_cauchy_matrix(
            importance_sample_,
            projection_dimension,
            random_engine);

    reduced_representatives_ =
        DenseMatrix::multiply_right_transposed(
            sampled_representatives,
            cauchy_matrix_);

    detail::validate_finite_result(
        reduced_representatives_.values(),
        "projected representatives");
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

    const std::size_t columns{
        importance_sample_.columns.size()};
    workspace.sampled_values_.resize(columns);

    for (std::size_t index = 0;
         index < columns;
         ++index) {
        const SampledColumn& column =
            importance_sample_.columns[index];
        const float value =
            query[column.source_column];
        detail::validate_finite_value(
            value,
            "sampled query coordinate");
        workspace.sampled_values_[index] = value;
    }

    workspace.projected_query_.resize(
        cauchy_matrix_.rows());
    project_sampled_query(
        workspace.sampled_values_,
        cauchy_matrix_,
        workspace.projected_query_);
    detail::validate_finite_result(
        std::span<const double>{
            workspace.projected_query_},
        "projected query");

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
