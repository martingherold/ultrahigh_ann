#include "hnsvw25/l1/hierarchical/l1_hierarchical_projection.hpp"

#include "coordinate_sampling/filter_columns.hpp"
#include "core/finite_values.hpp"
#include "core/fp32_matrix_operations.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ultrahigh_ann::detail {
namespace {

[[nodiscard]] std::size_t checked_product(std::size_t left,
                                          std::size_t right,
                                          const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

[[nodiscard]] std::size_t validate_sample(
    const CoordinateSample& importance_sample,
    std::size_t original_dimension)
{
    std::size_t sampled_multiplicity = 0;
    for (const SampledColumn& sample : importance_sample.columns) {
        if (sample.source_column >= original_dimension) {
            throw std::out_of_range("sampled column index out of range");
        }
        if (sample.multiplicity == 0 ||
            !std::isfinite(sample.inverse_probability) ||
            sample.inverse_probability <= 0.0) {
            throw std::invalid_argument(
                "sampled columns require positive finite weights");
        }
        if (sampled_multiplicity >
            std::numeric_limits<std::size_t>::max() - sample.multiplicity) {
            throw std::length_error("sampled multiplicity overflows");
        }
        sampled_multiplicity += sample.multiplicity;
    }
    return sampled_multiplicity;
}

[[nodiscard]] DoubleDenseMatrix build_cauchy_matrix(
    const CoordinateSample& importance_sample,
    std::size_t rows,
    std::mt19937_64& random_engine)
{
    const std::size_t columns = importance_sample.columns.size();
    const std::size_t value_count =
        checked_product(rows, columns, "Cauchy matrix");
    std::vector<double> data;
    data.reserve(value_count);
    std::cauchy_distribution<double> cauchy(0.0, 1.0);

    for (std::size_t row = 0; row < rows; ++row) {
        for (const SampledColumn& sample : importance_sample.columns) {
            const double coefficient =
                cauchy(random_engine) * sample.inverse_probability *
                static_cast<double>(sample.multiplicity);
            validate_finite_result(coefficient,
                                   "Cauchy projection coefficient");
            data.push_back(coefficient);
        }
    }
    return DoubleDenseMatrix(std::move(data), rows, columns);
}

struct L1ProjectionBasis {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix sampled_representatives;
    DoubleDenseMatrix cauchy_matrix;
    std::size_t sampled_multiplicity{};
};

[[nodiscard]] L1ProjectionBasis prepare_l1_projection_basis(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "hierarchical L1 index expects at least one representative");
    }
    if (projection_dimension == 0) {
        throw std::invalid_argument(
            "projection_dimension has to be a positive integer");
    }
    if (importance_sample.columns.empty() && input.rows() > 1) {
        throw std::runtime_error(
            "importance sampling selected no coordinates");
    }

    const std::size_t sampled_multiplicity =
        validate_sample(importance_sample, input.cols());
    DenseMatrix sampled_representatives =
        filter_columns(input, importance_sample.columns);
    DoubleDenseMatrix cauchy_matrix = build_cauchy_matrix(
        importance_sample, projection_dimension, random_engine);
    return L1ProjectionBasis{
        .importance_sample = std::move(importance_sample),
        .original_dimension = input.cols(),
        .sampled_representatives = std::move(sampled_representatives),
        .cauchy_matrix = std::move(cauchy_matrix),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

}  // namespace

L1HierarchicalProjection build_l1_hierarchical_projection(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
{
    L1ProjectionBasis basis = prepare_l1_projection_basis(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    DoubleDenseMatrix projected_representatives =
        DenseMatrix::multiply_right_transposed(basis.sampled_representatives,
                                               basis.cauchy_matrix);
    validate_finite_result(projected_representatives.values(),
                           "projected representatives");

    return L1HierarchicalProjection{
        .importance_sample = std::move(basis.importance_sample),
        .original_dimension = basis.original_dimension,
        .cauchy_matrix = std::move(basis.cauchy_matrix),
        .projected_representatives = std::move(projected_representatives),
        .sampled_multiplicity = basis.sampled_multiplicity,
    };
}

L1HierarchicalProjectionFp32 build_l1_hierarchical_projection_fp32(
    const DenseMatrix& input,
    CoordinateSample importance_sample,
    std::size_t projection_dimension,
    std::mt19937_64& random_engine)
{
    L1ProjectionBasis basis = prepare_l1_projection_basis(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    DenseMatrix cauchy_matrix = narrow_matrix_to_fp32(
        basis.cauchy_matrix, "FP32 Cauchy projection coefficient");
    DenseMatrix projected_representatives = multiply_right_transposed_fp32(
        basis.sampled_representatives, cauchy_matrix,
        "FP32 Cauchy projected representative");

    return L1HierarchicalProjectionFp32{
        .importance_sample = std::move(basis.importance_sample),
        .original_dimension = basis.original_dimension,
        .cauchy_matrix = std::move(cauchy_matrix),
        .projected_representatives = std::move(projected_representatives),
        .sampled_multiplicity = basis.sampled_multiplicity,
    };
}

void project_l1_hierarchical_query(
    std::span<const float> query,
    std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    const DoubleDenseMatrix& cauchy_matrix,
    std::span<float> sampled_workspace,
    std::span<double> projected_output)
{
    if (query.size() != original_dimension) {
        throw std::invalid_argument(
            "query dimension does not match index dimension");
    }
    if (sampled_workspace.size() != sampled_columns.size() ||
        cauchy_matrix.cols() != sampled_columns.size()) {
        throw std::logic_error(
            "sampled query and Cauchy dimensions do not match");
    }
    if (projected_output.size() != cauchy_matrix.rows()) {
        throw std::logic_error("projected query dimensions do not match");
    }

    for (std::size_t index = 0; index < sampled_columns.size(); ++index) {
        const float value = query[sampled_columns[index].source_column];
        validate_finite_value(value, "sampled query coordinate");
        sampled_workspace[index] = value;
    }

    for (std::size_t projection = 0; projection < cauchy_matrix.rows();
         ++projection) {
        const auto coefficients = cauchy_matrix.row(projection);
        double sum = 0.0;
        for (std::size_t column = 0; column < sampled_workspace.size();
             ++column) {
            sum += static_cast<double>(sampled_workspace[column]) *
                   coefficients[column];
        }
        validate_finite_result(sum, "projected query");
        projected_output[projection] = sum;
    }
}

void pack_l1_hierarchical_sampled_queries(
    std::span<const float> queries,
    std::size_t query_count,
    std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    std::span<float> output)
{
    const std::size_t expected_query_values = checked_product(
        query_count, original_dimension, "hierarchical L1 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    const std::size_t expected_output_values = checked_product(
        query_count, sampled_columns.size(),
        "hierarchical L1 sampled query batch");
    if (output.size() != expected_output_values) {
        throw std::invalid_argument(
            "hierarchical L1 sampled-query output has the wrong size");
    }

    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        const std::size_t query_offset = query_index * original_dimension;
        const std::size_t output_offset =
            query_index * sampled_columns.size();
        for (std::size_t sampled_index = 0;
             sampled_index < sampled_columns.size(); ++sampled_index) {
            const float value = queries[
                query_offset + sampled_columns[sampled_index].source_column];
            validate_finite_value(value, "sampled query coordinate");
            output[output_offset + sampled_index] = value;
        }
    }
}

}  // namespace ultrahigh_ann::detail
