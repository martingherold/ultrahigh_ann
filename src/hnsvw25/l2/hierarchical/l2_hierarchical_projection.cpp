#include "hnsvw25/l2/hierarchical/l2_hierarchical_projection.hpp"

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

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

[[nodiscard]] std::size_t
validate_sample(const CoordinateSample& importance_sample,
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

[[nodiscard]] DoubleDenseMatrix
build_jl_matrix(const CoordinateSample& importance_sample, std::size_t rows,
                std::mt19937_64& random_engine)
{
    const std::size_t columns = importance_sample.columns.size();
    const std::size_t value_count = checked_product(rows, columns, "JL matrix");
    const double row_scaling = 1.0 / std::sqrt(static_cast<double>(rows));
    std::vector<double> data;
    data.reserve(value_count);
    for (std::size_t row = 0; row < rows; ++row) {
        for (const SampledColumn& sample : importance_sample.columns) {
            std::binomial_distribution<std::size_t> positive_signs(
                sample.multiplicity, 0.5);
            const std::size_t positives = positive_signs(random_engine);
            const double signed_sum = 2.0 * static_cast<double>(positives) -
                                      static_cast<double>(sample.multiplicity);
            const double coefficient = signed_sum *
                                       std::sqrt(sample.inverse_probability) *
                                       row_scaling;
            validate_finite_result(coefficient, "JL projection coefficient");
            data.push_back(coefficient);
        }
    }
    return DoubleDenseMatrix(std::move(data), rows, columns);
}

struct L2ProjectionBasis {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix sampled_representatives;
    DoubleDenseMatrix jl_matrix;
    std::size_t sampled_multiplicity{};
};

[[nodiscard]] L2ProjectionBasis prepare_l2_projection_basis(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine)
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "hierarchical L2 index expects at least one representative");
    }
    if (projection_dimension == 0) {
        throw std::invalid_argument(
            "projection_dimension has to be a positive integer");
    }
    if (importance_sample.columns.empty() && input.rows() > 1) {
        throw std::runtime_error("importance sampling selected no coordinates");
    }

    const std::size_t sampled_multiplicity =
        validate_sample(importance_sample, input.cols());
    DenseMatrix sampled_representatives =
        filter_columns(input, importance_sample.columns);
    DoubleDenseMatrix jl_matrix =
        build_jl_matrix(importance_sample, projection_dimension, random_engine);
    return L2ProjectionBasis{
        .importance_sample = std::move(importance_sample),
        .original_dimension = input.cols(),
        .sampled_representatives = std::move(sampled_representatives),
        .jl_matrix = std::move(jl_matrix),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

}  // namespace

L2HierarchicalProjection build_l2_hierarchical_projection(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine)
{
    L2ProjectionBasis basis = prepare_l2_projection_basis(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    DoubleDenseMatrix projected_representatives =
        DenseMatrix::multiply_right_transposed(basis.sampled_representatives,
                                               basis.jl_matrix);
    validate_finite_result(projected_representatives.values(),
                           "projected representatives");

    return L2HierarchicalProjection{
        .importance_sample = std::move(basis.importance_sample),
        .original_dimension = basis.original_dimension,
        .jl_matrix = std::move(basis.jl_matrix),
        .projected_representatives = std::move(projected_representatives),
        .sampled_multiplicity = basis.sampled_multiplicity,
    };
}

L2HierarchicalProjectionFp32 build_l2_hierarchical_projection_fp32(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine)
{
    L2ProjectionBasis basis = prepare_l2_projection_basis(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    DenseMatrix jl_matrix = narrow_matrix_to_fp32(
        basis.jl_matrix, "FP32 JL projection coefficient");
    DenseMatrix projected_representatives = multiply_right_transposed_fp32(
        basis.sampled_representatives, jl_matrix,
        "FP32 JL projected representative");

    return L2HierarchicalProjectionFp32{
        .importance_sample = std::move(basis.importance_sample),
        .original_dimension = basis.original_dimension,
        .jl_matrix = std::move(jl_matrix),
        .projected_representatives = std::move(projected_representatives),
        .sampled_multiplicity = basis.sampled_multiplicity,
    };
}

void project_l2_hierarchical_query(
    std::span<const float> query, std::size_t original_dimension,
    std::span<const SampledColumn> sampled_columns,
    const DoubleDenseMatrix& jl_matrix, std::span<float> sampled_workspace,
    std::span<double> projected_output)
{
    if (query.size() != original_dimension) {
        throw std::invalid_argument(
            "query dimension does not match index dimension");
    }
    if (sampled_workspace.size() != sampled_columns.size() ||
        jl_matrix.cols() != sampled_columns.size()) {
        throw std::logic_error("sampled query and JL dimensions do not match");
    }
    if (projected_output.size() != jl_matrix.rows()) {
        throw std::logic_error("projected query dimensions do not match");
    }

    for (std::size_t index = 0; index < sampled_columns.size(); ++index) {
        const float value = query[sampled_columns[index].source_column];
        validate_finite_value(value, "sampled query coordinate");
        sampled_workspace[index] = value;
    }

    for (std::size_t projection = 0; projection < jl_matrix.rows();
         ++projection) {
        const auto coefficients = jl_matrix.row(projection);
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

}  // namespace ultrahigh_ann::detail
