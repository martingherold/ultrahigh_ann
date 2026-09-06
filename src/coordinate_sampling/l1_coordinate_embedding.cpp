#include "coordinate_sampling/l1_coordinate_embedding.hpp"

#include "coordinate_sampling/filter_columns.hpp"
#include "core/finite_values.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
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

} // namespace

L1CoordinateEmbedding
build_l1_coordinate_embedding(const DenseMatrix& input,
                              CoordinateSample coordinate_sample)
{
    std::vector<std::size_t> source_columns;
    std::vector<float> weights;
    source_columns.reserve(coordinate_sample.columns.size());
    weights.reserve(coordinate_sample.columns.size());

    std::size_t sampled_multiplicity = 0;
    for (const SampledColumn& column : coordinate_sample.columns) {
        if (column.source_column >= input.cols()) {
            throw std::out_of_range("sampled column index out of range");
        }
        if (column.multiplicity == 0 ||
            !std::isfinite(column.inverse_probability) ||
            column.inverse_probability <= 0.0) {
            throw std::invalid_argument(
                "sampled columns require positive finite weights");
        }
        if (sampled_multiplicity >
            std::numeric_limits<std::size_t>::max() - column.multiplicity) {
            throw std::length_error("sampled multiplicity overflows");
        }
        sampled_multiplicity += column.multiplicity;

        const double weight = static_cast<double>(column.multiplicity) *
                              column.inverse_probability;
        const float float_weight = static_cast<float>(weight);
        if (!std::isfinite(weight) || weight <= 0.0 ||
            !std::isfinite(float_weight) || float_weight <= 0.0F) {
            throw std::invalid_argument(
                "sampled coordinate weight is not representable in float32");
        }
        source_columns.push_back(column.source_column);
        weights.push_back(float_weight);
    }

    DenseMatrix sampled_representatives =
        filter_columns(input, coordinate_sample.columns);
    validate_finite_values(sampled_representatives.values(),
                           "sampled representatives");

    return L1CoordinateEmbedding{
        .original_dimension = input.cols(),
        .source_columns = std::move(source_columns),
        .weights = std::move(weights),
        .sampled_representatives = std::move(sampled_representatives),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

void pack_l1_coordinate_queries(std::span<const float> queries,
                                std::size_t query_count,
                                std::size_t original_dimension,
                                std::span<const std::size_t> source_columns,
                                std::span<float> output)
{
    const std::size_t expected_query_values = checked_product(
        query_count, original_dimension, "sampled L1 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    const std::size_t sampled_dimension = source_columns.size();
    const std::size_t expected_output_values = checked_product(
        query_count, sampled_dimension, "sampled L1 packed query batch");
    if (output.size() != expected_output_values) {
        throw std::invalid_argument(
            "sampled L1 packed-query output has the wrong size");
    }

    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        const std::size_t query_offset = query_index * original_dimension;
        const std::size_t output_offset = query_index * sampled_dimension;
        for (std::size_t column = 0; column < sampled_dimension; ++column) {
            const float value = queries[query_offset + source_columns[column]];
            validate_finite_value(value, "sampled query coordinate");
            output[output_offset + column] = value;
        }
    }
}

} // namespace ultrahigh_ann::detail
