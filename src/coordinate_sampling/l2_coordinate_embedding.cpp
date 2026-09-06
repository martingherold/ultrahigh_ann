#include "coordinate_sampling/l2_coordinate_embedding.hpp"

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

[[nodiscard]] std::size_t checked_product(std::size_t left,
                                          std::size_t right,
                                          const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

}  // namespace

L2CoordinateEmbedding build_l2_coordinate_embedding(
    const DenseMatrix& input,
    CoordinateSample coordinate_sample)
{
    const std::size_t sampled_dimension = coordinate_sample.columns.size();
    const std::size_t transformed_count = checked_product(
        input.rows(), sampled_dimension, "sampled L2 representatives");
    std::vector<std::size_t> source_columns;
    std::vector<float> scales;
    source_columns.reserve(sampled_dimension);
    scales.reserve(sampled_dimension);

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
        const float scale = static_cast<float>(std::sqrt(weight));
        if (!std::isfinite(weight) || weight <= 0.0 ||
            !std::isfinite(scale) || scale <= 0.0F) {
            throw std::invalid_argument(
                "sampled coordinate weight is not representable in float32");
        }
        source_columns.push_back(column.source_column);
        scales.push_back(scale);
    }

    std::vector<float> transformed_values(transformed_count);
    std::size_t output_index = 0;
    for (std::size_t row = 0; row < input.rows(); ++row) {
        const auto representative = input.row(row);
        for (std::size_t column = 0; column < sampled_dimension; ++column) {
            transformed_values[output_index++] =
                representative[source_columns[column]] * scales[column];
        }
    }

    return L2CoordinateEmbedding{
        .original_dimension = input.cols(),
        .source_columns = std::move(source_columns),
        .scales = std::move(scales),
        .transformed_representatives = DenseMatrix(
            std::move(transformed_values), input.rows(), sampled_dimension),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

void pack_l2_coordinate_queries(std::span<const float> queries,
                                std::size_t query_count,
                                std::size_t original_dimension,
                                std::span<const std::size_t> source_columns,
                                std::span<const float> scales,
                                std::span<float> output)
{
    if (source_columns.size() != scales.size()) {
        throw std::logic_error("sampled L2 embedding dimensions do not match");
    }
    const std::size_t expected_query_values = checked_product(
        query_count, original_dimension, "sampled L2 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    const std::size_t sampled_dimension = source_columns.size();
    const std::size_t expected_output_values = checked_product(
        query_count, sampled_dimension, "sampled L2 packed query batch");
    if (output.size() != expected_output_values) {
        throw std::invalid_argument(
            "sampled L2 packed-query output has the wrong size");
    }

    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        const std::size_t query_offset = query_index * original_dimension;
        const std::size_t output_offset = query_index * sampled_dimension;
        for (std::size_t column = 0; column < sampled_dimension; ++column) {
            const float value = queries[query_offset + source_columns[column]];
            validate_finite_value(value, "sampled query coordinate");
            const float transformed = value * scales[column];
            validate_finite_value(transformed,
                                  "weighted sampled query coordinate");
            output[output_offset + column] = transformed;
        }
    }
}

std::size_t l2_embedding_metadata_bytes(
    const L2CoordinateEmbedding& embedding) noexcept
{
    return embedding.source_columns.size() * sizeof(std::size_t) +
           embedding.scales.size() * sizeof(float);
}

}  // namespace ultrahigh_ann::detail
