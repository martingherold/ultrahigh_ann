#include "coordinate_sampling/sampled_coordinate_l1_ann_index.hpp"

#include "core/finite_values.hpp"
#include "coordinate_sampling/coordinate_sampling.hpp"
#include "coordinate_sampling/filter_columns.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>

namespace ultrahigh_ann {
namespace {

[[nodiscard]] CoordinateSample build_coordinate_sample_checked(
    const DenseMatrix& input,
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    if (probabilities.size() != input.cols()) {
        throw std::invalid_argument(
            "sampling probability dimension does not match input");
    }
    return build_coordinate_sample(
        probabilities,
        repetitions,
        random_engine);
}

[[nodiscard]] double weighted_l1_distance(
    std::span<const float> query,
    std::span<const float> sampled_representative,
    std::span<const SampledColumn> columns)
{
    if (sampled_representative.size() != columns.size()) {
        throw std::logic_error("sample dimensions do not match");
    }

    double distance = 0.0;
    for (std::size_t index = 0; index < columns.size(); ++index) {
        const SampledColumn& column = columns[index];
        const double difference = std::abs(
            static_cast<double>(query[column.source_column]) -
            static_cast<double>(sampled_representative[index]));
        distance +=
            static_cast<double>(column.multiplicity) *
            column.inverse_probability * difference;
    }
    return distance;
}

}  // namespace

SampledCoordinateL1AnnIndex::SampledCoordinateL1AnnIndex(
    const DenseMatrix& input,
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : SampledCoordinateL1AnnIndex(
          input,
          build_coordinate_sample_checked(
              input,
              probabilities,
              repetitions,
              random_engine))
{
}

SampledCoordinateL1AnnIndex::SampledCoordinateL1AnnIndex(
    const DenseMatrix& input,
    CoordinateSample coordinate_sample)
    : coordinate_sample_(std::move(coordinate_sample)),
      sampled_representatives_(
          filter_columns(input, coordinate_sample_.columns)),
      initial_cols_(input.cols())
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "SampledCoordinateL1AnnIndex expects at least one representative");
    }
}

std::size_t SampledCoordinateL1AnnIndex::query(
    std::span<const float> query) const
{
    if (query.size() != initial_cols_) {
        throw std::invalid_argument(
            "query dimension does not match index dimension");
    }
    for (const SampledColumn& column : coordinate_sample_.columns) {
        detail::validate_finite_value(
            query[column.source_column],
            "sampled query coordinate");
    }

    const std::size_t rows = sampled_representatives_.rows();
    if (rows == 1) {
        return 0;
    }
    const std::span<const SampledColumn> columns{
        coordinate_sample_.columns};
    std::size_t result = 0;
    double minimum_distance = weighted_l1_distance(
        query,
        sampled_representatives_.row(0),
        columns);
    for (std::size_t row = 1; row < rows; ++row) {
        const double distance = weighted_l1_distance(
            query,
            sampled_representatives_.row(row),
            columns);
        if (distance < minimum_distance) {
            minimum_distance = distance;
            result = row;
        }
    }
    return result;
}

IndexSpaceUsage SampledCoordinateL1AnnIndex::space_usage() const noexcept
{
    std::size_t sampled_multiplicity = 0;
    for (const SampledColumn& column : coordinate_sample_.columns) {
        sampled_multiplicity += column.multiplicity;
    }
    return IndexSpaceUsage{
        .index_payload_bytes =
            coordinate_sample_.columns.size() * sizeof(SampledColumn) +
            sampled_representatives_.values().size_bytes(),
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates = coordinate_sample_.columns.size(),
        .sampled_multiplicity = sampled_multiplicity,
    };
}

}  // namespace ultrahigh_ann
