#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"

#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"
#include "coordinate_sampling/l2_coordinate_embedding.hpp"
#include "ultrahigh_ann/core/dense_l2_scan_index.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    return build_coordinate_sample(probabilities, repetitions, random_engine);
}

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

struct SampledCoordinateL2AnnIndex::Impl {
    Impl(const DenseMatrix& input, CoordinateSample coordinate_sample)
        : embedding(detail::build_l2_coordinate_embedding(
              input,
              std::move(coordinate_sample))),
          scan(embedding.transformed_representatives)
    {
    }

    detail::L2CoordinateEmbedding embedding;
    DenseL2ScanIndex scan;
};

SampledCoordinateL2AnnIndex::SampledCoordinateL2AnnIndex(
    const DenseMatrix& input,
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : SampledCoordinateL2AnnIndex(
          input,
          build_coordinate_sample_checked(input,
                                          probabilities,
                                          repetitions,
                                          random_engine))
{
}

SampledCoordinateL2AnnIndex::SampledCoordinateL2AnnIndex(
    const DenseMatrix& input,
    CoordinateSample coordinate_sample)
    : implementation_(
          std::make_unique<Impl>(input, std::move(coordinate_sample)))
{
}

SampledCoordinateL2AnnIndex::SampledCoordinateL2AnnIndex(
    SampledCoordinateL2AnnIndex&&) noexcept = default;

SampledCoordinateL2AnnIndex& SampledCoordinateL2AnnIndex::operator=(
    SampledCoordinateL2AnnIndex&&) noexcept = default;

SampledCoordinateL2AnnIndex::~SampledCoordinateL2AnnIndex() = default;

std::size_t SampledCoordinateL2AnnIndex::query(
    std::span<const float> query,
    CpuDenseL2QueryStrategy strategy) const
{
    std::vector<float> packed_query(
        implementation_->embedding.source_columns.size());
    detail::pack_l2_coordinate_queries(
        query, 1, implementation_->embedding.original_dimension,
        implementation_->embedding.source_columns,
        implementation_->embedding.scales, packed_query);
    return implementation_->scan.query(packed_query, strategy);
}

void SampledCoordinateL2AnnIndex::query_batch(
    std::span<const float> queries,
    std::size_t query_count,
    std::span<std::size_t> output,
    CpuDenseL2QueryStrategy strategy) const
{
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "sampled L2 output size does not match query count");
    }
    const std::size_t packed_value_count = checked_product(
        query_count, implementation_->embedding.source_columns.size(),
        "sampled L2 packed query batch");
    std::vector<float> packed_queries(packed_value_count);
    detail::pack_l2_coordinate_queries(
        queries, query_count, implementation_->embedding.original_dimension,
        implementation_->embedding.source_columns,
        implementation_->embedding.scales, packed_queries);
    implementation_->scan.query_batch(packed_queries, query_count, output,
                                      strategy);
}

IndexSpaceUsage SampledCoordinateL2AnnIndex::space_usage() const noexcept
{
    IndexSpaceUsage usage = implementation_->scan.space_usage();
    usage.index_payload_bytes +=
        detail::l2_embedding_metadata_bytes(implementation_->embedding);
    usage.query_workspace_payload_bytes =
        implementation_->embedding.source_columns.size() * sizeof(float);
    usage.unique_query_coordinates =
        implementation_->embedding.source_columns.size();
    usage.sampled_multiplicity =
        implementation_->embedding.sampled_multiplicity;
    return usage;
}

}  // namespace ultrahigh_ann
