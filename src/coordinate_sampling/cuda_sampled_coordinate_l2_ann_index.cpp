#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index.hpp"

#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"
#include "coordinate_sampling/l2_coordinate_embedding.hpp"
#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"

#include <array>
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

[[nodiscard]] std::size_t checked_product(std::size_t left,
                                          std::size_t right,
                                          const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

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

[[nodiscard]] CudaDenseL2QueryStrategy dense_strategy(
    CudaSampledCoordinateL2QueryStrategy strategy)
{
    switch (strategy) {
    case CudaSampledCoordinateL2QueryStrategy::direct:
        return CudaDenseL2QueryStrategy::direct;
    case CudaSampledCoordinateL2QueryStrategy::gemm:
        return CudaDenseL2QueryStrategy::gemm;
    }
    throw std::invalid_argument("unknown CUDA sampled L2 query strategy");
}

}  // namespace

struct CudaSampledCoordinateL2AnnIndex::Impl {
    Impl(const DenseMatrix& input,
         CoordinateSample coordinate_sample,
         int device)
        : Impl(detail::build_l2_coordinate_embedding(
                   input,
                   std::move(coordinate_sample)),
               device)
    {
    }

    Impl(detail::L2CoordinateEmbedding prepared, int device)
        : initial_dimension(prepared.original_dimension),
          source_columns(std::move(prepared.source_columns)),
          scales(std::move(prepared.scales)),
          sampled_multiplicity(prepared.sampled_multiplicity),
          scan(prepared.transformed_representatives, device)
    {
    }

    std::size_t initial_dimension{};
    std::vector<std::size_t> source_columns;
    std::vector<float> scales;
    std::size_t sampled_multiplicity{};
    CudaDenseL2ScanIndex scan;
};

struct CudaSampledCoordinateL2AnnIndex::QueryWorkspace::Impl {
    Impl(const CudaSampledCoordinateL2AnnIndex::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          packed_query_count(checked_product(maximum_batch_size_value,
                                             owner_value->source_columns.size(),
                                             "CUDA sampled query workspace")),
          packed_queries(packed_query_count),
          dense_workspace(
              owner_value->scan.make_query_workspace(maximum_batch_size_value)),
          payload_bytes(packed_queries.size() * sizeof(float) +
                        dense_workspace.payload_bytes())
    {
    }

    const CudaSampledCoordinateL2AnnIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::size_t packed_query_count{};
    std::vector<float> packed_queries;
    CudaDenseL2ScanIndex::QueryWorkspace dense_workspace;
    std::size_t payload_bytes{};
};

bool cuda_sampled_coordinate_l2_available() noexcept
{
    return cuda_dense_l2_scan_available();
}

CudaSampledCoordinateL2AnnIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaSampledCoordinateL2AnnIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaSampledCoordinateL2AnnIndex::QueryWorkspace&
CudaSampledCoordinateL2AnnIndex::QueryWorkspace::operator=(
    QueryWorkspace&&) noexcept = default;

CudaSampledCoordinateL2AnnIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t CudaSampledCoordinateL2AnnIndex::QueryWorkspace::
    maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t CudaSampledCoordinateL2AnnIndex::QueryWorkspace::payload_bytes()
    const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaSampledCoordinateL2AnnIndex::CudaSampledCoordinateL2AnnIndex(
    const DenseMatrix& input,
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine,
    int device)
    : CudaSampledCoordinateL2AnnIndex(
          input,
          build_coordinate_sample_checked(input,
                                          probabilities,
                                          repetitions,
                                          random_engine),
          device)
{
}

CudaSampledCoordinateL2AnnIndex::CudaSampledCoordinateL2AnnIndex(
    const DenseMatrix& input,
    CoordinateSample coordinate_sample,
    int device)
    : implementation_(
          std::make_unique<Impl>(input, std::move(coordinate_sample), device))
{
}

CudaSampledCoordinateL2AnnIndex::CudaSampledCoordinateL2AnnIndex(
    CudaSampledCoordinateL2AnnIndex&&) noexcept = default;

CudaSampledCoordinateL2AnnIndex& CudaSampledCoordinateL2AnnIndex::operator=(
    CudaSampledCoordinateL2AnnIndex&&) noexcept = default;

CudaSampledCoordinateL2AnnIndex::~CudaSampledCoordinateL2AnnIndex() = default;

CudaSampledCoordinateL2AnnIndex::QueryWorkspace
CudaSampledCoordinateL2AnnIndex::make_query_workspace(
    std::size_t maximum_batch_size) const
{
    return QueryWorkspace(std::make_unique<QueryWorkspace::Impl>(
        implementation_.get(), maximum_batch_size));
}

std::size_t CudaSampledCoordinateL2AnnIndex::query(
    std::span<const float> query,
    QueryWorkspace& workspace,
    CudaSampledCoordinateL2QueryStrategy strategy) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, workspace, strategy);
    return output.front();
}

void CudaSampledCoordinateL2AnnIndex::query_batch(
    std::span<const float> queries,
    std::size_t query_count,
    std::span<std::size_t> output,
    QueryWorkspace& workspace,
    CudaSampledCoordinateL2QueryStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA sampled workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA sampled workspace capacity");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA sampled output size does not match query count");
    }
    if (query_count == 0) {
        return;
    }

    const std::size_t sampled_dimension =
        implementation_->source_columns.size();
    const std::size_t packed_value_count = checked_product(
        query_count, sampled_dimension, "CUDA sampled packed query batch");
    std::span<float> packed_queries{workspace.implementation_->packed_queries};
    detail::pack_l2_coordinate_queries(
        queries, query_count, implementation_->initial_dimension,
        implementation_->source_columns, implementation_->scales,
        packed_queries.first(packed_value_count));
    implementation_->scan.query_batch(
        std::span<const float>{packed_queries.first(packed_value_count)},
        query_count, output, workspace.implementation_->dense_workspace,
        dense_strategy(strategy));
}

IndexSpaceUsage CudaSampledCoordinateL2AnnIndex::space_usage() const noexcept
{
    IndexSpaceUsage usage = implementation_->scan.space_usage();
    usage.index_payload_bytes +=
        implementation_->source_columns.size() * sizeof(std::size_t) +
        implementation_->scales.size() * sizeof(float);
    usage.unique_query_coordinates = implementation_->source_columns.size();
    usage.sampled_multiplicity = implementation_->sampled_multiplicity;
    return usage;
}

int CudaSampledCoordinateL2AnnIndex::device() const noexcept
{
    return implementation_->scan.device();
}

const std::string& CudaSampledCoordinateL2AnnIndex::device_name() const noexcept
{
    return implementation_->scan.device_name();
}

}  // namespace ultrahigh_ann
