#include "ultrahigh_ann/hnsvw25/l2/hierarchical/cuda_hierarchical_l2_ann_index.hpp"

#include "core/cuda_float_projection.hpp"
#include "core/finite_values.hpp"
#include "hnsvw25/l2/hierarchical/l2_hierarchical_projection.hpp"
#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ultrahigh_ann {
namespace {

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      const char* description)
{
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left + right;
}

template <class T>
[[nodiscard]] std::size_t checked_bytes(std::size_t count,
                                        const char* description)
{
    return checked_product(count, sizeof(T), description);
}

void validate_index_parameters(const DenseMatrix& input,
                               std::size_t projection_dimension)
{
    if (input.rows() == 0) {
        throw std::invalid_argument(
            "CudaHierarchicalL2AnnIndex expects at least one representative");
    }
    if (projection_dimension == 0) {
        throw std::invalid_argument(
            "projection_dimension has to be a positive integer");
    }
}

[[nodiscard]] CoordinateSample
build_sample_checked(const DenseMatrix& input,
                     std::span<const double> probabilities,
                     std::size_t repetitions, std::size_t projection_dimension,
                     std::mt19937_64& random_engine)
{
    validate_index_parameters(input, projection_dimension);
    if (probabilities.size() != input.cols()) {
        throw std::invalid_argument(
            "importance probability dimension does not match input");
    }
    return build_l2_importance_sample(probabilities, repetitions,
                                      random_engine);
}

[[nodiscard]] CoordinateSample
build_sample_checked(const DenseMatrix& input, std::size_t repetitions,
                     std::size_t projection_dimension,
                     std::mt19937_64& random_engine, int device,
                     ExecutionPolicy probability_execution_policy)
{
    validate_index_parameters(input, projection_dimension);
    switch (probability_execution_policy) {
    case ExecutionPolicy::sequential:
    case ExecutionPolicy::cpu_parallel:
        return build_l2_importance_sample(input, repetitions, random_engine,
                                          probability_execution_policy);
    case ExecutionPolicy::gpu_fp32:
    case ExecutionPolicy::gpu_cublas_fp32: {
        const L2GpuDistanceBackend distance_backend =
            probability_execution_policy == ExecutionPolicy::gpu_cublas_fp32
                ? L2GpuDistanceBackend::cublas
                : L2GpuDistanceBackend::direct;
        const L2GpuProbabilityResult result =
            compute_l2_importance_probabilities_gpu(input, device, 128,
                                                    distance_backend);
        return build_l2_importance_sample(result.probabilities, repetitions,
                                          random_engine);
    }
    }
    throw std::invalid_argument("unknown L2 probability execution policy");
}

[[nodiscard]] CudaDenseL2QueryStrategy
dense_strategy(CudaHierarchicalL2QueryStrategy strategy)
{
    switch (strategy) {
    case CudaHierarchicalL2QueryStrategy::direct:
        return CudaDenseL2QueryStrategy::direct;
    case CudaHierarchicalL2QueryStrategy::gemm:
        return CudaDenseL2QueryStrategy::gemm;
    }
    throw std::invalid_argument("unknown CUDA hierarchical L2 query strategy");
}

[[nodiscard]] detail::CudaFloatProjectionStrategy
projection_strategy(CudaHierarchicalL2QueryStrategy strategy)
{
    switch (strategy) {
    case CudaHierarchicalL2QueryStrategy::direct:
        return detail::CudaFloatProjectionStrategy::direct;
    case CudaHierarchicalL2QueryStrategy::gemm:
        return detail::CudaFloatProjectionStrategy::gemm;
    }
    throw std::invalid_argument("unknown CUDA hierarchical L2 query strategy");
}

void pack_sampled_queries(std::span<const float> queries,
                          std::size_t query_count,
                          std::size_t original_dimension,
                          std::span<const SampledColumn> sampled_columns,
                          std::span<float> output)
{
    const std::size_t expected_query_values = checked_product(
        query_count, original_dimension, "CUDA hierarchical query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    const std::size_t expected_output_values =
        checked_product(query_count, sampled_columns.size(),
                        "CUDA hierarchical sampled query batch");
    if (output.size() != expected_output_values) {
        throw std::invalid_argument(
            "CUDA hierarchical sampled-query output has the wrong size");
    }

    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        const auto query = queries.subspan(query_index * original_dimension,
                                           original_dimension);
        const std::size_t output_offset = query_index * sampled_columns.size();
        for (std::size_t sampled_index = 0;
             sampled_index < sampled_columns.size(); ++sampled_index) {
            const float value =
                query[sampled_columns[sampled_index].source_column];
            detail::validate_finite_value(value, "sampled query coordinate");
            output[output_offset + sampled_index] = value;
        }
    }
}

struct CudaProjection {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix jl_matrix;
    DenseMatrix projected_representatives;
    std::size_t representative_count{};
    std::size_t sampled_multiplicity{};
};

[[nodiscard]] CudaProjection prepare_cuda_projection(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine)
{
    auto projection = detail::build_l2_hierarchical_projection_fp32(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    return CudaProjection{
        .importance_sample = std::move(projection.importance_sample),
        .original_dimension = projection.original_dimension,
        .jl_matrix = std::move(projection.jl_matrix),
        .projected_representatives =
            std::move(projection.projected_representatives),
        .representative_count = input.rows(),
        .sampled_multiplicity = projection.sampled_multiplicity,
    };
}

}  // namespace

struct CudaHierarchicalL2AnnIndex::Impl {
    Impl(CudaProjection prepared, int device)
        : importance_sample(std::move(prepared.importance_sample)),
          original_dimension(prepared.original_dimension),
          representative_count(prepared.representative_count),
          sampled_multiplicity(prepared.sampled_multiplicity),
          projection(prepared.jl_matrix, device),
          scan(prepared.projected_representatives, device)
    {}

    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    std::size_t representative_count{};
    std::size_t sampled_multiplicity{};
    detail::CudaFloatProjection projection;
    CudaDenseL2ScanIndex scan;
};

struct CudaHierarchicalL2AnnIndex::QueryWorkspace::Impl {
    Impl(const CudaHierarchicalL2AnnIndex::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          sampled_values(
              checked_product(maximum_batch_size_value,
                              owner_value->importance_sample.columns.size(),
                              "CUDA hierarchical sampled query workspace")),
          projection_workspace(
              owner_value->projection.make_workspace(maximum_batch_size_value)),
          dense_workspace(
              owner_value->scan.make_query_workspace(maximum_batch_size_value)),
          payload_bytes(checked_add(
              checked_bytes<float>(sampled_values.size(),
                                   "CUDA hierarchical sampled workspace"),
              checked_add(projection_workspace.payload_bytes(),
                          dense_workspace.payload_bytes(),
                          "CUDA hierarchical workspace"),
              "CUDA hierarchical workspace"))
    {}

    const CudaHierarchicalL2AnnIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::vector<float> sampled_values;
    detail::CudaFloatProjection::Workspace projection_workspace;
    CudaDenseL2ScanIndex::QueryWorkspace dense_workspace;
    std::size_t payload_bytes{};
};

bool cuda_hierarchical_l2_available() noexcept
{
    return cuda_dense_l2_scan_available();
}

CudaHierarchicalL2AnnIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{}

CudaHierarchicalL2AnnIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaHierarchicalL2AnnIndex::QueryWorkspace&
CudaHierarchicalL2AnnIndex::QueryWorkspace::operator=(
    QueryWorkspace&&) noexcept = default;

CudaHierarchicalL2AnnIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t
CudaHierarchicalL2AnnIndex::QueryWorkspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t
CudaHierarchicalL2AnnIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaHierarchicalL2AnnIndex::CudaHierarchicalL2AnnIndex(
    const DenseMatrix& input, std::size_t repetitions,
    std::size_t projection_dimension, std::mt19937_64& random_engine,
    int device, ExecutionPolicy probability_execution_policy)
    : CudaHierarchicalL2AnnIndex(
          input,
          build_sample_checked(input, repetitions, projection_dimension,
                               random_engine, device,
                               probability_execution_policy),
          projection_dimension, random_engine, device)
{}

CudaHierarchicalL2AnnIndex::CudaHierarchicalL2AnnIndex(
    const DenseMatrix& input, std::span<const double> importance_probabilities,
    std::size_t repetitions, std::size_t projection_dimension,
    std::mt19937_64& random_engine, int device)
    : CudaHierarchicalL2AnnIndex(
          input,
          build_sample_checked(input, importance_probabilities, repetitions,
                               projection_dimension, random_engine),
          projection_dimension, random_engine, device)
{}

CudaHierarchicalL2AnnIndex::CudaHierarchicalL2AnnIndex(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine,
    int device)
    : implementation_(std::make_unique<Impl>(
          prepare_cuda_projection(input, std::move(importance_sample),
                                  projection_dimension, random_engine),
          device))
{}

CudaHierarchicalL2AnnIndex::CudaHierarchicalL2AnnIndex(
    CudaHierarchicalL2AnnIndex&&) noexcept = default;

CudaHierarchicalL2AnnIndex& CudaHierarchicalL2AnnIndex::operator=(
    CudaHierarchicalL2AnnIndex&&) noexcept = default;

CudaHierarchicalL2AnnIndex::~CudaHierarchicalL2AnnIndex() = default;

CudaHierarchicalL2AnnIndex::QueryWorkspace
CudaHierarchicalL2AnnIndex::make_query_workspace(
    std::size_t maximum_batch_size) const
{
    return QueryWorkspace(std::make_unique<QueryWorkspace::Impl>(
        implementation_.get(), maximum_batch_size));
}

std::size_t CudaHierarchicalL2AnnIndex::query(
    std::span<const float> query, QueryWorkspace& workspace,
    CudaHierarchicalL2QueryStrategy strategy) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, workspace, strategy);
    return output.front();
}

void CudaHierarchicalL2AnnIndex::query_batch(
    std::span<const float> queries, std::size_t query_count,
    std::span<std::size_t> output, QueryWorkspace& workspace,
    CudaHierarchicalL2QueryStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA hierarchical workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA hierarchical workspace capacity");
    }
    const std::size_t expected_query_values =
        checked_product(query_count, implementation_->original_dimension,
                        "CUDA hierarchical query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA hierarchical output size does not match query count");
    }
    if (query_count == 0) {
        return;
    }
    if (implementation_->representative_count == 1) {
        std::ranges::fill(output, std::size_t{0});
        return;
    }

    const std::size_t sampled_value_count = checked_product(
        query_count, implementation_->importance_sample.columns.size(),
        "CUDA hierarchical sampled query batch");
    std::span<float> sampled_values{workspace.implementation_->sampled_values};
    pack_sampled_queries(queries, query_count,
                         implementation_->original_dimension,
                         implementation_->importance_sample.columns,
                         sampled_values.first(sampled_value_count));
    const float* projected_queries =
        implementation_->projection.project_batch_float(
            std::span<const float>{sampled_values.first(sampled_value_count)},
            query_count, workspace.implementation_->projection_workspace,
            projection_strategy(strategy));
    implementation_->scan.query_device_batch(
        projected_queries, query_count, output,
        workspace.implementation_->dense_workspace, dense_strategy(strategy));
}

IndexSpaceUsage CudaHierarchicalL2AnnIndex::space_usage() const noexcept
{
    IndexSpaceUsage usage = implementation_->scan.space_usage();
    usage.index_payload_bytes +=
        implementation_->importance_sample.columns.size() *
            sizeof(SampledColumn) +
        implementation_->projection.payload_bytes();
    usage.unique_query_coordinates =
        implementation_->importance_sample.columns.size();
    usage.sampled_multiplicity = implementation_->sampled_multiplicity;
    return usage;
}

int CudaHierarchicalL2AnnIndex::device() const noexcept
{
    return implementation_->scan.device();
}

const std::string& CudaHierarchicalL2AnnIndex::device_name() const noexcept
{
    return implementation_->scan.device_name();
}

}  // namespace ultrahigh_ann
