#include "ultrahigh_ann/hnsvw25/l1/hierarchical/cuda_hierarchical_l1_ann_index.hpp"

#include "core/cuda_float_projection.hpp"
#include "hnsvw25/l1/hierarchical/cuda_median_l1_scan_index.hpp"
#include "hnsvw25/l1/hierarchical/l1_hierarchical_projection.hpp"
#include "ultrahigh_ann/core/cuda_dense_l1_scan_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

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
            "CudaHierarchicalL1AnnIndex expects at least one representative");
    }
    if (projection_dimension == 0) {
        throw std::invalid_argument(
            "projection_dimension has to be a positive integer");
    }
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
        return build_l1_importance_sample(input, repetitions, random_engine,
                                          probability_execution_policy);
    case ExecutionPolicy::gpu_fp32: {
        const L1GpuProbabilityResult result =
            compute_l1_importance_probabilities_gpu(input, device);
        return build_l1_importance_sample(result.probabilities, repetitions,
                                          random_engine);
    }
    case ExecutionPolicy::gpu_cublas_fp32:
        throw std::invalid_argument(
            "gpu_cublas_fp32 is unavailable for L1 probability computation; "
            "L1 distance has no matrix-multiplication identity");
    }
    throw std::invalid_argument("unknown L1 probability execution policy");
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
    return build_importance_sample(probabilities, repetitions, random_engine);
}

[[nodiscard]] detail::CudaFloatProjectionStrategy
projection_strategy(CudaHierarchicalL1QueryStrategy strategy)
{
    switch (strategy) {
    case CudaHierarchicalL1QueryStrategy::direct:
        return detail::CudaFloatProjectionStrategy::direct;
    case CudaHierarchicalL1QueryStrategy::gemm:
        return detail::CudaFloatProjectionStrategy::gemm;
    }
    throw std::invalid_argument("unknown CUDA hierarchical L1 query strategy");
}

struct CudaProjection {
    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    DenseMatrix cauchy_matrix;
    DenseMatrix projected_representatives;
    std::size_t sampled_multiplicity{};
};

[[nodiscard]] CudaProjection prepare_cuda_projection(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine)
{
    auto projection = detail::build_l1_hierarchical_projection_fp32(
        input, std::move(importance_sample), projection_dimension,
        random_engine);
    return CudaProjection{
        .importance_sample = std::move(projection.importance_sample),
        .original_dimension = projection.original_dimension,
        .cauchy_matrix = std::move(projection.cauchy_matrix),
        .projected_representatives =
            std::move(projection.projected_representatives),
        .sampled_multiplicity = projection.sampled_multiplicity,
    };
}

}  // namespace

struct CudaHierarchicalL1AnnIndex::Impl {
    Impl(CudaProjection prepared, int device)
        : importance_sample(std::move(prepared.importance_sample)),
          original_dimension(prepared.original_dimension),
          representative_count(prepared.projected_representatives.rows()),
          sampled_multiplicity(prepared.sampled_multiplicity),
          projection(prepared.cauchy_matrix, device),
          scan(prepared.projected_representatives, device)
    {
    }

    CoordinateSample importance_sample;
    std::size_t original_dimension{};
    std::size_t representative_count{};
    std::size_t sampled_multiplicity{};
    detail::CudaFloatProjection projection;
    detail::CudaMedianL1ScanIndex scan;
};

struct CudaHierarchicalL1AnnIndex::QueryWorkspace::Impl {
    Impl(const CudaHierarchicalL1AnnIndex::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          sampled_values(checked_product(
              maximum_batch_size, owner->importance_sample.columns.size(),
              "CUDA hierarchical L1 sampled query workspace")),
          projection_workspace(
              owner->projection.make_workspace(maximum_batch_size)),
          scan_workspace(owner->scan.make_workspace(maximum_batch_size)),
          payload_bytes(
              checked_add(checked_bytes<float>(
                              sampled_values.size(),
                              "CUDA hierarchical L1 sampled query workspace"),
                          checked_add(projection_workspace.payload_bytes(),
                                      scan_workspace.payload_bytes(),
                                      "CUDA hierarchical L1 workspace"),
                          "CUDA hierarchical L1 workspace"))
    {
    }

    const CudaHierarchicalL1AnnIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::vector<float> sampled_values;
    detail::CudaFloatProjection::Workspace projection_workspace;
    detail::CudaMedianL1ScanIndex::Workspace scan_workspace;
    std::size_t payload_bytes{};
};

bool cuda_hierarchical_l1_available() noexcept
{
    return cuda_dense_l1_scan_available();
}

CudaHierarchicalL1AnnIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaHierarchicalL1AnnIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaHierarchicalL1AnnIndex::QueryWorkspace&
CudaHierarchicalL1AnnIndex::QueryWorkspace::operator=(
    QueryWorkspace&&) noexcept = default;

CudaHierarchicalL1AnnIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t
CudaHierarchicalL1AnnIndex::QueryWorkspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t
CudaHierarchicalL1AnnIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaHierarchicalL1AnnIndex::CudaHierarchicalL1AnnIndex(
    const DenseMatrix& input, std::size_t repetitions,
    std::size_t projection_dimension, std::mt19937_64& random_engine,
    int device)
    : CudaHierarchicalL1AnnIndex(input, repetitions, projection_dimension,
                                 random_engine, device,
                                 ExecutionPolicy::sequential)
{
}

CudaHierarchicalL1AnnIndex::CudaHierarchicalL1AnnIndex(
    const DenseMatrix& input, std::size_t repetitions,
    std::size_t projection_dimension, std::mt19937_64& random_engine,
    int device, ExecutionPolicy probability_execution_policy)
    : CudaHierarchicalL1AnnIndex(
          input,
          build_sample_checked(input, repetitions, projection_dimension,
                               random_engine, device,
                               probability_execution_policy),
          projection_dimension, random_engine, device)
{
}

CudaHierarchicalL1AnnIndex::CudaHierarchicalL1AnnIndex(
    const DenseMatrix& input, std::span<const double> importance_probabilities,
    std::size_t repetitions, std::size_t projection_dimension,
    std::mt19937_64& random_engine, int device)
    : CudaHierarchicalL1AnnIndex(
          input,
          build_sample_checked(input, importance_probabilities, repetitions,
                               projection_dimension, random_engine),
          projection_dimension, random_engine, device)
{
}

CudaHierarchicalL1AnnIndex::CudaHierarchicalL1AnnIndex(
    const DenseMatrix& input, CoordinateSample importance_sample,
    std::size_t projection_dimension, std::mt19937_64& random_engine,
    int device)
    : implementation_(std::make_unique<Impl>(
          prepare_cuda_projection(input, std::move(importance_sample),
                                  projection_dimension, random_engine),
          device))
{
}

CudaHierarchicalL1AnnIndex::CudaHierarchicalL1AnnIndex(
    CudaHierarchicalL1AnnIndex&&) noexcept = default;

CudaHierarchicalL1AnnIndex& CudaHierarchicalL1AnnIndex::operator=(
    CudaHierarchicalL1AnnIndex&&) noexcept = default;

CudaHierarchicalL1AnnIndex::~CudaHierarchicalL1AnnIndex() = default;

CudaHierarchicalL1AnnIndex::QueryWorkspace
CudaHierarchicalL1AnnIndex::make_query_workspace(
    std::size_t maximum_batch_size) const
{
    return QueryWorkspace(std::make_unique<QueryWorkspace::Impl>(
        implementation_.get(), maximum_batch_size));
}

std::size_t CudaHierarchicalL1AnnIndex::query(
    std::span<const float> query, QueryWorkspace& workspace,
    CudaHierarchicalL1QueryStrategy strategy) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, workspace, strategy);
    return output.front();
}

void CudaHierarchicalL1AnnIndex::query_batch(
    std::span<const float> queries, std::size_t query_count,
    std::span<std::size_t> output, QueryWorkspace& workspace,
    CudaHierarchicalL1QueryStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA hierarchical L1 workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA hierarchical L1 workspace capacity");
    }
    const std::size_t expected_query_values =
        checked_product(query_count, implementation_->original_dimension,
                        "CUDA hierarchical L1 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match original dimension");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA hierarchical L1 output size does not match query count");
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
        "CUDA hierarchical L1 sampled query batch");
    std::span<float> sampled_values{workspace.implementation_->sampled_values};
    detail::pack_l1_hierarchical_sampled_queries(
        queries, query_count, implementation_->original_dimension,
        implementation_->importance_sample.columns,
        sampled_values.first(sampled_value_count));
    const float* projected_queries =
        implementation_->projection.project_batch_float(
            std::span<const float>{sampled_values.first(sampled_value_count)},
            query_count, workspace.implementation_->projection_workspace,
            projection_strategy(strategy));
    implementation_->scan.query_device_batch(
        projected_queries, query_count, output,
        workspace.implementation_->scan_workspace);
}

IndexSpaceUsage CudaHierarchicalL1AnnIndex::space_usage() const noexcept
{
    return IndexSpaceUsage{
        .index_payload_bytes =
            implementation_->importance_sample.columns.size() *
                sizeof(SampledColumn) +
            implementation_->projection.payload_bytes() +
            implementation_->scan.payload_bytes(),
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates =
            implementation_->importance_sample.columns.size(),
        .sampled_multiplicity = implementation_->sampled_multiplicity,
    };
}

int CudaHierarchicalL1AnnIndex::device() const noexcept
{
    return implementation_->scan.device();
}

const std::string& CudaHierarchicalL1AnnIndex::device_name() const noexcept
{
    return implementation_->scan.device_name();
}

}  // namespace ultrahigh_ann
