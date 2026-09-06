#include "ultrahigh_ann/hnsvw25/l2/flat/cuda_l2_ann_index.hpp"

#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>

namespace ultrahigh_ann {
namespace {

[[nodiscard]] CoordinateSample
build_cuda_flat_sample(const DenseMatrix& input, std::size_t repetitions,
                       std::mt19937_64& random_engine, int device,
                       ExecutionPolicy probability_execution_policy)
{
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

}  // namespace

CudaFlatL2AnnIndex::CudaFlatL2AnnIndex(
    const DenseMatrix& input, std::size_t repetitions,
    std::mt19937_64& random_engine, int device,
    ExecutionPolicy probability_execution_policy)
    : index_(input,
             build_cuda_flat_sample(input, repetitions, random_engine, device,
                                    probability_execution_policy),
             device)
{}

CudaFlatL2AnnIndex::CudaFlatL2AnnIndex(
    const DenseMatrix& input, std::span<const double> importance_probabilities,
    std::size_t repetitions, std::mt19937_64& random_engine, int device)
    : index_(input, importance_probabilities, repetitions, random_engine,
             device)
{}

CudaFlatL2AnnIndex::QueryWorkspace
CudaFlatL2AnnIndex::make_query_workspace(std::size_t maximum_batch_size) const
{
    return index_.make_query_workspace(maximum_batch_size);
}

std::size_t
CudaFlatL2AnnIndex::query(std::span<const float> query,
                          QueryWorkspace& workspace,
                          CudaSampledCoordinateL2QueryStrategy strategy) const
{
    return index_.query(query, workspace, strategy);
}

void CudaFlatL2AnnIndex::query_batch(
    std::span<const float> queries, std::size_t query_count,
    std::span<std::size_t> output, QueryWorkspace& workspace,
    CudaSampledCoordinateL2QueryStrategy strategy) const
{
    index_.query_batch(queries, query_count, output, workspace, strategy);
}

IndexSpaceUsage CudaFlatL2AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

int CudaFlatL2AnnIndex::device() const noexcept
{
    return index_.device();
}

const std::string& CudaFlatL2AnnIndex::device_name() const noexcept
{
    return index_.device_name();
}

}  // namespace ultrahigh_ann
