#include "ultrahigh_ann/hnsvw25/l1/flat/cuda_l1_ann_index.hpp"

#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

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

}  // namespace

CudaFlatL1AnnIndex::CudaFlatL1AnnIndex(const DenseMatrix& input,
                                       std::size_t repetitions,
                                       std::mt19937_64& random_engine,
                                       int device)
    : CudaFlatL1AnnIndex(input, repetitions, random_engine, device,
                         ExecutionPolicy::sequential)
{
}

CudaFlatL1AnnIndex::CudaFlatL1AnnIndex(
    const DenseMatrix& input, std::size_t repetitions,
    std::mt19937_64& random_engine, int device,
    ExecutionPolicy probability_execution_policy)
    : index_(input,
             build_cuda_flat_sample(input, repetitions, random_engine, device,
                                    probability_execution_policy),
             device)
{
}

CudaFlatL1AnnIndex::CudaFlatL1AnnIndex(
    const DenseMatrix& input, std::span<const double> importance_probabilities,
    std::size_t repetitions, std::mt19937_64& random_engine, int device)
    : index_(input, importance_probabilities, repetitions, random_engine,
             device)
{
}

CudaFlatL1AnnIndex::QueryWorkspace
CudaFlatL1AnnIndex::make_query_workspace(std::size_t maximum_batch_size) const
{
    return index_.make_query_workspace(maximum_batch_size);
}

std::size_t CudaFlatL1AnnIndex::query(std::span<const float> query,
                                      QueryWorkspace& workspace) const
{
    return index_.query(query, workspace);
}

void CudaFlatL1AnnIndex::query_batch(std::span<const float> queries,
                                     std::size_t query_count,
                                     std::span<std::size_t> output,
                                     QueryWorkspace& workspace) const
{
    index_.query_batch(queries, query_count, output, workspace);
}

IndexSpaceUsage CudaFlatL1AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

int CudaFlatL1AnnIndex::device() const noexcept
{
    return index_.device();
}

const std::string& CudaFlatL1AnnIndex::device_name() const noexcept
{
    return index_.device_name();
}

}  // namespace ultrahigh_ann
