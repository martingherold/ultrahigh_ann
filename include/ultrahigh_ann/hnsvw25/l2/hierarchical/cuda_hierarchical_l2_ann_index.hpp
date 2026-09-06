#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_hierarchical_l2_available() noexcept;

enum class CudaHierarchicalL2QueryStrategy {
    direct,
    gemm,
};

// Gather sampled query coordinates on the host, then apply the paper's
// Rademacher JL projection and exhaustively scan projected representatives on
// one CUDA device. Both stages use binary32 arithmetic: direct selects CUDA
// reduction kernels, while gemm selects cuBLAS SGEMM. Transfer of the compact
// sampled query is part of query time. The randomized transform is prepared
// once on the host during construction and narrowed to binary32. Projected
// representatives are then constructed from that narrowed transform with
// binary32 accumulation before both are uploaded.
class CudaHierarchicalL2AnnIndex {
  public:
    class QueryWorkspace {
      public:
        QueryWorkspace(const QueryWorkspace&) = delete;
        QueryWorkspace& operator=(const QueryWorkspace&) = delete;
        QueryWorkspace(QueryWorkspace&&) noexcept;
        QueryWorkspace& operator=(QueryWorkspace&&) noexcept;
        ~QueryWorkspace();

        [[nodiscard]] std::size_t maximum_batch_size() const noexcept;
        [[nodiscard]] std::size_t payload_bytes() const noexcept;

      private:
        friend class CudaHierarchicalL2AnnIndex;
        struct Impl;

        explicit QueryWorkspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    CudaHierarchicalL2AnnIndex(const DenseMatrix& input,
                               std::size_t repetitions,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0,
                               ExecutionPolicy probability_execution_policy =
                                   ExecutionPolicy::gpu_fp32);

    CudaHierarchicalL2AnnIndex(const DenseMatrix& input,
                               std::span<const double> importance_probabilities,
                               std::size_t repetitions,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0);

    // Exposed for deterministic reuse and backend-comparison tests.
    CudaHierarchicalL2AnnIndex(const DenseMatrix& input,
                               CoordinateSample importance_sample,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0);

    CudaHierarchicalL2AnnIndex(const CudaHierarchicalL2AnnIndex&) = delete;
    CudaHierarchicalL2AnnIndex&
    operator=(const CudaHierarchicalL2AnnIndex&) = delete;
    CudaHierarchicalL2AnnIndex(CudaHierarchicalL2AnnIndex&&) noexcept;
    CudaHierarchicalL2AnnIndex&
    operator=(CudaHierarchicalL2AnnIndex&&) noexcept;
    ~CudaHierarchicalL2AnnIndex();

    [[nodiscard]] QueryWorkspace
    make_query_workspace(std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t
    query(std::span<const float> query, QueryWorkspace& workspace,
          CudaHierarchicalL2QueryStrategy strategy =
              CudaHierarchicalL2QueryStrategy::direct) const;

    // Queries are a contiguous row-major query_count by original-dimension
    // matrix. The output span receives one representative row per query.
    void query_batch(std::span<const float> queries, std::size_t query_count,
                     std::span<std::size_t> output, QueryWorkspace& workspace,
                     CudaHierarchicalL2QueryStrategy strategy =
                         CudaHierarchicalL2QueryStrategy::direct) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace ultrahigh_ann
