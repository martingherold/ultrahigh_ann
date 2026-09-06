#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_hierarchical_l1_available() noexcept;

enum class CudaHierarchicalL1QueryStrategy {
    direct,
    gemm,
};

// Gather sampled query coordinates on the host, project them with the paper's
// binary32 Cauchy transform on one CUDA device, and select the representative
// having the smallest median absolute projected difference. The direct and
// gemm strategies select a reduction kernel or SGEMM for projection; both use
// the same deterministic binary32 CUDA median and argmin kernels. The transform
// is prepared once on the host and narrowed to binary32. Projected
// representatives are then constructed from that narrowed transform with
// binary32 accumulation before both are uploaded.
class CudaHierarchicalL1AnnIndex {
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
        friend class CudaHierarchicalL1AnnIndex;
        struct Impl;

        explicit QueryWorkspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    CudaHierarchicalL1AnnIndex(const DenseMatrix& input,
                               std::size_t repetitions,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0);

    CudaHierarchicalL1AnnIndex(const DenseMatrix& input,
                               std::size_t repetitions,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device,
                               ExecutionPolicy probability_execution_policy);

    CudaHierarchicalL1AnnIndex(const DenseMatrix& input,
                               std::span<const double> importance_probabilities,
                               std::size_t repetitions,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0);

    // Exposed for deterministic backend-comparison tests and sample reuse.
    CudaHierarchicalL1AnnIndex(const DenseMatrix& input,
                               CoordinateSample importance_sample,
                               std::size_t projection_dimension,
                               std::mt19937_64& random_engine, int device = 0);

    CudaHierarchicalL1AnnIndex(const CudaHierarchicalL1AnnIndex&) = delete;
    CudaHierarchicalL1AnnIndex&
    operator=(const CudaHierarchicalL1AnnIndex&) = delete;
    CudaHierarchicalL1AnnIndex(CudaHierarchicalL1AnnIndex&&) noexcept;
    CudaHierarchicalL1AnnIndex&
    operator=(CudaHierarchicalL1AnnIndex&&) noexcept;
    ~CudaHierarchicalL1AnnIndex();

    [[nodiscard]] QueryWorkspace
    make_query_workspace(std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t
    query(std::span<const float> query, QueryWorkspace& workspace,
          CudaHierarchicalL1QueryStrategy strategy =
              CudaHierarchicalL1QueryStrategy::direct) const;

    // Queries are a contiguous row-major query_count by original-dimension
    // matrix. Only sampled coordinates cross the host-device boundary.
    void query_batch(std::span<const float> queries, std::size_t query_count,
                     std::span<std::size_t> output, QueryWorkspace& workspace,
                     CudaHierarchicalL1QueryStrategy strategy =
                         CudaHierarchicalL1QueryStrategy::direct) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace ultrahigh_ann
