#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_dense_l2_scan_available() noexcept;

enum class CudaDenseL2QueryStrategy {
    // One full-coordinate distance kernel per query-representative pair.
    direct,
    // Batched SGEMM with direct argmin selection over the binary32 identity
    // ||r||^2 + ||q||^2 - 2 r*q. Cancellation can affect the selected row.
    gemm,
};

// Exhaustive squared-L2 scan over an arbitrary dense representation retained
// on one CUDA device. Distance accumulation uses binary32 arithmetic.
class CudaDenseL2ScanIndex {
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
        friend class CudaDenseL2ScanIndex;
        struct Impl;

        explicit QueryWorkspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    explicit CudaDenseL2ScanIndex(const DenseMatrix& representatives,
                                  int device = 0);
    CudaDenseL2ScanIndex(DenseMatrix&&) = delete;
    CudaDenseL2ScanIndex(const DenseMatrix&&) = delete;
    CudaDenseL2ScanIndex(const CudaDenseL2ScanIndex&) = delete;
    CudaDenseL2ScanIndex& operator=(const CudaDenseL2ScanIndex&) = delete;
    CudaDenseL2ScanIndex(CudaDenseL2ScanIndex&&) noexcept;
    CudaDenseL2ScanIndex& operator=(CudaDenseL2ScanIndex&&) noexcept;
    ~CudaDenseL2ScanIndex();

    [[nodiscard]] QueryWorkspace
    make_query_workspace(std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t query(std::span<const float> query,
                                    QueryWorkspace& workspace) const;

    // Queries are a contiguous row-major query_count by dimension matrix.
    // The output span receives one representative row index per query.
    void query_batch(std::span<const float> queries, std::size_t query_count,
                     std::span<std::size_t> output, QueryWorkspace& workspace,
                     CudaDenseL2QueryStrategy strategy =
                         CudaDenseL2QueryStrategy::direct) const;

    // Advanced composition API. device_queries must address a contiguous
    // row-major query_count by dimension matrix on this index's CUDA device.
    // The caller owns the storage and must keep it alive until this call
    // returns. Results are copied back into the host output span.
    void query_device_batch(
        const float* device_queries,
        std::size_t query_count,
        std::span<std::size_t> output,
        QueryWorkspace& workspace,
        CudaDenseL2QueryStrategy strategy =
            CudaDenseL2QueryStrategy::direct) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace ultrahigh_ann
