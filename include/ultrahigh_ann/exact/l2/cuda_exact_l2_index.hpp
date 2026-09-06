#pragma once

#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_exact_l2_available() noexcept;

using CudaExactL2QueryStrategy = CudaDenseL2QueryStrategy;

// Exact-search facade over CudaDenseL2ScanIndex using the original,
// full-coordinate representative matrix. Distance accumulation uses binary32
// arithmetic; predictions should be validated against ExactL2Index's binary64
// CPU accumulation.
class CudaExactL2Index {
public:
    using QueryWorkspace = CudaDenseL2ScanIndex::QueryWorkspace;

    explicit CudaExactL2Index(const DenseMatrix& representatives,
                              int device = 0);
    CudaExactL2Index(DenseMatrix&&) = delete;
    CudaExactL2Index(const DenseMatrix&&) = delete;
    CudaExactL2Index(const CudaExactL2Index&) = delete;
    CudaExactL2Index& operator=(const CudaExactL2Index&) = delete;
    CudaExactL2Index(CudaExactL2Index&&) noexcept;
    CudaExactL2Index& operator=(CudaExactL2Index&&) noexcept;
    ~CudaExactL2Index();

    [[nodiscard]] QueryWorkspace make_query_workspace(
        std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t query(std::span<const float> query,
                                    QueryWorkspace& workspace) const;

    // Queries are a contiguous row-major query_count by dimension matrix.
    // The output span receives one representative row index per query.
    void query_batch(std::span<const float> queries,
                     std::size_t query_count,
                     std::span<std::size_t> output,
                     QueryWorkspace& workspace,
                     CudaExactL2QueryStrategy strategy =
                         CudaExactL2QueryStrategy::direct) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

private:
    CudaDenseL2ScanIndex scan_;
};

}  // namespace ultrahigh_ann
