#pragma once

#include "ultrahigh_ann/core/cuda_dense_l1_scan_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_exact_l1_available() noexcept;

// Exact-search facade over CudaDenseL1ScanIndex using the original,
// full-coordinate representative matrix. Distance accumulation uses binary32
// arithmetic; predictions should be validated against ExactL1Index's binary64
// CPU accumulation when nearest representatives have very similar distances.
class CudaExactL1Index {
public:
    using QueryWorkspace = CudaDenseL1ScanIndex::QueryWorkspace;

    explicit CudaExactL1Index(const DenseMatrix& representatives,
                              int device = 0);
    CudaExactL1Index(DenseMatrix&&) = delete;
    CudaExactL1Index(const DenseMatrix&&) = delete;
    CudaExactL1Index(const CudaExactL1Index&) = delete;
    CudaExactL1Index& operator=(const CudaExactL1Index&) = delete;
    CudaExactL1Index(CudaExactL1Index&&) noexcept;
    CudaExactL1Index& operator=(CudaExactL1Index&&) noexcept;
    ~CudaExactL1Index();

    [[nodiscard]] QueryWorkspace
    make_query_workspace(std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t query(std::span<const float> query,
                                    QueryWorkspace& workspace) const;

    // Queries are a contiguous row-major query_count by dimension matrix.
    // The output span receives one representative row index per query.
    void query_batch(std::span<const float> queries, std::size_t query_count,
                     std::span<std::size_t> output,
                     QueryWorkspace& workspace) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

private:
    CudaDenseL1ScanIndex scan_;
};

} // namespace ultrahigh_ann
