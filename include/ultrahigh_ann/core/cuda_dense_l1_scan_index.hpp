#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_dense_l1_scan_available() noexcept;

// Exhaustive L1 scan over an arbitrary dense representation retained on one
// CUDA device. Distance accumulation uses binary32 arithmetic. The weighted
// overload evaluates sum_j weights[j] * abs(r[j] - q[j]).
class CudaDenseL1ScanIndex {
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
        friend class CudaDenseL1ScanIndex;
        struct Impl;

        explicit QueryWorkspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    explicit CudaDenseL1ScanIndex(const DenseMatrix& representatives,
                                  int device = 0);
    CudaDenseL1ScanIndex(const DenseMatrix& representatives,
                         std::span<const float> coordinate_weights,
                         int device = 0);
    CudaDenseL1ScanIndex(DenseMatrix&&) = delete;
    CudaDenseL1ScanIndex(const DenseMatrix&&) = delete;
    CudaDenseL1ScanIndex(const CudaDenseL1ScanIndex&) = delete;
    CudaDenseL1ScanIndex& operator=(const CudaDenseL1ScanIndex&) = delete;
    CudaDenseL1ScanIndex(CudaDenseL1ScanIndex&&) noexcept;
    CudaDenseL1ScanIndex& operator=(CudaDenseL1ScanIndex&&) noexcept;
    ~CudaDenseL1ScanIndex();

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
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace ultrahigh_ann
