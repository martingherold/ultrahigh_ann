#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

enum class CpuDenseL2QueryStrategy {
    sequential,
    // Assign each complete query scan to one OpenMP thread.
    parallel_queries,
    // Split every query's representative scan across OpenMP threads.
    parallel_representatives,
    // Select a parallel dimension from the batch size and thread count.
    automatic,
};

// Exhaustive squared-L2 scan over an arbitrary dense representation. The
// borrowed representative matrix must outlive this scan engine.
class DenseL2ScanIndex {
public:
    explicit DenseL2ScanIndex(const DenseMatrix& representatives);
    DenseL2ScanIndex(DenseMatrix&&) = delete;
    DenseL2ScanIndex(const DenseMatrix&&) = delete;

    [[nodiscard]] std::size_t query(
        std::span<const float> query,
        CpuDenseL2QueryStrategy strategy =
            CpuDenseL2QueryStrategy::sequential) const;

    // Queries are a contiguous row-major query_count by dimension matrix.
    void query_batch(std::span<const float> queries,
                     std::size_t query_count,
                     std::span<std::size_t> output,
                     CpuDenseL2QueryStrategy strategy =
                         CpuDenseL2QueryStrategy::sequential) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    const DenseMatrix& representatives_;
};

}  // namespace ultrahigh_ann
