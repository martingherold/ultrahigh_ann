#pragma once

#include "ultrahigh_ann/core/dense_l2_scan_index.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

// Exact full-coordinate L2 scan. Squared distances are compared because the
// square root is monotone. The borrowed matrix must outlive this index.
class ExactL2Index {
public:
    explicit ExactL2Index(const DenseMatrix& representatives);
    ExactL2Index(DenseMatrix&&) = delete;
    ExactL2Index(const DenseMatrix&&) = delete;

    [[nodiscard]] std::size_t query(
        std::span<const float> query,
        CpuDenseL2QueryStrategy strategy =
            CpuDenseL2QueryStrategy::sequential) const;

    void query_batch(std::span<const float> queries,
                     std::size_t query_count,
                     std::span<std::size_t> output,
                     CpuDenseL2QueryStrategy strategy =
                         CpuDenseL2QueryStrategy::sequential) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    DenseL2ScanIndex scan_;
};

}  // namespace ultrahigh_ann
