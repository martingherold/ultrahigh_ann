#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

// Exact full-coordinate L1 scan. This index does not own its representative
// matrix. The matrix must outlive the index and must not be moved from while
// the index exists.
class ExactL1Index {
public:
    explicit ExactL1Index(const DenseMatrix& representatives);
    ExactL1Index(DenseMatrix&&) = delete;
    ExactL1Index(const DenseMatrix&&) = delete;

    [[nodiscard]] std::size_t query(
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    const DenseMatrix& representatives_;
};

}  // namespace ultrahigh_ann
