#pragma once

#include "core/dense_matrix.hpp"
#include "core/index_space_usage.hpp"

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
        std::span<const float> query) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;

private:
    const DenseMatrix& representatives_;
};

}  // namespace ultrahigh_ann
