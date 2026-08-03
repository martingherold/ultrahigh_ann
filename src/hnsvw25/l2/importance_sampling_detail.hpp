#pragma once

#include "core/dense_matrix.hpp"

#include <vector>

namespace ultrahigh_ann::detail {

[[nodiscard]] std::vector<double> compute_l2_sampling_probabilities(
    const DenseMatrix& representatives);

}  // namespace ultrahigh_ann::detail
