#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <vector>

namespace ultrahigh_ann::detail {

[[nodiscard]] std::vector<double> compute_l2_sampling_probabilities(
    const DenseMatrix& representatives);
[[nodiscard]] std::vector<double>
compute_l2_sampling_probabilities_cpu_parallel(
    const DenseMatrix& representatives);

}  // namespace ultrahigh_ann::detail
