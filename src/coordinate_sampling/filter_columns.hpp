#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"

#include <span>

namespace ultrahigh_ann
{

[[nodiscard]] DenseMatrix filter_columns(
    const DenseMatrix& representatives,
    std::span<const SampledColumn> columns);
    
} // namespace ultrahigh_ann
