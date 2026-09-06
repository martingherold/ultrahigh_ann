#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ultrahigh_ann::detail {

struct L1CoordinateEmbedding {
    std::size_t original_dimension{};
    std::vector<std::size_t> source_columns;
    std::vector<float> weights;
    DenseMatrix sampled_representatives;
    std::size_t sampled_multiplicity{};
};

[[nodiscard]] L1CoordinateEmbedding
build_l1_coordinate_embedding(const DenseMatrix& input,
                              CoordinateSample coordinate_sample);

void pack_l1_coordinate_queries(std::span<const float> queries,
                                std::size_t query_count,
                                std::size_t original_dimension,
                                std::span<const std::size_t> source_columns,
                                std::span<float> output);

} // namespace ultrahigh_ann::detail
