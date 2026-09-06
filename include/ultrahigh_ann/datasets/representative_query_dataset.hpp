#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace ultrahigh_ann {

struct RepresentativeQueryDataset {
    DenseMatrix representatives;
    DenseMatrix queries;
    std::vector<std::size_t> representative_labels;
    std::vector<std::size_t> query_labels;
};

// Load a two-dimensional little-endian float32 NPY matrix.
[[nodiscard]] DenseMatrix load_float_matrix_npy(
    const std::filesystem::path& path);

// Load a one-dimensional little-endian uint16 NPY label array.
[[nodiscard]] std::vector<std::size_t> load_label_vector_npy(
    const std::filesystem::path& path);

// Load only representatives.npy from a transformed dataset directory.
[[nodiscard]] DenseMatrix load_representative_matrix(
    const std::filesystem::path& directory);

// Load files produced by the dataset transforms. This library convenience
// loader retains the legacy row-index fallback when representative_labels.npy
// is absent. The benchmark uses a stricter paired-label contract for quality
// metrics.
[[nodiscard]] RepresentativeQueryDataset
load_representative_query_dataset(
    const std::filesystem::path& directory);

}  // namespace ultrahigh_ann
