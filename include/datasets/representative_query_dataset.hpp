#pragma once

#include "core/dense_matrix.hpp"

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

// Load files produced by the dataset transforms. When
// representative_labels.npy is absent, row indices are used as representative
// labels for backward compatibility.
[[nodiscard]] RepresentativeQueryDataset
load_representative_query_dataset(
    const std::filesystem::path& directory);

}  // namespace ultrahigh_ann
