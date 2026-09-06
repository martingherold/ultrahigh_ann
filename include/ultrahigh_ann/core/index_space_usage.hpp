#pragma once

#include <cstddef>

namespace ultrahigh_ann {

// Logical payload sizes exclude allocator metadata and C++ object bookkeeping.
// They describe the standalone data an index must retain after construction.
struct IndexSpaceUsage {
    std::size_t index_payload_bytes{};
    std::size_t query_workspace_payload_bytes{};
    std::size_t unique_query_coordinates{};
    std::size_t sampled_multiplicity{};
};

}  // namespace ultrahigh_ann
