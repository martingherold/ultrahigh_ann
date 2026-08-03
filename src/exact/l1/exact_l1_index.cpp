#include "exact/l1/exact_l1_index.hpp"

#include "core/finite_values.hpp"
#include "core/l1_distance.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace ultrahigh_ann {

ExactL1Index::ExactL1Index(
    const DenseMatrix& representatives)
    : representatives_(representatives)
{
    if (representatives_.rows() == 0) {
        throw std::invalid_argument(
            "ExactL1Index expects at least one representative");
    }

    detail::validate_finite_values(
        representatives_.values(),
        "representatives");
}

std::size_t ExactL1Index::query(
    std::span<const float> query) const
{
    if (query.size() != representatives_.cols()) {
        throw std::invalid_argument(
            "query dimension does not match index dimension");
    }

    detail::validate_finite_values(query, "query");

    std::size_t nearest_index = 0;
    double nearest_distance =
        detail::l1_dist(
            query,
            representatives_.row(0));

    for (std::size_t row = 1;
         row < representatives_.rows();
         ++row) {
        const double distance =
            detail::l1_dist(
                query,
                representatives_.row(row));

        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_index = row;
        }
    }

    return nearest_index;
}

IndexSpaceUsage ExactL1Index::space_usage() const noexcept
{
    return IndexSpaceUsage{
        .index_payload_bytes = representatives_.values().size_bytes(),
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates = representatives_.cols(),
        .sampled_multiplicity = representatives_.cols(),
    };
}

}  // namespace ultrahigh_ann
