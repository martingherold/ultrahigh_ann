#include "l2_distance.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace ultrahigh_ann::detail {

double squared_l2_dist(
    std::span<const float> first,
    std::span<const float> second)
{
    if (first.size() != second.size()) {
        throw std::logic_error(
            "cannot compare rows with different dimensions");
    }
    double distance = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        const double difference =
            static_cast<double>(first[index]) -
            static_cast<double>(second[index]);
        distance += difference * difference;
    }
    return distance;
}

}  // namespace ultrahigh_ann::detail
