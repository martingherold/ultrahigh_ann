#include "l1_distance.hpp"

#include <cmath>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>

namespace ultrahigh_ann::detail
{
    
[[nodiscard]]double l1_dist(std::span<const float> first,std::span<const float> second){
    if (first.size() != second.size()){
        throw std::logic_error("Cannot compare the distance between rows of different dimensions");
    }
    return std::transform_reduce(
    first.begin(),
    first.end(),
    second.begin(),
    0.0,
    std::plus<>{},
    [](float first, float second) {
        return std::abs(
            static_cast<double>(first) -
            static_cast<double>(second));
    });

}

} // namespace ultrahigh_ann::detail
