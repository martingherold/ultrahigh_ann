#pragma once

#include <span>

namespace ultrahigh_ann::detail {

[[nodiscard]] double squared_l2_dist(
    std::span<const float> first,
    std::span<const float> second);

}  // namespace ultrahigh_ann::detail
