#pragma once

#include <span>

namespace ultrahigh_ann::detail
{
[[nodiscard]]double l1_dist(std::span<const float> first, std::span<const float> second);
} // namespace ultrahigh_ann::detail
