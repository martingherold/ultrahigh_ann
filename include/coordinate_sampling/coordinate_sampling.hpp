#pragma once

#include "coordinate_sampling/coordinate_sample.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

// Independently include coordinate j in each of `repetitions` trials with
// probability probabilities[j]. Repeated occurrences are aggregated.
[[nodiscard]] CoordinateSample build_coordinate_sample(
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine);

}  // namespace ultrahigh_ann
