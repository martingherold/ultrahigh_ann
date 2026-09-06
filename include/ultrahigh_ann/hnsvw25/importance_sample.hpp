#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"

namespace ultrahigh_ann {

// Compatibility name for callers that construct coordinate samples from
// importance probabilities. The representation itself is probability-policy
// independent and is also used by uniform sampling.
using ImportanceSample = CoordinateSample;

}  // namespace ultrahigh_ann
