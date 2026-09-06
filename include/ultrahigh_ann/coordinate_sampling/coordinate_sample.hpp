#pragma once

#include <cstddef>
#include <vector>

namespace ultrahigh_ann {

// One coordinate retained by repeated independent Bernoulli sampling.
// Repetitions of the same coordinate are aggregated into multiplicity.
struct SampledColumn {
    std::size_t source_column;
    std::size_t multiplicity;
    double inverse_probability;
};

struct CoordinateSample {
    std::vector<SampledColumn> columns;
};

}  // namespace ultrahigh_ann
