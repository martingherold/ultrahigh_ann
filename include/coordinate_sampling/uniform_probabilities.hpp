#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace ultrahigh_ann {

// Sum the coordinate inclusion probabilities, i.e. the instance-sensitive
// sampling mass S(C).
[[nodiscard]] double compute_sampling_mass(
    std::span<const double> probabilities);

// Give every coordinate the same probability while preserving S(C). At a
// fixed repetition count T this matches the importance sampler's expected
// multiplicity T * S(C).
[[nodiscard]] std::vector<double>
make_mass_matched_uniform_probabilities(
    std::size_t dimension,
    double sampling_mass);

}  // namespace ultrahigh_ann
