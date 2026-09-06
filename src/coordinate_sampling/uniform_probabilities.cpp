#include "ultrahigh_ann/coordinate_sampling/uniform_probabilities.hpp"

#include "core/finite_values.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace ultrahigh_ann {

double compute_sampling_mass(std::span<const double> probabilities)
{
    double mass = 0.0;
    for (const double probability : probabilities) {
        detail::validate_finite_domain_value(
            probability,
            "sampling probability");
        if (probability < 0.0 || probability > 1.0) {
            throw std::domain_error("invalid sampling probability");
        }
        mass += probability;
    }
    return mass;
}

std::vector<double> make_mass_matched_uniform_probabilities(
    std::size_t dimension,
    double sampling_mass)
{
    detail::validate_finite_domain_value(
        sampling_mass,
        "sampling mass");
    if (sampling_mass < 0.0 ||
        sampling_mass > static_cast<double>(dimension)) {
        throw std::domain_error(
            "sampling mass must be between zero and the dimension");
    }
    if (dimension == 0) {
        return {};
    }
    const double probability =
        sampling_mass / static_cast<double>(dimension);
    return std::vector<double>(dimension, probability);
}

}  // namespace ultrahigh_ann
