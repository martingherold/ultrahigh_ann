#include "coordinate_sampling/uniform_probabilities.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

bool expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
    }
    return condition;
}

}  // namespace

int main()
{
    const std::array<double, 4> importance{
        0.5,
        0.25,
        0.0,
        0.25,
    };
    const double mass = ultrahigh_ann::compute_sampling_mass(importance);
    const auto uniform =
        ultrahigh_ann::make_mass_matched_uniform_probabilities(
            importance.size(),
            mass);

    bool passed = true;
    passed &= expect(
        std::abs(mass - 1.0) < 1.0e-12,
        "sampling mass must equal the sum of probabilities");
    passed &= expect(
        uniform.size() == importance.size(),
        "uniform model must preserve the dimension");
    for (const double probability : uniform) {
        passed &= expect(
            std::abs(probability - 0.25) < 1.0e-12,
            "uniform model must spread the mass equally");
    }
    passed &= expect(
        std::abs(ultrahigh_ann::compute_sampling_mass(uniform) - mass) <
            1.0e-12,
        "uniform model must preserve the sampling mass");

    bool invalid_probability_rejected = false;
    try {
        const std::array<double, 1> invalid{
            std::numeric_limits<double>::quiet_NaN(),
        };
        static_cast<void>(
            ultrahigh_ann::compute_sampling_mass(invalid));
    } catch (const std::domain_error&) {
        invalid_probability_rejected = true;
    }
    passed &= expect(
        invalid_probability_rejected,
        "non-finite probabilities must be rejected");

    bool invalid_mass_rejected = false;
    try {
        static_cast<void>(
            ultrahigh_ann::make_mass_matched_uniform_probabilities(
                2,
                3.0));
    } catch (const std::domain_error&) {
        invalid_mass_rejected = true;
    }
    passed &= expect(
        invalid_mass_rejected,
        "sampling mass above the dimension must be rejected");

    return passed ? 0 : 1;
}
