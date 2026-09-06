#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"

#include "core/finite_values.hpp"

#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ultrahigh_ann {

CoordinateSample build_coordinate_sample(
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    std::vector<SampledColumn> columns;
    for (std::size_t column = 0; column < probabilities.size(); ++column) {
        const double probability = probabilities[column];
        detail::validate_finite_domain_value(
            probability,
            "sampling probability");
        if (probability < 0.0 || probability > 1.0) {
            throw std::domain_error("invalid sampling probability");
        }
        if (probability == 0.0 || repetitions == 0) {
            continue;
        }

        std::binomial_distribution<std::size_t> distribution(
            repetitions,
            probability);
        const std::size_t multiplicity = distribution(random_engine);
        if (multiplicity != 0) {
            columns.push_back(SampledColumn{
                .source_column = column,
                .multiplicity = multiplicity,
                .inverse_probability = 1.0 / probability,
            });
        }
    }
    return CoordinateSample{std::move(columns)};
}

}  // namespace ultrahigh_ann
