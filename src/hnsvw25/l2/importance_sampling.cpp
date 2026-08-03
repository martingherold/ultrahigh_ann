#include "hnsvw25/l2/importance_sampling.hpp"

#include "core/finite_values.hpp"
#include "core/l2_distance.hpp"
#include "coordinate_sampling/coordinate_sampling.hpp"
#include "hnsvw25/l2/importance_sampling_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace ultrahigh_ann {
namespace detail {

std::vector<double> compute_l2_sampling_probabilities(
    const DenseMatrix& representatives)
{
    validate_finite_values(representatives.values(), "representatives");
    std::vector<double> probabilities(representatives.cols(), 0.0);

    for (std::size_t first = 0; first < representatives.rows(); ++first) {
        const auto first_row = representatives.row(first);
        for (std::size_t second = first + 1;
             second < representatives.rows();
             ++second) {
            const auto second_row = representatives.row(second);
            const double squared_distance =
                squared_l2_dist(first_row, second_row);
            if (squared_distance == 0.0) {
                continue;
            }
            const double inverse_distance = 1.0 / squared_distance;
            for (std::size_t column = 0;
                 column < representatives.cols();
                 ++column) {
                const double difference =
                    static_cast<double>(first_row[column]) -
                    static_cast<double>(second_row[column]);
                const double probability = std::min(
                    1.0,
                    difference * difference * inverse_distance);
                probabilities[column] = std::max(
                    probabilities[column],
                    probability);
            }
        }
    }
    return probabilities;
}

}  // namespace detail

std::vector<double> compute_l2_importance_probabilities(
    const DenseMatrix& representatives)
{
    return detail::compute_l2_sampling_probabilities(representatives);
}

CoordinateSample build_l2_importance_sample(
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    return build_coordinate_sample(
        probabilities,
        repetitions,
        random_engine);
}

CoordinateSample build_l2_importance_sample(
    const DenseMatrix& representatives,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    const auto probabilities =
        compute_l2_importance_probabilities(representatives);
    return build_l2_importance_sample(
        probabilities,
        repetitions,
        random_engine);
}

}  // namespace ultrahigh_ann
