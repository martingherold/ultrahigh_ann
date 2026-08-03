#include "coordinate_sampling/coordinate_sampling.hpp"
#include "hnsvw25/l1/importance_sampling_detail.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <string_view>
#include <utility>
#include <vector>

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
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::build_coordinate_sample;
    using ultrahigh_ann::detail::compute_l1_sampling_probabilities;

    std::vector<float> values{
        0.0F, 0.0F,
        5.0F, 0.0F};
    const DenseMatrix representatives(std::move(values), 2, 2);

    const auto probabilities =
        compute_l1_sampling_probabilities(representatives);

    bool passed = true;
    passed &= expect(
        probabilities.size() == 2,
        "one probability must be produced per coordinate");
    passed &= expect(
        probabilities.size() >= 2 &&
            std::abs(probabilities[0] - 1.0) < 1.0e-12,
        "the differing coordinate must have probability one");
    passed &= expect(
        probabilities.size() >= 2 &&
            std::abs(probabilities[1]) < 1.0e-12,
        "the identical coordinate must have probability zero");

    constexpr std::size_t repetitions = 16;
    std::mt19937_64 random_engine(42);
    const auto coordinate_sample = build_coordinate_sample(
        probabilities,
        repetitions,
        random_engine);
    const auto& columns = coordinate_sample.columns;

    passed &= expect(
        columns.size() == 1,
        "exactly one coordinate must be sampled");

    if (columns.size() == 1) {
        passed &= expect(
            columns[0].source_column == 0,
            "the first coordinate must be sampled");
        passed &= expect(
            columns[0].multiplicity == repetitions,
            "a probability-one coordinate must occur in every repetition");
        passed &= expect(
            std::abs(columns[0].inverse_probability - 1.0) < 1.0e-12,
            "a probability-one coordinate must have inverse probability one");
    }

    return passed ? 0 : 1;
}
