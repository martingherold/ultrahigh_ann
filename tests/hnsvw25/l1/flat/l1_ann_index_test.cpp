#include "ultrahigh_ann/hnsvw25/l1/flat/l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
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
    using ultrahigh_ann::FlatL1AnnIndex;

    std::vector<float> values{
        0.0F, 0.0F,
        5.0F, 0.0F};
    const DenseMatrix representatives(std::move(values), 2, 2);

    constexpr std::size_t repetitions = 16;
    std::mt19937_64 random_engine(42);
    const FlatL1AnnIndex index(
        representatives,
        repetitions,
        random_engine);
    const auto probabilities =
        ultrahigh_ann::compute_l1_importance_probabilities(
            representatives);
    std::mt19937_64 cached_random_engine(42);
    const FlatL1AnnIndex cached_index(
        representatives,
        probabilities,
        repetitions,
        cached_random_engine);

    const std::array<float, 2> near_first{1.0F, 100.0F};
    const std::array<float, 2> near_second{4.0F, -100.0F};

    bool passed = true;
    const ultrahigh_ann::IndexSpaceUsage space_usage =
        index.space_usage();
    passed &= expect(
        space_usage.index_payload_bytes ==
            sizeof(ultrahigh_ann::SampledColumn) + 2 * sizeof(float),
        "flat index space must include sampled columns and values");
    passed &= expect(
        space_usage.query_workspace_payload_bytes == 0,
        "flat queries must not require a dynamic workspace");
    passed &= expect(
        space_usage.unique_query_coordinates == 1 &&
            space_usage.sampled_multiplicity == repetitions,
        "flat space usage must report its sampled coordinates");
    passed &= expect(
        index.query(std::span<const float>{near_first}) == 0,
        "query near the first representative must return index zero");
    passed &= expect(
        index.query(std::span<const float>{near_second}) == 1,
        "query near the second representative must return index one");
    passed &= expect(
        cached_index.query(std::span<const float>{near_first}) ==
                index.query(std::span<const float>{near_first}) &&
            cached_index.query(std::span<const float>{near_second}) ==
                index.query(std::span<const float>{near_second}),
        "a precomputed probability model must reproduce the legacy index");

    const std::array<float, 2> non_finite_sampled_coordinate{
        std::numeric_limits<float>::quiet_NaN(),
        0.0F};
    bool value_error_thrown = false;
    try {
        static_cast<void>(
            index.query(
                std::span<const float>{
                    non_finite_sampled_coordinate}));
    } catch (const std::invalid_argument&) {
        value_error_thrown = true;
    }
    passed &= expect(
        value_error_thrown,
        "a non-finite sampled coordinate must throw invalid_argument");

    return passed ? 0 : 1;
}
