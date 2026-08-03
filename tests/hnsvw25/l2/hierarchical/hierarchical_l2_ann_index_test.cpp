#include "hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.hpp"
#include "hnsvw25/l2/importance_sampling.hpp"

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
    using ultrahigh_ann::HierarchicalL2AnnIndex;

    std::vector<float> values{
        0.0F, 0.0F,
        5.0F, 0.0F};
    const DenseMatrix representatives(std::move(values), 2, 2);

    constexpr std::size_t repetitions = 1;
    constexpr std::size_t projection_dimension = 5;
    std::mt19937_64 random_engine(42);
    const HierarchicalL2AnnIndex index(
        representatives,
        repetitions,
        projection_dimension,
        random_engine);
    const auto probabilities =
        ultrahigh_ann::compute_l2_importance_probabilities(
            representatives);
    std::mt19937_64 cached_random_engine(42);
    const HierarchicalL2AnnIndex cached_index(
        representatives,
        probabilities,
        repetitions,
        projection_dimension,
        cached_random_engine);

    const std::array<float, 2> near_first{1.0F, 100.0F};
    const std::array<float, 2> near_second{4.0F, -100.0F};
    auto workspace = index.make_query_workspace();

    bool passed = true;
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    passed &= expect(
        space_usage.index_payload_bytes ==
            sizeof(ultrahigh_ann::SampledColumn) +
                projection_dimension * sizeof(double) +
                2 * projection_dimension * sizeof(double),
        "hierarchical space must include sampled, projection, and reduced data");
    passed &= expect(
        space_usage.query_workspace_payload_bytes ==
            sizeof(float) + projection_dimension * sizeof(double),
        "hierarchical space must include one reusable query workspace");
    passed &= expect(
        space_usage.unique_query_coordinates == 1 &&
            space_usage.sampled_multiplicity == repetitions,
        "hierarchical space usage must report sampled coordinates");
    passed &= expect(
        index.query(std::span<const float>{near_first}, workspace) == 0,
        "query near the first representative must return index zero");
    passed &= expect(
        index.query(std::span<const float>{near_second}, workspace) == 1,
        "query near the second representative must return index one");
    passed &= expect(
        index.query(std::span<const float>{near_first}) == 0,
        "the convenience query overload must return the same result");
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
        static_cast<void>(index.query(
            std::span<const float>{non_finite_sampled_coordinate},
            workspace));
    } catch (const std::invalid_argument&) {
        value_error_thrown = true;
    }
    passed &= expect(
        value_error_thrown,
        "a non-finite sampled coordinate must throw invalid_argument");

    std::mt19937_64 invalid_engine(123);
    const std::mt19937_64 unchanged_engine = invalid_engine;
    bool projection_error_thrown = false;
    try {
        const HierarchicalL2AnnIndex invalid_index(
            representatives,
            repetitions,
            0,
            invalid_engine);
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        projection_error_thrown = true;
    }
    passed &= expect(
        projection_error_thrown,
        "a zero projection dimension must throw invalid_argument");
    passed &= expect(
        invalid_engine == unchanged_engine,
        "invalid arguments must be rejected before consuming randomness");

    return passed ? 0 : 1;
}
