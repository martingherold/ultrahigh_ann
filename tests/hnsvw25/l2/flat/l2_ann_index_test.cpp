#include "ultrahigh_ann/hnsvw25/l2/flat/l2_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

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
    using ultrahigh_ann::CpuDenseL2QueryStrategy;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::FlatL2AnnIndex;

    std::vector<float> values{0.0F, 0.0F, 5.0F, 0.0F};
    const DenseMatrix representatives(std::move(values), 2, 2);

    constexpr std::size_t repetitions = 16;
    std::mt19937_64 random_engine(42);
    const FlatL2AnnIndex index(representatives, repetitions, random_engine);
    const auto probabilities =
        ultrahigh_ann::compute_l2_importance_probabilities(representatives);
    std::mt19937_64 cached_random_engine(42);
    const FlatL2AnnIndex cached_index(representatives, probabilities,
                                      repetitions, cached_random_engine);

    const std::array<float, 2> near_first{1.0F, 100.0F};
    const std::array<float, 2> near_second{4.0F, -100.0F};

    bool passed = true;
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    passed &= expect(
        space_usage.index_payload_bytes ==
            sizeof(std::size_t) + sizeof(float) + 2 * sizeof(float),
        "flat index space must include the reduced values, source columns, "
        "and scales");
    passed &= expect(space_usage.query_workspace_payload_bytes == sizeof(float),
                     "flat space must include one packed sampled query");
    passed &= expect(space_usage.unique_query_coordinates == 1 &&
                         space_usage.sampled_multiplicity == repetitions,
                     "flat space usage must report its sampled coordinates");
    passed &=
        expect(index.query(std::span<const float>{near_first}) == 0,
               "query near the first representative must return index zero");
    passed &=
        expect(index.query(std::span<const float>{near_second}) == 1,
               "query near the second representative must return index one");
    passed &= expect(
        cached_index.query(std::span<const float>{near_first}) ==
                index.query(std::span<const float>{near_first}) &&
            cached_index.query(std::span<const float>{near_second}) ==
                index.query(std::span<const float>{near_second}),
        "a precomputed probability model must reproduce the legacy index");

    const std::array<float, 4> batched_queries{
        near_first[0],
        near_first[1],
        near_second[0],
        near_second[1],
    };
    std::array<std::size_t, 2> batched_results{};
    index.query_batch(batched_queries, 2, batched_results);
    passed &= expect(
        batched_results[0] == index.query(near_first) &&
            batched_results[1] == index.query(near_second),
        "flat batch queries must use the same reduced scan as single queries");
    std::array<std::size_t, 2> parallel_results{};
    index.query_batch(batched_queries, 2, parallel_results,
                      CpuDenseL2QueryStrategy::parallel_queries);
    passed &= expect(parallel_results == batched_results,
                     "the flat facade must forward CPU query parallelism");
    passed &= expect(
        index.query(near_second,
                    CpuDenseL2QueryStrategy::parallel_representatives) == 1,
        "the flat facade must forward CPU representative parallelism");

    bool probability_dimension_rejected = false;
    try {
        std::mt19937_64 invalid_engine(42);
        const std::array<double, 1> short_probabilities{1.0};
        const FlatL2AnnIndex invalid_index(representatives, short_probabilities,
                                           repetitions, invalid_engine);
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        probability_dimension_rejected = true;
    }
    passed &= expect(probability_dimension_rejected,
                     "probability and input dimensions must match");

    const std::array<float, 2> non_finite_sampled_coordinate{
        std::numeric_limits<float>::quiet_NaN(), 0.0F};
    bool value_error_thrown = false;
    try {
        static_cast<void>(
            index.query(std::span<const float>{non_finite_sampled_coordinate}));
    } catch (const std::invalid_argument&) {
        value_error_thrown = true;
    }
    passed &=
        expect(value_error_thrown,
               "a non-finite sampled coordinate must throw invalid_argument");

    return passed ? 0 : 1;
}
