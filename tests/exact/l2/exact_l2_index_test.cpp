#include "ultrahigh_ann/core/dense_l2_scan_index.hpp"
#include "ultrahigh_ann/exact/l2/exact_l2_index.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
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
    using ultrahigh_ann::DenseL2ScanIndex;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::ExactL2Index;

    static_assert(!std::is_constructible_v<ExactL2Index, DenseMatrix&&>);
    static_assert(!std::is_constructible_v<ExactL2Index, const DenseMatrix&&>);

    std::vector<float> values{0.0F, 0.0F, 5.0F, 0.0F, 2.0F, 4.0F};
    const DenseMatrix representatives(std::move(values), 3, 2);
    const ExactL2Index index(representatives);
    const DenseL2ScanIndex scan(representatives);

    bool passed = true;
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    passed &=
        expect(space_usage.index_payload_bytes == 6 * sizeof(float),
               "exact index space must include the representative payload");
    passed &= expect(space_usage.query_workspace_payload_bytes == 0,
                     "exact queries must not require a dynamic workspace");
    passed &= expect(space_usage.unique_query_coordinates == 2 &&
                         space_usage.sampled_multiplicity == 2,
                     "exact search must access every query coordinate once");

    const std::array<float, 2> near_first{1.0F, 0.0F};
    passed &= expect(index.query(std::span<const float>{near_first}) == 0,
                     "the nearest representative must be returned");

    const std::array<float, 2> near_third{3.0F, 3.0F};
    passed &= expect(index.query(std::span<const float>{near_third}) == 2,
                     "squared L2 distance must use every coordinate");

    const std::array<float, 2> tied{2.5F, 0.0F};
    passed &= expect(index.query(std::span<const float>{tied}) == 0,
                     "ties must resolve to the lowest representative index");
    passed &=
        expect(scan.query(near_first) == index.query(near_first) &&
                   scan.query(near_third) == index.query(near_third) &&
                   scan.query(tied) == index.query(tied),
               "the exact facade must match the shared dense scan engine");

    constexpr std::array<float, 6> batched_queries{
        1.0F, 0.0F, 3.0F, 3.0F, 2.5F, 0.0F,
    };
    std::array<std::size_t, 3> batched_results{};
    index.query_batch(batched_queries, 3, batched_results);
    passed &= expect(
        batched_results == std::array<std::size_t, 3>{0, 2, 0},
        "exact batch queries must preserve single-query results and ties");
    constexpr std::array strategies{
        CpuDenseL2QueryStrategy::sequential,
        CpuDenseL2QueryStrategy::parallel_queries,
        CpuDenseL2QueryStrategy::parallel_representatives,
        CpuDenseL2QueryStrategy::automatic,
    };
    for (const CpuDenseL2QueryStrategy strategy : strategies) {
        std::array<std::size_t, 3> strategy_results{};
        index.query_batch(batched_queries, 3, strategy_results, strategy);
        passed &= expect(
            strategy_results == batched_results,
            "every exact CPU strategy must preserve batch results and ties");
        passed &=
            expect(index.query(tied, strategy) == 0,
                   "every exact CPU strategy must preserve single-query ties");

        std::array<std::size_t, 3> scan_results{};
        scan.query_batch(batched_queries, 3, scan_results, strategy);
        passed &= expect(scan_results == strategy_results,
                         "the exact facade must forward every CPU strategy");
    }

    bool strategy_error_thrown = false;
    try {
        index.query_batch(batched_queries, 3, batched_results,
                          static_cast<CpuDenseL2QueryStrategy>(999));
    } catch (const std::invalid_argument&) {
        strategy_error_thrown = true;
    }
    passed &= expect(strategy_error_thrown,
                     "an unknown CPU dense L2 strategy must be rejected");

    const std::array<float, 1> wrong_dimension{1.0F};
    bool dimension_error_thrown = false;
    try {
        static_cast<void>(index.query(std::span<const float>{wrong_dimension}));
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &=
        expect(dimension_error_thrown,
               "a query with the wrong dimension must throw invalid_argument");

    const std::array<float, 2> non_finite_query{
        std::numeric_limits<float>::quiet_NaN(), 0.0F};
    bool query_value_error_thrown = false;
    try {
        static_cast<void>(
            index.query(std::span<const float>{non_finite_query}));
    } catch (const std::invalid_argument&) {
        query_value_error_thrown = true;
    }
    passed &= expect(query_value_error_thrown,
                     "a non-finite query value must throw invalid_argument");

    std::vector<float> non_finite_values{
        std::numeric_limits<float>::quiet_NaN(), 0.0F};
    const DenseMatrix non_finite_representatives(std::move(non_finite_values),
                                                 1, 2);
    bool representative_value_error_thrown = false;
    try {
        const ExactL2Index non_finite_index(non_finite_representatives);
        static_cast<void>(non_finite_index);
    } catch (const std::invalid_argument&) {
        representative_value_error_thrown = true;
    }
    passed &= expect(representative_value_error_thrown,
                     "non-finite representatives must throw invalid_argument");

    std::vector<float> empty_values;
    const DenseMatrix empty_representatives(std::move(empty_values), 0, 2);
    bool empty_error_thrown = false;
    try {
        const ExactL2Index empty_index(empty_representatives);
        static_cast<void>(empty_index);
    } catch (const std::invalid_argument&) {
        empty_error_thrown = true;
    }
    passed &=
        expect(empty_error_thrown,
               "an empty representative matrix must throw invalid_argument");

    return passed ? 0 : 1;
}
