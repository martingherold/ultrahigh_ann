#include "ultrahigh_ann/coordinate_sampling/uniform_l1_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/uniform_l2_ann_index.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <random>
#include <span>
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

template <class Index> bool test_direct_uniform_index(std::string_view metric)
{
    std::vector<float> values{
        0.0F,
        0.0F,
        5.0F,
        5.0F,
    };
    const ultrahigh_ann::DenseMatrix representatives(std::move(values), 2, 2);
    constexpr std::size_t repetitions = 4;
    std::mt19937_64 random_engine(42);
    const Index index(representatives, 2.0, repetitions, random_engine);
    const std::array<float, 2> near_first{1.0F, 1.0F};
    const std::array<float, 2> near_second{4.0F, 4.0F};
    const ultrahigh_ann::IndexSpaceUsage usage = index.space_usage();

    bool passed = true;
    passed &= expect(index.query(std::span<const float>{near_first}) == 0,
                     "uniform index must find the first representative");
    passed &= expect(index.query(std::span<const float>{near_second}) == 1,
                     "uniform index must find the second representative");
    passed &= expect(usage.unique_query_coordinates == 2 &&
                         usage.sampled_multiplicity == 2 * repetitions,
                     "uniform index must report direct coordinate sampling");
    if constexpr (std::is_same_v<Index, ultrahigh_ann::UniformL2AnnIndex>) {
        const std::array<float, 4> queries{
            near_first[0],
            near_first[1],
            near_second[0],
            near_second[1],
        };
        std::array<std::size_t, 2> results{};
        index.query_batch(
            queries, 2, results,
            ultrahigh_ann::CpuDenseL2QueryStrategy::parallel_queries);
        passed &= expect(results == std::array<std::size_t, 2>{0, 1},
                         "uniform L2 must forward CPU query parallelism");
        passed &= expect(
            index.query(near_second, ultrahigh_ann::CpuDenseL2QueryStrategy::
                                         parallel_representatives) == 1,
            "uniform L2 must forward CPU representative parallelism");
    }

    std::mt19937_64 empty_engine(42);
    const Index empty_index(representatives, 0.0, repetitions, empty_engine);
    passed &=
        expect(empty_index.query(std::span<const float>{near_second}) == 0,
               "an empty sample must resolve the all-zero distance tie");
    passed &= expect(empty_index.space_usage().unique_query_coordinates == 0 &&
                         empty_index.space_usage().sampled_multiplicity == 0,
                     "an empty sample must report zero coordinate acquisition");
    if (!passed) {
        std::cerr << "metric: " << metric << '\n';
    }
    return passed;
}

}  // namespace

int main()
{
    bool passed = true;
    passed &= test_direct_uniform_index<ultrahigh_ann::UniformL1AnnIndex>("l1");
    passed &= test_direct_uniform_index<ultrahigh_ann::UniformL2AnnIndex>("l2");
    return passed ? 0 : 1;
}
