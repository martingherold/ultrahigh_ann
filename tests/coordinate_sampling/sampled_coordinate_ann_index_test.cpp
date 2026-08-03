#include "hnsvw25/l1/flat/l1_ann_index.hpp"
#include "coordinate_sampling/uniform_l1_ann_index.hpp"
#include "coordinate_sampling/sampled_coordinate_l1_ann_index.hpp"
#include "hnsvw25/l2/flat/l2_ann_index.hpp"
#include "coordinate_sampling/uniform_l2_ann_index.hpp"
#include "coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
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

bool same_space_usage(
    const ultrahigh_ann::IndexSpaceUsage& first,
    const ultrahigh_ann::IndexSpaceUsage& second)
{
    return
        first.index_payload_bytes == second.index_payload_bytes &&
        first.query_workspace_payload_bytes ==
            second.query_workspace_payload_bytes &&
        first.unique_query_coordinates == second.unique_query_coordinates &&
        first.sampled_multiplicity == second.sampled_multiplicity;
}

template<class Backend, class FlatIndex, class UniformIndex>
bool test_facades(std::string_view metric)
{
    std::vector<float> values{
        0.0F, 0.0F, 0.0F,
        5.0F, 5.0F, 5.0F,
    };
    const ultrahigh_ann::DenseMatrix representatives(
        std::move(values),
        2,
        3);
    const std::array<double, 3> importance_probabilities{1.0, 0.25, 0.0};
    constexpr std::size_t repetitions = 7;
    constexpr std::uint64_t seed = 42;

    std::mt19937_64 backend_engine(seed);
    const Backend importance_backend(
        representatives,
        importance_probabilities,
        repetitions,
        backend_engine);
    std::mt19937_64 flat_engine(seed);
    const FlatIndex flat(
        representatives,
        importance_probabilities,
        repetitions,
        flat_engine);

    constexpr double sampling_mass = 1.25;
    const std::array<double, 3> uniform_probabilities{
        sampling_mass / 3.0,
        sampling_mass / 3.0,
        sampling_mass / 3.0,
    };
    std::mt19937_64 uniform_backend_engine(seed);
    const Backend uniform_backend(
        representatives,
        uniform_probabilities,
        repetitions,
        uniform_backend_engine);
    std::mt19937_64 uniform_engine(seed);
    const UniformIndex uniform(
        representatives,
        sampling_mass,
        repetitions,
        uniform_engine);

    const std::array<std::array<float, 3>, 3> queries{{
        {1.0F, 1.0F, 1.0F},
        {4.0F, 4.0F, 4.0F},
        {2.0F, 3.0F, 2.0F},
    }};

    bool passed = true;
    for (const auto& query : queries) {
        passed &= expect(
            flat.query(std::span<const float>{query}) ==
                importance_backend.query(std::span<const float>{query}),
            "flat facade must match the importance-configured backend");
        passed &= expect(
            uniform.query(std::span<const float>{query}) ==
                uniform_backend.query(std::span<const float>{query}),
            "uniform facade must match the uniform-configured backend");
    }
    passed &= expect(
        same_space_usage(flat.space_usage(), importance_backend.space_usage()),
        "flat facade must expose backend space usage");
    passed &= expect(
        same_space_usage(uniform.space_usage(), uniform_backend.space_usage()),
        "uniform facade must expose backend space usage");

    if (!passed) {
        std::cerr << "metric: " << metric << '\n';
    }
    return passed;
}

}  // namespace

int main()
{
    bool passed = true;
    passed &= test_facades<
        ultrahigh_ann::SampledCoordinateL1AnnIndex,
        ultrahigh_ann::FlatL1AnnIndex,
        ultrahigh_ann::UniformL1AnnIndex>("l1");
    passed &= test_facades<
        ultrahigh_ann::SampledCoordinateL2AnnIndex,
        ultrahigh_ann::FlatL2AnnIndex,
        ultrahigh_ann::UniformL2AnnIndex>("l2");
    return passed ? 0 : 1;
}
