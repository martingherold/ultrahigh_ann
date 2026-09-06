#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l1_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l1_ann_index.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"
#include "ultrahigh_ann/hnsvw25/l1/flat/cuda_l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/flat/l1_ann_index.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
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

} // namespace

int main()
{
#if defined(ULTRAHIGH_ANN_TEST_REQUIRES_CUDA_DEVICE)
    if (!ultrahigh_ann::cuda_runtime_info().has_value()) {
        std::cout << "CUDA build has no usable device\n";
        return 77;
    }
#endif
    using ultrahigh_ann::CoordinateSample;
    using ultrahigh_ann::CudaFlatL1AnnIndex;
    using ultrahigh_ann::CudaSampledCoordinateL1AnnIndex;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::FlatL1AnnIndex;
    using ultrahigh_ann::SampledColumn;
    using ultrahigh_ann::SampledCoordinateL1AnnIndex;

    static_assert(
        !std::is_copy_constructible_v<CudaSampledCoordinateL1AnnIndex>);
    static_assert(
        std::is_move_constructible_v<CudaSampledCoordinateL1AnnIndex>);

    const DenseMatrix representatives(
        std::vector<float>{
            0.0F,
            99.0F,
            0.0F,
            -5.0F,
            4.0F,
            -99.0F,
            1.0F,
            -5.0F,
            2.0F,
            0.0F,
            5.0F,
            -5.0F,
        },
        3, 4);
    const CoordinateSample sample{
        .columns =
            {
                SampledColumn{
                    .source_column = 0,
                    .multiplicity = 2,
                    .inverse_probability = 0.5,
                },
                SampledColumn{
                    .source_column = 2,
                    .multiplicity = 1,
                    .inverse_probability = 4.0,
                },
            },
    };

    if (!ultrahigh_ann::cuda_sampled_coordinate_l1_available()) {
        bool unavailable_threw = false;
        try {
            const CudaSampledCoordinateL1AnnIndex index(representatives,
                                                        sample);
            static_cast<void>(index);
        } catch (const std::runtime_error&) {
            unavailable_threw = true;
        }
        return expect(unavailable_threw,
                      "an unavailable CUDA sampled L1 backend must fail "
                      "explicitly")
                   ? 0
                   : 1;
    }

    const SampledCoordinateL1AnnIndex cpu_index(representatives, sample);
    const CudaSampledCoordinateL1AnnIndex cuda_index(representatives, sample);
    auto workspace = cuda_index.make_query_workspace(4);

    bool passed = true;
    passed &= expect(cuda_index.device() == 0,
                     "the default sampled CUDA L1 device must be zero");
    passed &= expect(!cuda_index.device_name().empty(),
                     "the sampled CUDA L1 device name must be reported");
    passed &= expect(workspace.maximum_batch_size() == 4,
                     "the sampled CUDA L1 workspace must report capacity");

    const auto usage = cuda_index.space_usage();
    passed &= expect(
        usage.index_payload_bytes ==
            3 * 2 * sizeof(float) + 2 * sizeof(float) + 2 * sizeof(std::size_t),
        "the sampled CUDA L1 index must retain representatives, weights, and "
        "source columns");
    passed &= expect(
        usage.unique_query_coordinates == 2 && usage.sampled_multiplicity == 3,
        "the sampled CUDA L1 index must report coordinate sampling usage");
    passed &= expect(
        workspace.payload_bytes() == 2 * (4 * 2 * sizeof(float)) +
                                         4 * 3 * sizeof(float) +
                                         4 * sizeof(std::size_t),
        "the sampled CUDA L1 workspace must include host packing and device "
        "buffers");

    constexpr std::array<float, 16> query_values{
        0.5F, 10.0F, 0.25F, 20.0F, 3.8F, -10.0F, 1.1F, 20.0F,
        2.0F, 10.0F, 0.5F,  20.0F, 2.0F, -10.0F, 5.0F, 20.0F,
    };
    std::array<std::size_t, 4> results{};
    cuda_index.query_batch(query_values, 4, results, workspace);
    for (std::size_t query_index = 0; query_index < results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 4, 4);
        passed &= expect(results[query_index] == cpu_index.query(query),
                         "sampled CUDA L1 predictions must match the CPU "
                         "backend on this well-conditioned fixture");
    }
    passed &= expect(results[2] == 0,
                     "sampled CUDA L1 ties must select the lowest row");

    const std::array<float, 4> non_finite_unsampled{
        0.5F,
        std::numeric_limits<float>::quiet_NaN(),
        0.25F,
        std::numeric_limits<float>::infinity(),
    };
    passed &= expect(cuda_index.query(non_finite_unsampled, workspace) ==
                         cpu_index.query(non_finite_unsampled),
                     "unsampled query coordinates must not be transferred or "
                     "validated");

    const CoordinateSample empty_sample;
    const CudaSampledCoordinateL1AnnIndex empty_index(representatives,
                                                      empty_sample);
    auto empty_workspace = empty_index.make_query_workspace(2);
    std::array<std::size_t, 2> empty_results{};
    empty_index.query_batch(std::span<const float>{query_values}.first(8), 2,
                            empty_results, empty_workspace);
    passed &= expect(empty_results[0] == 0 && empty_results[1] == 0,
                     "an empty L1 coordinate sample must return row zero");

    constexpr std::array<double, 4> probabilities{1.0, 0.0, 1.0, 0.0};
    constexpr std::size_t repetitions = 2;
    std::mt19937_64 cpu_engine(42);
    const FlatL1AnnIndex cpu_flat(representatives, probabilities, repetitions,
                                  cpu_engine);
    std::mt19937_64 cuda_engine(42);
    const CudaFlatL1AnnIndex cuda_flat(representatives, probabilities,
                                       repetitions, cuda_engine);
    auto flat_workspace = cuda_flat.make_query_workspace(4);
    std::array<std::size_t, 4> flat_results{};
    cuda_flat.query_batch(query_values, 4, flat_results, flat_workspace);
    for (std::size_t query_index = 0; query_index < flat_results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 4, 4);
        passed &= expect(flat_results[query_index] == cpu_flat.query(query),
                         "the CUDA flat L1 facade must match the CPU facade");
    }

    std::mt19937_64 expected_convenience_engine(77);
    const FlatL1AnnIndex expected_convenience_flat(
        representatives, repetitions, expected_convenience_engine);
    std::mt19937_64 cuda_convenience_engine(77);
    const CudaFlatL1AnnIndex cuda_convenience_flat(
        representatives, repetitions, cuda_convenience_engine);
    auto convenience_workspace =
        cuda_convenience_flat.make_query_workspace(1);
    passed &= expect(
        cuda_convenience_flat.query(
            std::span<const float>{query_values}.first(4),
            convenience_workspace) ==
            expected_convenience_flat.query(
                std::span<const float>{query_values}.first(4)),
        "the CUDA flat L1 convenience constructor must use the CPU "
        "importance model");

    bool capacity_error_thrown = false;
    try {
        std::array<std::size_t, 5> output{};
        std::array<float, 20> queries{};
        cuda_index.query_batch(queries, 5, output, workspace);
    } catch (const std::invalid_argument&) {
        capacity_error_thrown = true;
    }
    passed &= expect(capacity_error_thrown,
                     "a batch exceeding sampled CUDA L1 workspace capacity "
                     "must be rejected");

    bool dimension_error_thrown = false;
    try {
        std::array<std::size_t, 2> output{};
        const std::array<float, 7> queries{};
        cuda_index.query_batch(queries, 2, output, workspace);
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &= expect(dimension_error_thrown,
                     "incorrect sampled CUDA L1 query dimensions must be "
                     "rejected");

    const CudaSampledCoordinateL1AnnIndex other_index(representatives, sample);
    bool ownership_error_thrown = false;
    try {
        static_cast<void>(other_index.query(
            std::span<const float>{query_values}.first(4), workspace));
    } catch (const std::invalid_argument&) {
        ownership_error_thrown = true;
    }
    passed &= expect(ownership_error_thrown,
                     "a workspace from another sampled CUDA L1 index must be "
                     "rejected");

    bool invalid_weight_thrown = false;
    try {
        CoordinateSample invalid_sample{
            .columns =
                {
                    SampledColumn{
                        .source_column = 0,
                        .multiplicity = 0,
                        .inverse_probability = 1.0,
                    },
                },
        };
        const CudaSampledCoordinateL1AnnIndex invalid_index(
            representatives, std::move(invalid_sample));
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        invalid_weight_thrown = true;
    }
    passed &= expect(invalid_weight_thrown,
                     "invalid sampled CUDA L1 weights must be rejected");

    return passed ? 0 : 1;
}
