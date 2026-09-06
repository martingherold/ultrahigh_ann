#include "ultrahigh_ann/coordinate_sampling/cuda_sampled_coordinate_l2_ann_index.hpp"
#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"
#include "ultrahigh_ann/hnsvw25/l2/flat/cuda_l2_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l2/flat/l2_ann_index.hpp"

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

}  // namespace

int main()
{
#if defined(ULTRAHIGH_ANN_TEST_REQUIRES_CUDA_DEVICE)
    if (!ultrahigh_ann::cuda_runtime_info().has_value()) {
        std::cout << "CUDA build has no usable device\n";
        return 77;
    }
#endif
    using ultrahigh_ann::CoordinateSample;
    using ultrahigh_ann::CudaFlatL2AnnIndex;
    using ultrahigh_ann::CudaSampledCoordinateL2AnnIndex;
    using ultrahigh_ann::CudaSampledCoordinateL2QueryStrategy;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::FlatL2AnnIndex;
    using ultrahigh_ann::SampledColumn;
    using ultrahigh_ann::SampledCoordinateL2AnnIndex;

    static_assert(
        !std::is_copy_constructible_v<CudaSampledCoordinateL2AnnIndex>);
    static_assert(
        std::is_move_constructible_v<CudaSampledCoordinateL2AnnIndex>);

    std::vector<float> representative_values{
        0.0F, 99.0F, 0.0F, -5.0F, 4.0F, -99.0F,
        1.0F, -5.0F, 2.0F, 0.0F,  5.0F, -5.0F,
    };
    const DenseMatrix representatives(std::move(representative_values), 3, 4);
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

    if (!ultrahigh_ann::cuda_sampled_coordinate_l2_available()) {
        bool unavailable_threw = false;
        try {
            const CudaSampledCoordinateL2AnnIndex index(representatives,
                                                        sample);
            static_cast<void>(index);
        } catch (const std::runtime_error&) {
            unavailable_threw = true;
        }
        return expect(
                   unavailable_threw,
                   "an unavailable CUDA sampled backend must fail explicitly")
                   ? 0
                   : 1;
    }

    const SampledCoordinateL2AnnIndex cpu_index(representatives, sample);
    const CudaSampledCoordinateL2AnnIndex cuda_index(representatives, sample);
    auto workspace = cuda_index.make_query_workspace(4);

    bool passed = true;
    passed &= expect(cuda_index.device() == 0,
                     "the default sampled CUDA device must be zero");
    passed &= expect(!cuda_index.device_name().empty(),
                     "the sampled CUDA device name must be reported");
    passed &= expect(workspace.maximum_batch_size() == 4,
                     "the sampled workspace must report its capacity");

    const auto usage = cuda_index.space_usage();
    passed &= expect(
        usage.index_payload_bytes == 3 * 2 * sizeof(float) + 3 * sizeof(float) +
                                         2 * sizeof(std::size_t) +
                                         2 * sizeof(float),
        "the sampled CUDA index must retain its transformed matrix, norms, "
        "source columns, and scales");
    passed &= expect(
        usage.unique_query_coordinates == 2 && usage.sampled_multiplicity == 3,
        "the sampled CUDA index must report coordinate sampling usage");
    passed &= expect(
        workspace.payload_bytes() ==
            2 * (4 * 2 * sizeof(float)) + 4 * 3 * sizeof(float) +
                4 * sizeof(float) + 4 * sizeof(std::size_t),
        "the sampled workspace must include host packing and CUDA buffers");

    constexpr std::array<float, 16> query_values{
        0.5F, 10.0F, 0.25F, 20.0F, 3.8F, -10.0F, 1.1F, 20.0F,
        2.0F, 10.0F, 0.5F,  20.0F, 2.0F, -10.0F, 5.0F, 20.0F,
    };
    std::array<std::size_t, 4> direct_results{};
    cuda_index.query_batch(query_values, 4, direct_results, workspace,
                           CudaSampledCoordinateL2QueryStrategy::direct);

    std::array<std::size_t, 4> gemm_results{};
    cuda_index.query_batch(query_values, 4, gemm_results, workspace,
                           CudaSampledCoordinateL2QueryStrategy::gemm);

    for (std::size_t query_index = 0; query_index < 4; ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 4, 4);
        const std::size_t expected = cpu_index.query(query);
        passed &= expect(
            direct_results[query_index] == expected,
            "direct sampled CUDA predictions must match the CPU backend");
        passed &= expect(
            gemm_results[query_index] == expected,
            "GEMM sampled CUDA predictions must match this well-conditioned "
            "CPU fixture");
    }
    passed &= expect(direct_results[2] == 0 && gemm_results[2] == 0,
                     "sampled CUDA ties must resolve to the lowest row index");

    const std::array<float, 4> non_finite_unsampled{
        0.5F,
        std::numeric_limits<float>::quiet_NaN(),
        0.25F,
        std::numeric_limits<float>::infinity(),
    };
    passed &= expect(
        cuda_index.query(non_finite_unsampled, workspace,
                         CudaSampledCoordinateL2QueryStrategy::gemm) ==
            cpu_index.query(non_finite_unsampled),
        "unsampled query coordinates must not be transferred or validated");

    const CoordinateSample empty_sample;
    const CudaSampledCoordinateL2AnnIndex empty_index(representatives,
                                                      empty_sample);
    auto empty_workspace = empty_index.make_query_workspace(2);
    std::array<std::size_t, 2> empty_results{};
    empty_index.query_batch(std::span<const float>{query_values}.first(8), 2,
                            empty_results, empty_workspace,
                            CudaSampledCoordinateL2QueryStrategy::gemm);
    passed &= expect(
        empty_results[0] == 0 && empty_results[1] == 0,
        "an empty coordinate sample must deterministically return row zero");

    constexpr std::array<double, 4> probabilities{1.0, 0.0, 1.0, 0.0};
    constexpr std::size_t repetitions = 2;
    std::mt19937_64 cpu_engine(42);
    const FlatL2AnnIndex cpu_flat(representatives, probabilities, repetitions,
                                  cpu_engine);
    std::mt19937_64 cuda_engine(42);
    const CudaFlatL2AnnIndex cuda_flat(representatives, probabilities,
                                       repetitions, cuda_engine);
    auto flat_workspace = cuda_flat.make_query_workspace(4);
    std::array<std::size_t, 4> flat_results{};
    cuda_flat.query_batch(query_values, 4, flat_results, flat_workspace,
                          CudaSampledCoordinateL2QueryStrategy::gemm);
    for (std::size_t query_index = 0; query_index < 4; ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 4, 4);
        passed &= expect(flat_results[query_index] == cpu_flat.query(query),
                         "the CUDA flat facade must match the CPU flat facade");
    }

    const auto gpu_probability_result =
        ultrahigh_ann::compute_l2_importance_probabilities_gpu(representatives,
                                                               0, 128);
    std::mt19937_64 expected_gpu_policy_engine(77);
    const FlatL2AnnIndex expected_gpu_policy_flat(
        representatives, gpu_probability_result.probabilities, repetitions,
        expected_gpu_policy_engine);
    std::mt19937_64 default_gpu_policy_engine(77);
    const CudaFlatL2AnnIndex default_gpu_policy_flat(
        representatives, repetitions, default_gpu_policy_engine);
    auto default_gpu_policy_workspace =
        default_gpu_policy_flat.make_query_workspace(1);
    passed &=
        expect(default_gpu_policy_flat.query(
                   std::span<const float>{query_values}.first(4),
                   default_gpu_policy_workspace) ==
                   expected_gpu_policy_flat.query(
                       std::span<const float>{query_values}.first(4)),
               "the CUDA flat convenience constructor must default to GPU FP32 "
               "probabilities on its selected device");

    std::mt19937_64 expected_sequential_engine(91);
    const FlatL2AnnIndex expected_sequential_flat(representatives, repetitions,
                                                  expected_sequential_engine);
    std::mt19937_64 explicit_sequential_engine(91);
    const CudaFlatL2AnnIndex explicit_sequential_flat(
        representatives, repetitions, explicit_sequential_engine, 0,
        ultrahigh_ann::ExecutionPolicy::sequential);
    auto explicit_sequential_workspace =
        explicit_sequential_flat.make_query_workspace(1);
    passed &=
        expect(explicit_sequential_flat.query(
                   std::span<const float>{query_values}.first(4),
                   explicit_sequential_workspace) ==
                   expected_sequential_flat.query(
                       std::span<const float>{query_values}.first(4)),
               "the CUDA flat convenience constructor must honor an explicit "
               "probability policy override");

    bool capacity_error_thrown = false;
    try {
        std::array<std::size_t, 5> output{};
        std::array<float, 20> queries{};
        cuda_index.query_batch(queries, 5, output, workspace);
    } catch (const std::invalid_argument&) {
        capacity_error_thrown = true;
    }
    passed &=
        expect(capacity_error_thrown,
               "a batch exceeding sampled workspace capacity must be rejected");

    bool dimension_error_thrown = false;
    try {
        std::array<std::size_t, 2> output{};
        const std::array<float, 7> queries{};
        cuda_index.query_batch(queries, 2, output, workspace);
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &= expect(dimension_error_thrown,
                     "incorrect sampled query dimensions must be rejected");

    const CudaSampledCoordinateL2AnnIndex other_index(representatives, sample);
    bool ownership_error_thrown = false;
    try {
        static_cast<void>(other_index.query(
            std::span<const float>{query_values}.first(4), workspace));
    } catch (const std::invalid_argument&) {
        ownership_error_thrown = true;
    }
    passed &=
        expect(ownership_error_thrown,
               "a workspace from another sampled CUDA index must be rejected");

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
        const CudaSampledCoordinateL2AnnIndex invalid_index(
            representatives, std::move(invalid_sample));
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        invalid_weight_thrown = true;
    }
    passed &= expect(invalid_weight_thrown,
                     "invalid sampled coordinate weights must be rejected");

    return passed ? 0 : 1;
}
