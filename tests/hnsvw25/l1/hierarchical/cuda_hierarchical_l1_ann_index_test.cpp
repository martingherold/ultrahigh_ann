#include "ultrahigh_ann/hnsvw25/l1/hierarchical/cuda_hierarchical_l1_ann_index.hpp"
#include "ultrahigh_ann/hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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
    using ultrahigh_ann::CudaHierarchicalL1AnnIndex;
    using ultrahigh_ann::CudaHierarchicalL1QueryStrategy;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::HierarchicalL1AnnIndex;
    using ultrahigh_ann::SampledColumn;

    static_assert(!std::is_copy_constructible_v<CudaHierarchicalL1AnnIndex>);
    static_assert(std::is_move_constructible_v<CudaHierarchicalL1AnnIndex>);
    static_assert(!std::is_copy_constructible_v<
                  CudaHierarchicalL1AnnIndex::QueryWorkspace>);
    static_assert(std::is_move_constructible_v<
                  CudaHierarchicalL1AnnIndex::QueryWorkspace>);

    const DenseMatrix representatives(
        std::vector<float>{0.0F, 99.0F, 5.0F, -99.0F}, 2, 2);
    constexpr std::array<double, 2> probabilities{1.0, 0.0};
    constexpr std::size_t repetitions = 16;
    constexpr std::size_t projection_dimension = 6;
    constexpr std::uint64_t seed = 1729;

    if (!ultrahigh_ann::cuda_hierarchical_l1_available()) {
        bool unavailable_threw = false;
        try {
            std::mt19937_64 random_engine(seed);
            const CudaHierarchicalL1AnnIndex index(
                representatives, probabilities, repetitions,
                projection_dimension, random_engine);
            static_cast<void>(index);
        } catch (const std::runtime_error&) {
            unavailable_threw = true;
        }
        return expect(unavailable_threw,
                      "an unavailable CUDA L1 hierarchy must fail explicitly")
                   ? 0
                   : 1;
    }

    std::mt19937_64 cpu_engine(seed);
    const HierarchicalL1AnnIndex cpu_index(representatives, probabilities,
                                           repetitions, projection_dimension,
                                           cpu_engine);
    std::mt19937_64 cuda_engine(seed);
    const CudaHierarchicalL1AnnIndex cuda_index(
        representatives, probabilities, repetitions, projection_dimension,
        cuda_engine);
    auto workspace = cuda_index.make_query_workspace(4);

    bool passed = true;
    passed &=
        expect(cpu_engine == cuda_engine,
               "CPU and CUDA L1 hierarchy construction must consume identical "
               "randomness");
    passed &= expect(cuda_index.device() == 0,
                     "the default hierarchical CUDA L1 device must be zero");
    passed &= expect(!cuda_index.device_name().empty(),
                     "the hierarchical CUDA L1 device name must be reported");
    passed &= expect(workspace.maximum_batch_size() == 4,
                     "the hierarchical CUDA L1 workspace must report capacity");

    const auto usage = cuda_index.space_usage();
    passed &= expect(
        usage.index_payload_bytes ==
            sizeof(SampledColumn) + projection_dimension * sizeof(float) +
                2 * projection_dimension * sizeof(float),
        "the CUDA L1 hierarchy must retain its sample, Cauchy matrix, and "
        "projected representatives");
    passed &= expect(usage.query_workspace_payload_bytes == 0,
                     "batch-dependent CUDA L1 workspace must not be fixed "
                     "index space");
    passed &=
        expect(usage.unique_query_coordinates == 1 &&
                   usage.sampled_multiplicity == repetitions,
               "the CUDA L1 hierarchy must report coordinate sampling usage");
    passed &= expect(
        workspace.payload_bytes() ==
            2 * (4 * sizeof(float)) +
                4 * projection_dimension * sizeof(float) + sizeof(int) +
                4 * 2 * sizeof(float) +
                4 * sizeof(std::size_t) + sizeof(int),
        "the CUDA L1 hierarchy workspace must include sampling, projection, "
        "median, and result buffers");

    constexpr std::array<float, 8> query_values{
        1.0F, 100.0F, 4.0F, -100.0F, 2.5F, 5.0F, 5.0F, 0.0F,
    };
    std::array<std::size_t, 4> direct_results{};
    cuda_index.query_batch(query_values, 4, direct_results, workspace,
                           CudaHierarchicalL1QueryStrategy::direct);
    std::array<std::size_t, 4> gemm_results{};
    cuda_index.query_batch(query_values, 4, gemm_results, workspace,
                           CudaHierarchicalL1QueryStrategy::gemm);
    for (std::size_t query_index = 0; query_index < 4; ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 2, 2);
        const std::size_t expected = cpu_index.query(query);
        passed &=
            expect(direct_results[query_index] == expected,
                   "direct hierarchical CUDA L1 predictions must match the CPU "
                   "index");
        passed &= expect(
            gemm_results[query_index] == expected,
            "SGEMM hierarchical CUDA L1 predictions must match the CPU index");
    }
    passed &= expect(direct_results[2] == 0 && gemm_results[2] == 0,
                     "CUDA L1 hierarchy ties must select the lowest row");

    const std::array<float, 2> non_finite_unsampled{
        1.0F, std::numeric_limits<float>::quiet_NaN()};
    passed &=
        expect(cuda_index.query(non_finite_unsampled, workspace,
                                CudaHierarchicalL1QueryStrategy::direct) ==
                   cpu_index.query(non_finite_unsampled),
               "direct Cauchy projection must ignore unsampled coordinates");
    passed &=
        expect(cuda_index.query(non_finite_unsampled, workspace,
                                CudaHierarchicalL1QueryStrategy::gemm) ==
                   cpu_index.query(non_finite_unsampled),
               "SGEMM Cauchy projection must ignore unsampled coordinates");

    const std::array<float, 2> non_finite_sampled{
        std::numeric_limits<float>::infinity(), 0.0F};
    for (const auto strategy : {CudaHierarchicalL1QueryStrategy::direct,
                                CudaHierarchicalL1QueryStrategy::gemm}) {
        bool finite_error_thrown = false;
        try {
            static_cast<void>(
                cuda_index.query(non_finite_sampled, workspace, strategy));
        } catch (const std::invalid_argument&) {
            finite_error_thrown = true;
        }
        passed &= expect(finite_error_thrown,
                         "CUDA Cauchy projection must reject a non-finite "
                         "sampled coordinate");
    }

    bool capacity_error_thrown = false;
    try {
        std::array<float, 10> queries{};
        std::array<std::size_t, 5> output{};
        cuda_index.query_batch(queries, 5, output, workspace);
    } catch (const std::invalid_argument&) {
        capacity_error_thrown = true;
    }
    passed &= expect(capacity_error_thrown,
                     "a CUDA L1 hierarchy batch beyond capacity must fail");

    bool dimension_error_thrown = false;
    try {
        std::array<float, 3> queries{};
        std::array<std::size_t, 2> output{};
        cuda_index.query_batch(queries, 2, output, workspace);
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &= expect(dimension_error_thrown,
                     "incorrect CUDA L1 hierarchy dimensions must fail");

    std::mt19937_64 other_engine(seed);
    const CudaHierarchicalL1AnnIndex other_index(
        representatives, probabilities, repetitions, projection_dimension,
        other_engine);
    bool ownership_error_thrown = false;
    try {
        static_cast<void>(other_index.query(
            std::span<const float>{query_values}.first(2), workspace));
    } catch (const std::invalid_argument&) {
        ownership_error_thrown = true;
    }
    passed &= expect(ownership_error_thrown,
                     "a workspace from another CUDA L1 hierarchy must fail");

    constexpr std::size_t wide_dimension = 513;
    constexpr std::size_t wide_representative_count = 4;
    constexpr std::size_t wide_projection_dimension = 17;
    std::vector<float> wide_values;
    wide_values.reserve(wide_representative_count * wide_dimension);
    for (std::size_t row = 0; row < wide_representative_count; ++row) {
        for (std::size_t column = 0; column < wide_dimension; ++column) {
            const auto centered_pattern = static_cast<int>(column % 7) - 3;
            wide_values.push_back(2.0F * static_cast<float>(row) +
                                  0.01F * static_cast<float>(centered_pattern));
        }
    }
    const std::vector<float> wide_queries = wide_values;
    const DenseMatrix wide_representatives(
        std::move(wide_values), wide_representative_count, wide_dimension);
    CoordinateSample wide_sample;
    wide_sample.columns.reserve(wide_dimension);
    for (std::size_t column = 0; column < wide_dimension; ++column) {
        wide_sample.columns.push_back(SampledColumn{
            .source_column = column,
            .multiplicity = 1,
            .inverse_probability = 1.0,
        });
    }
    std::mt19937_64 wide_engine(seed);
    const CudaHierarchicalL1AnnIndex wide_index(
        wide_representatives, std::move(wide_sample), wide_projection_dimension,
        wide_engine);
    auto wide_workspace =
        wide_index.make_query_workspace(wide_representative_count);
    std::array<std::size_t, wide_representative_count> wide_direct_results{};
    wide_index.query_batch(wide_queries, wide_representative_count,
                           wide_direct_results, wide_workspace,
                           CudaHierarchicalL1QueryStrategy::direct);
    std::array<std::size_t, wide_representative_count> wide_gemm_results{};
    wide_index.query_batch(wide_queries, wide_representative_count,
                           wide_gemm_results, wide_workspace,
                           CudaHierarchicalL1QueryStrategy::gemm);
    for (std::size_t query_index = 0; query_index < wide_representative_count;
         ++query_index) {
        passed &=
            expect(wide_direct_results[query_index] == query_index,
                   "direct CUDA Cauchy projection must handle more sampled "
                   "coordinates than threads");
        passed &=
            expect(wide_gemm_results[query_index] == query_index,
                   "SGEMM CUDA Cauchy projection must preserve batched layout");
    }

    return passed ? 0 : 1;
}
