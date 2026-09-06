#include "ultrahigh_ann/core/cuda_dense_l1_scan_index.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"
#include "ultrahigh_ann/exact/l1/cuda_exact_l1_index.hpp"
#include "ultrahigh_ann/exact/l1/exact_l1_index.hpp"

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

} // namespace

int main()
{
#if defined(ULTRAHIGH_ANN_TEST_REQUIRES_CUDA_DEVICE)
    if (!ultrahigh_ann::cuda_runtime_info().has_value()) {
        std::cout << "CUDA build has no usable device\n";
        return 77;
    }
#endif
    using ultrahigh_ann::CudaDenseL1ScanIndex;
    using ultrahigh_ann::CudaExactL1Index;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::ExactL1Index;

    static_assert(!std::is_copy_constructible_v<CudaExactL1Index>);
    static_assert(std::is_move_constructible_v<CudaExactL1Index>);
    static_assert(!std::is_constructible_v<CudaExactL1Index, DenseMatrix&&>);
    static_assert(!std::is_copy_constructible_v<CudaDenseL1ScanIndex>);
    static_assert(std::is_move_constructible_v<CudaDenseL1ScanIndex>);
    static_assert(
        !std::is_constructible_v<CudaDenseL1ScanIndex, DenseMatrix&&>);
    static_assert(std::is_same_v<CudaExactL1Index::QueryWorkspace,
                                 CudaDenseL1ScanIndex::QueryWorkspace>);

    const DenseMatrix representatives(
        std::vector<float>{0.0F, 0.0F, 5.0F, 0.0F, 2.0F, 4.0F}, 3, 2);

    if (!ultrahigh_ann::cuda_exact_l1_available()) {
        bool exact_unavailable_threw = false;
        try {
            const CudaExactL1Index index(representatives);
            static_cast<void>(index);
        } catch (const std::runtime_error&) {
            exact_unavailable_threw = true;
        }
        bool dense_unavailable_threw = false;
        try {
            const CudaDenseL1ScanIndex scan(representatives);
            static_cast<void>(scan);
        } catch (const std::runtime_error&) {
            dense_unavailable_threw = true;
        }
        return expect(exact_unavailable_threw && dense_unavailable_threw,
                      "unavailable CUDA L1 facades and engines must fail "
                      "explicitly")
                   ? 0
                   : 1;
    }

    const ExactL1Index cpu_index(representatives);
    const CudaExactL1Index cuda_index(representatives);
    const CudaDenseL1ScanIndex cuda_scan(representatives);
    auto workspace = cuda_index.make_query_workspace(4);
    auto scan_workspace = cuda_scan.make_query_workspace(4);

    bool passed = true;
    passed &= expect(cuda_index.device() == 0,
                     "the default CUDA L1 device must be zero");
    passed &= expect(!cuda_index.device_name().empty(),
                     "the CUDA L1 device name must be reported");
    passed &= expect(workspace.maximum_batch_size() == 4,
                     "the CUDA L1 workspace must report its capacity");
    passed &= expect(cuda_index.space_usage().index_payload_bytes ==
                         6 * sizeof(float),
                     "the CUDA L1 index must retain its representatives");
    passed &= expect(
        workspace.payload_bytes() == 4 * 2 * sizeof(float) +
                                         4 * 3 * sizeof(float) +
                                         4 * sizeof(std::size_t),
        "the CUDA L1 workspace must report queries, distances, and results");

    constexpr std::array<float, 8> query_values{
        1.0F, 0.0F, 3.0F, 3.0F, 2.5F, 0.0F, 5.0F, 0.0F,
    };
    std::array<std::size_t, 4> cuda_results{};
    cuda_index.query_batch(query_values, 4, cuda_results, workspace);
    for (std::size_t query_index = 0; query_index < cuda_results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 2, 2);
        passed &= expect(cuda_results[query_index] == cpu_index.query(query),
                         "batched CUDA predictions must match exact L1 on "
                         "this well-conditioned fixture");
    }
    passed &= expect(cuda_results[2] == 0,
                     "CUDA L1 ties must resolve to the lowest row index");

    std::array<std::size_t, 4> dense_results{};
    cuda_scan.query_batch(query_values, 4, dense_results, scan_workspace);
    passed &= expect(dense_results == cuda_results,
                     "the exact facade and dense CUDA L1 engine must agree");
    passed &=
        expect(cuda_index.query(std::span<const float>{query_values}.first(2),
                                workspace) == 0,
               "the single-query CUDA L1 API must return the nearest row");

    const std::array<float, 2> weights{0.5F, 4.0F};
    const CudaDenseL1ScanIndex weighted_scan(representatives, weights);
    auto weighted_workspace = weighted_scan.make_query_workspace(1);
    constexpr std::array<float, 2> weighted_query{3.0F, 1.0F};
    passed &=
        expect(weighted_scan.query(weighted_query, weighted_workspace) == 1,
               "the weighted CUDA L1 scan must apply coordinate "
               "weights after absolute differences");
    passed &=
        expect(weighted_scan.space_usage().index_payload_bytes ==
                   6 * sizeof(float) + 2 * sizeof(float),
               "weighted CUDA L1 space must include device coordinate weights");

    bool capacity_error_thrown = false;
    try {
        std::array<std::size_t, 5> output{};
        std::array<float, 10> queries{};
        cuda_index.query_batch(queries, 5, output, workspace);
    } catch (const std::invalid_argument&) {
        capacity_error_thrown = true;
    }
    passed &= expect(capacity_error_thrown,
                     "a batch exceeding CUDA L1 workspace capacity must be "
                     "rejected");

    bool dimension_error_thrown = false;
    try {
        std::array<std::size_t, 2> output{};
        const std::array<float, 3> queries{};
        cuda_index.query_batch(queries, 2, output, workspace);
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &= expect(dimension_error_thrown,
                     "incorrect CUDA L1 query dimensions must be rejected");

    bool output_error_thrown = false;
    try {
        std::array<std::size_t, 1> output{};
        cuda_index.query_batch(query_values, 4, output, workspace);
    } catch (const std::invalid_argument&) {
        output_error_thrown = true;
    }
    passed &= expect(output_error_thrown,
                     "incorrect CUDA L1 output dimensions must be rejected");

    bool non_finite_query_thrown = false;
    try {
        const std::array<float, 2> query{
            std::numeric_limits<float>::quiet_NaN(), 0.0F};
        static_cast<void>(cuda_index.query(query, workspace));
    } catch (const std::invalid_argument&) {
        non_finite_query_thrown = true;
    }
    passed &= expect(non_finite_query_thrown,
                     "non-finite CUDA L1 queries must be rejected");

    const CudaExactL1Index other_index(representatives);
    bool ownership_error_thrown = false;
    try {
        static_cast<void>(other_index.query(
            std::span<const float>{query_values}.first(2), workspace));
    } catch (const std::invalid_argument&) {
        ownership_error_thrown = true;
    }
    passed &= expect(ownership_error_thrown,
                     "a workspace from another CUDA L1 index must be "
                     "rejected");

    bool invalid_weight_thrown = false;
    try {
        constexpr std::array<float, 1> invalid_weights{-1.0F};
        const CudaDenseL1ScanIndex invalid_index(representatives,
                                                 invalid_weights);
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        invalid_weight_thrown = true;
    }
    passed &= expect(invalid_weight_thrown,
                     "invalid CUDA L1 coordinate weights must be rejected");

    return passed ? 0 : 1;
}
