#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"
#include "ultrahigh_ann/exact/l2/cuda_exact_l2_index.hpp"
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

} // namespace

int main()
{
#if defined(ULTRAHIGH_ANN_TEST_REQUIRES_CUDA_DEVICE)
    if (!ultrahigh_ann::cuda_runtime_info().has_value()) {
        std::cout << "CUDA build has no usable device\n";
        return 77;
    }
#endif
    using ultrahigh_ann::CudaDenseL2QueryStrategy;
    using ultrahigh_ann::CudaDenseL2ScanIndex;
    using ultrahigh_ann::CudaExactL2Index;
    using ultrahigh_ann::CudaExactL2QueryStrategy;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::ExactL2Index;

    static_assert(!std::is_copy_constructible_v<CudaExactL2Index>);
    static_assert(std::is_move_constructible_v<CudaExactL2Index>);
    static_assert(!std::is_constructible_v<CudaExactL2Index, DenseMatrix&&>);
    static_assert(!std::is_copy_constructible_v<CudaDenseL2ScanIndex>);
    static_assert(std::is_move_constructible_v<CudaDenseL2ScanIndex>);
    static_assert(
        !std::is_constructible_v<CudaDenseL2ScanIndex, DenseMatrix&&>);
    static_assert(
        std::is_same_v<CudaExactL2QueryStrategy, CudaDenseL2QueryStrategy>);
    static_assert(std::is_same_v<CudaExactL2Index::QueryWorkspace,
                                 CudaDenseL2ScanIndex::QueryWorkspace>);

    std::vector<float> representative_values{
        0.0F, 0.0F, 5.0F, 0.0F, 2.0F, 4.0F,
    };
    const DenseMatrix representatives(std::move(representative_values), 3, 2);

    if (!ultrahigh_ann::cuda_exact_l2_available()) {
        bool exact_unavailable_threw = false;
        try {
            const CudaExactL2Index index(representatives);
            static_cast<void>(index);
        } catch (const std::runtime_error&) {
            exact_unavailable_threw = true;
        }
        bool dense_unavailable_threw = false;
        try {
            const CudaDenseL2ScanIndex scan(representatives);
            static_cast<void>(scan);
        } catch (const std::runtime_error&) {
            dense_unavailable_threw = true;
        }
        return expect(
                   exact_unavailable_threw && dense_unavailable_threw,
                   "unavailable CUDA facades and engines must fail explicitly")
                   ? 0
                   : 1;
    }

    const ExactL2Index cpu_index(representatives);
    const CudaExactL2Index cuda_index(representatives);
    const CudaDenseL2ScanIndex cuda_scan(representatives);
    auto workspace = cuda_index.make_query_workspace(4);
    auto scan_workspace = cuda_scan.make_query_workspace(4);

    bool passed = true;
    passed &= expect(cuda_index.device() == 0,
                     "the default CUDA device must be zero");
    passed &= expect(!cuda_index.device_name().empty(),
                     "the CUDA device name must be reported");
    passed &= expect(workspace.maximum_batch_size() == 4,
                     "the workspace must report its batch capacity");
    passed &=
        expect(cuda_index.space_usage().index_payload_bytes ==
                   6 * sizeof(float) + 3 * sizeof(float),
               "the CUDA index must retain representatives and squared norms");
    passed &= expect(cuda_scan.space_usage().index_payload_bytes ==
                         cuda_index.space_usage().index_payload_bytes,
                     "the exact facade must expose the dense scan payload");
    passed &=
        expect(workspace.payload_bytes() ==
                   4 * 2 * sizeof(float) + 4 * 3 * sizeof(float) +
                       4 * sizeof(float) + 4 * sizeof(std::size_t),
               "the CUDA workspace must report queries, distances, norms, "
               "and results");

    constexpr std::array<float, 8> query_values{
        1.0F, 0.0F, 3.0F, 3.0F, 2.5F, 0.0F, 5.0F, 0.0F,
    };
    std::array<std::size_t, 4> cuda_results{};
    cuda_index.query_batch(query_values, 4, cuda_results, workspace);
    for (std::size_t query_index = 0; query_index < cuda_results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 2, 2);
        passed &=
            expect(cuda_results[query_index] == cpu_index.query(query),
                   "batched CUDA predictions must match sequential exact L2");
    }
    passed &= expect(cuda_results[2] == 0,
                     "CUDA ties must resolve to the lowest row index");
    std::array<std::size_t, 4> dense_results{};
    cuda_scan.query_batch(query_values, 4, dense_results, scan_workspace,
                          CudaDenseL2QueryStrategy::direct);
    passed &= expect(
        dense_results == cuda_results,
        "the exact facade and dense CUDA engine must share direct results");

    std::array<std::size_t, 4> gemm_results{};
    cuda_index.query_batch(query_values, 4, gemm_results, workspace,
                           CudaExactL2QueryStrategy::gemm);
    for (std::size_t query_index = 0; query_index < gemm_results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{query_values}.subspan(query_index * 2, 2);
        passed &=
            expect(gemm_results[query_index] == cpu_index.query(query),
                   "GEMM CUDA predictions must match this well-conditioned "
                   "CPU fixture");
    }
    passed &= expect(gemm_results[2] == 0,
                     "GEMM CUDA ties must resolve to the lowest row index");
    cuda_scan.query_batch(query_values, 4, dense_results, scan_workspace,
                          CudaDenseL2QueryStrategy::gemm);
    passed &= expect(
        dense_results == gemm_results,
        "the exact facade and dense CUDA engine must share GEMM results");

    constexpr float cancellation_base = 1'048'576.0F;
    const DenseMatrix cancellation_representatives(
        std::vector<float>{
            cancellation_base + 0.25F,
            cancellation_base + 0.125F,
        },
        2, 1);
    const CudaDenseL2ScanIndex cancellation_scan(cancellation_representatives);
    auto cancellation_workspace = cancellation_scan.make_query_workspace(1);
    constexpr std::array<float, 1> cancellation_query{cancellation_base};
    std::array<std::size_t, 1> cancellation_direct_result{};
    std::array<std::size_t, 1> cancellation_gemm_result{};
    cancellation_scan.query_batch(
        cancellation_query, 1, cancellation_direct_result,
        cancellation_workspace, CudaDenseL2QueryStrategy::direct);
    cancellation_scan.query_batch(
        cancellation_query, 1, cancellation_gemm_result, cancellation_workspace,
        CudaDenseL2QueryStrategy::gemm);
    passed &=
        expect(cancellation_direct_result[0] == 1,
               "direct CUDA L2 must distinguish the cancellation fixture");
    passed &= expect(cancellation_gemm_result[0] == 0,
                     "GEMM CUDA L2 must select its unrefined binary32 argmin");

    passed &=
        expect(cuda_index.query(std::span<const float>{query_values}.first(2),
                                workspace) == 0,
               "the single-query CUDA API must return the nearest row");

    bool capacity_error_thrown = false;
    try {
        std::array<std::size_t, 5> output{};
        std::array<float, 10> queries{};
        cuda_index.query_batch(queries, 5, output, workspace);
    } catch (const std::invalid_argument&) {
        capacity_error_thrown = true;
    }
    passed &= expect(capacity_error_thrown,
                     "a batch exceeding workspace capacity must be rejected");

    bool dimension_error_thrown = false;
    try {
        std::array<std::size_t, 2> output{};
        const std::array<float, 3> queries{};
        cuda_index.query_batch(queries, 2, output, workspace);
    } catch (const std::invalid_argument&) {
        dimension_error_thrown = true;
    }
    passed &= expect(dimension_error_thrown,
                     "incorrect CUDA query dimensions must be rejected");

    bool output_error_thrown = false;
    try {
        std::array<std::size_t, 1> output{};
        cuda_index.query_batch(query_values, 4, output, workspace);
    } catch (const std::invalid_argument&) {
        output_error_thrown = true;
    }
    passed &= expect(output_error_thrown,
                     "incorrect CUDA output dimensions must be rejected");

    bool non_finite_query_thrown = false;
    try {
        const std::array<float, 2> query{
            std::numeric_limits<float>::quiet_NaN(), 0.0F};
        static_cast<void>(cuda_index.query(query, workspace));
    } catch (const std::invalid_argument&) {
        non_finite_query_thrown = true;
    }
    passed &= expect(non_finite_query_thrown,
                     "non-finite CUDA queries must be rejected");

    const CudaExactL2Index other_index(representatives);
    bool ownership_error_thrown = false;
    try {
        static_cast<void>(other_index.query(
            std::span<const float>{query_values}.first(2), workspace));
    } catch (const std::invalid_argument&) {
        ownership_error_thrown = true;
    }
    passed &= expect(ownership_error_thrown,
                     "a workspace from another CUDA index must be rejected");

    std::vector<float> invalid_values{std::numeric_limits<float>::infinity(),
                                      0.0F};
    const DenseMatrix invalid_representatives(std::move(invalid_values), 1, 2);
    bool non_finite_representative_thrown = false;
    try {
        const CudaExactL2Index invalid_index(invalid_representatives);
        static_cast<void>(invalid_index);
    } catch (const std::invalid_argument&) {
        non_finite_representative_thrown = true;
    }
    passed &= expect(non_finite_representative_thrown,
                     "non-finite CUDA representatives must be rejected");

    return passed ? 0 : 1;
}
