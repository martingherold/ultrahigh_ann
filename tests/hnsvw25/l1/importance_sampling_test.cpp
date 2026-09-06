#include "hnsvw25/l1/importance_sampling_detail.hpp"
#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"
#include "ultrahigh_ann/core/cuda_runtime_info.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
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

bool samples_equal(const ultrahigh_ann::CoordinateSample& left,
                   const ultrahigh_ann::CoordinateSample& right)
{
    if (left.columns.size() != right.columns.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.columns.size(); ++index) {
        const auto& left_column = left.columns[index];
        const auto& right_column = right.columns[index];
        if (left_column.source_column != right_column.source_column ||
            left_column.multiplicity != right_column.multiplicity ||
            left_column.inverse_probability !=
                right_column.inverse_probability) {
            return false;
        }
    }
    return true;
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
    using ultrahigh_ann::build_l1_importance_sample;
    using ultrahigh_ann::compute_l1_importance_probabilities;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::ExecutionPolicy;
    using ultrahigh_ann::detail::compute_l1_sampling_probabilities;
    using ultrahigh_ann::detail::compute_l1_sampling_probabilities_cpu_parallel;

    const DenseMatrix representatives(
        std::vector<float>{0.0F, 0.0F, 3.0F, 4.0F}, 2, 2);
    const auto probabilities =
        compute_l1_sampling_probabilities(representatives);
    const auto parallel_probabilities =
        compute_l1_sampling_probabilities_cpu_parallel(representatives);
    const auto public_compatibility_probabilities =
        compute_l1_importance_probabilities(representatives);
    const auto public_sequential_probabilities =
        compute_l1_importance_probabilities(representatives,
                                            ExecutionPolicy::sequential);
    const auto public_parallel_probabilities =
        compute_l1_importance_probabilities(representatives,
                                            ExecutionPolicy::cpu_parallel);

    bool passed = true;
    passed &= expect(probabilities.size() == 2,
                     "one probability must be produced per coordinate");
    passed &= expect(probabilities.size() >= 2 &&
                         std::abs(probabilities[0] - 3.0 / 7.0) < 1.0e-12,
                     "the first probability must be its L1 contribution");
    passed &= expect(probabilities.size() >= 2 &&
                         std::abs(probabilities[1] - 4.0 / 7.0) < 1.0e-12,
                     "the second probability must be its L1 contribution");
    passed &= expect(parallel_probabilities == probabilities,
                     "parallel probabilities must match sequential values");
    passed &= expect(public_compatibility_probabilities == probabilities,
                     "the original probability overload must be preserved");
    passed &= expect(public_sequential_probabilities == probabilities,
                     "the sequential policy must dispatch correctly");
    passed &= expect(public_parallel_probabilities == probabilities,
                     "the parallel policy must dispatch correctly");

    const DenseMatrix varied_representatives(
        std::vector<float>{
            0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F,  0.0F, 1.0F, -2.0F,
            5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, -4.0F, 1.0F, 2.0F, 7.0F,
        },
        5, 4);
    const auto varied_probabilities =
        compute_l1_sampling_probabilities(varied_representatives);
    const auto varied_parallel_probabilities =
        compute_l1_sampling_probabilities_cpu_parallel(varied_representatives);
    passed &= expect(varied_parallel_probabilities == varied_probabilities,
                     "parallel varied probabilities must match sequential");

    if (ultrahigh_ann::l1_gpu_backend_available()) {
        const auto gpu_result =
            ultrahigh_ann::compute_l1_importance_probabilities_gpu(
                varied_representatives, 0, 3);
        passed &= expect(gpu_result.distance_backend ==
                             ultrahigh_ann::L1GpuDistanceBackend::direct,
                         "the L1 GPU result must identify the direct backend");
        passed &= expect(gpu_result.pair_count == 10,
                         "GPU metadata must contain the pair count");
        passed &= expect(gpu_result.pair_chunks == 3,
                         "GPU metadata must contain the chunk count");
        passed &= expect(gpu_result.device_working_set_bytes > 0,
                         "GPU metadata must contain its working set");
        passed &= expect(gpu_result.probabilities.size() ==
                             varied_probabilities.size(),
                         "GPU computation must preserve coordinate count");
        if (gpu_result.probabilities.size() == varied_probabilities.size()) {
            for (std::size_t column = 0; column < varied_probabilities.size();
                 ++column) {
                passed &= expect(
                    std::abs(gpu_result.probabilities[column] -
                             varied_probabilities[column]) < 1.0e-6,
                    "FP32 GPU probabilities must approximate the reference");
            }
        }

        const auto public_gpu_probabilities =
            compute_l1_importance_probabilities(varied_representatives,
                                                ExecutionPolicy::gpu_fp32);
        passed &= expect(public_gpu_probabilities.size() ==
                             varied_probabilities.size(),
                         "the public GPU policy must preserve coordinates");
        if (public_gpu_probabilities.size() == varied_probabilities.size()) {
            for (std::size_t column = 0; column < varied_probabilities.size();
                 ++column) {
                passed &= expect(
                    std::abs(public_gpu_probabilities[column] -
                             varied_probabilities[column]) < 1.0e-6,
                    "the public GPU policy must approximate the reference");
            }
        }

        bool zero_chunks_threw = false;
        try {
            static_cast<void>(
                ultrahigh_ann::compute_l1_importance_probabilities_gpu(
                    varied_representatives, 0, 0));
        } catch (const std::invalid_argument&) {
            zero_chunks_threw = true;
        }
        passed &=
            expect(zero_chunks_threw, "zero GPU pair chunks must be rejected");
    } else {
        bool unavailable_gpu_threw = false;
        try {
            static_cast<void>(
                ultrahigh_ann::compute_l1_importance_probabilities_gpu(
                    varied_representatives));
        } catch (const std::runtime_error&) {
            unavailable_gpu_threw = true;
        }
        passed &= expect(unavailable_gpu_threw,
                         "an unavailable GPU backend must fail explicitly");

        bool unavailable_policy_threw = false;
        try {
            static_cast<void>(compute_l1_importance_probabilities(
                varied_representatives, ExecutionPolicy::gpu_fp32));
        } catch (const std::runtime_error&) {
            unavailable_policy_threw = true;
        }
        passed &= expect(unavailable_policy_threw,
                         "the GPU policy must fail without CUDA");
    }

    bool cublas_policy_threw = false;
    try {
        static_cast<void>(compute_l1_importance_probabilities(
            varied_representatives, ExecutionPolicy::gpu_cublas_fp32));
    } catch (const std::invalid_argument&) {
        cublas_policy_threw = true;
    }
    passed &= expect(cublas_policy_threw,
                     "L1 must explicitly reject the cuBLAS policy");

    const DenseMatrix duplicate_representatives(std::vector<float>(9, 1.0F), 3,
                                                3);
    passed &=
        expect(compute_l1_sampling_probabilities_cpu_parallel(
                   duplicate_representatives) == std::vector<double>(3, 0.0),
               "duplicate representatives must have zero probabilities");

    const DenseMatrix single_representative(
        std::vector<float>{1.0F, 2.0F, 3.0F}, 1, 3);
    passed &= expect(compute_l1_sampling_probabilities_cpu_parallel(
                         single_representative) == std::vector<double>(3, 0.0),
                     "one representative must have zero probabilities");

    const DenseMatrix zero_column_representatives(std::vector<float>{}, 4, 0);
    passed &=
        expect(compute_l1_sampling_probabilities_cpu_parallel(
                   zero_column_representatives)
                   .empty(),
               "zero-column representatives must produce no probabilities");

    const DenseMatrix non_finite_representatives(
        std::vector<float>{0.0F, std::numeric_limits<float>::quiet_NaN()}, 2,
        1);
    bool non_finite_threw = false;
    try {
        static_cast<void>(compute_l1_importance_probabilities(
            non_finite_representatives, ExecutionPolicy::cpu_parallel));
    } catch (const std::invalid_argument&) {
        non_finite_threw = true;
    }
    passed &= expect(non_finite_threw,
                     "parallel computation must reject non-finite values");

    constexpr std::size_t repetitions = 16;
    const std::vector<double> deterministic_probabilities{1.0, 0.0};
    std::mt19937_64 random_engine(42);
    const auto coordinate_sample = build_l1_importance_sample(
        deterministic_probabilities, repetitions, random_engine);
    passed &= expect(coordinate_sample.columns.size() == 1,
                     "one probability-one coordinate must be sampled");
    if (coordinate_sample.columns.size() == 1) {
        passed &= expect(coordinate_sample.columns[0].source_column == 0,
                         "the first coordinate must be sampled");
        passed &=
            expect(coordinate_sample.columns[0].multiplicity == repetitions,
                   "a probability-one coordinate must repeat T times");
        passed &= expect(
            coordinate_sample.columns[0].inverse_probability == 1.0,
            "a probability-one coordinate must have inverse probability one");
    }

    std::mt19937_64 sequential_engine(1234);
    std::mt19937_64 parallel_engine(1234);
    const auto sequential_sample = build_l1_importance_sample(
        varied_representatives, repetitions, sequential_engine,
        ExecutionPolicy::sequential);
    const auto parallel_sample = build_l1_importance_sample(
        varied_representatives, repetitions, parallel_engine,
        ExecutionPolicy::cpu_parallel);
    passed &= expect(samples_equal(sequential_sample, parallel_sample),
                     "parallel representative sampling must match sequential");

    std::mt19937_64 compatibility_representative_engine(1234);
    const auto compatibility_representative_sample =
        ultrahigh_ann::build_importance_sample(
            varied_representatives, repetitions,
            compatibility_representative_engine);
    passed &= expect(
        samples_equal(sequential_sample, compatibility_representative_sample),
        "the original representative sample overload must be preserved");

    std::mt19937_64 compatibility_engine(42);
    const auto compatibility_sample = ultrahigh_ann::build_importance_sample(
        deterministic_probabilities, repetitions, compatibility_engine);
    passed &= expect(samples_equal(coordinate_sample, compatibility_sample),
                     "the compatibility sample spelling must be preserved");

    return passed ? 0 : 1;
}
