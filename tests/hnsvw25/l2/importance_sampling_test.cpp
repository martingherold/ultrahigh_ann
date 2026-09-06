#include "hnsvw25/l2/importance_sampling_detail.hpp"
#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"
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
    using ultrahigh_ann::build_l2_importance_sample;
    using ultrahigh_ann::compute_l2_importance_probabilities;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::ExecutionPolicy;
    using ultrahigh_ann::detail::compute_l2_sampling_probabilities;
    using ultrahigh_ann::detail::compute_l2_sampling_probabilities_cpu_parallel;

    std::vector<float> values{0.0F, 0.0F, 3.0F, 4.0F};
    const DenseMatrix representatives(std::move(values), 2, 2);
    const auto probabilities =
        compute_l2_sampling_probabilities(representatives);
    const auto parallel_probabilities =
        compute_l2_sampling_probabilities_cpu_parallel(representatives);
    const auto public_sequential_probabilities =
        compute_l2_importance_probabilities(representatives,
                                            ExecutionPolicy::sequential);
    const auto public_parallel_probabilities =
        compute_l2_importance_probabilities(representatives,
                                            ExecutionPolicy::cpu_parallel);

    bool passed = true;
    passed &= expect(probabilities.size() == 2,
                     "one probability must be produced per coordinate");
    passed &=
        expect(probabilities.size() >= 2 &&
                   std::abs(probabilities[0] - 9.0 / 25.0) < 1.0e-12,
               "the first probability must be its squared L2 contribution");
    passed &=
        expect(probabilities.size() >= 2 &&
                   std::abs(probabilities[1] - 16.0 / 25.0) < 1.0e-12,
               "the second probability must be its squared L2 contribution");
    passed &=
        expect(parallel_probabilities == probabilities,
               "parallel probabilities must match sequential probabilities");
    passed &=
        expect(public_sequential_probabilities == probabilities,
               "the sequential policy must dispatch to the sequential result");
    passed &=
        expect(public_parallel_probabilities == probabilities,
               "the parallel policy must dispatch to the parallel result");

    std::vector<float> varied_values{0.0F, 0.0F, 0.0F,  0.0F, 3.0F, 4.0F, 0.0F,
                                     0.0F, 1.0F, -2.0F, 5.0F, 0.0F, 0.0F, 0.0F,
                                     0.0F, 0.0F, -4.0F, 1.0F, 2.0F, 7.0F};
    const DenseMatrix varied_representatives(std::move(varied_values), 5, 4);
    const auto varied_probabilities =
        compute_l2_sampling_probabilities(varied_representatives);
    const auto varied_parallel_probabilities =
        compute_l2_sampling_probabilities_cpu_parallel(varied_representatives);
    passed &=
        expect(varied_parallel_probabilities == varied_probabilities,
               "parallel probabilities must match for varied representatives");
    const auto varied_public_parallel_probabilities =
        compute_l2_importance_probabilities(varied_representatives,
                                            ExecutionPolicy::cpu_parallel);
    passed &=
        expect(varied_public_parallel_probabilities == varied_probabilities,
               "the public parallel policy must preserve varied probabilities");

    if (ultrahigh_ann::l2_gpu_backend_available()) {
        const auto gpu_result =
            ultrahigh_ann::compute_l2_importance_probabilities_gpu(
                varied_representatives, 0, 3);
        passed &=
            expect(gpu_result.pair_count == 10,
                   "GPU metadata must contain the representative pair count");
        passed &=
            expect(gpu_result.pair_chunks == 3,
                   "GPU metadata must contain the effective pair chunk count");
        passed &= expect(gpu_result.probabilities.size() ==
                             varied_probabilities.size(),
                         "GPU computation must preserve the coordinate count");
        if (gpu_result.probabilities.size() == varied_probabilities.size()) {
            for (std::size_t column = 0; column < varied_probabilities.size();
                 ++column) {
                passed &= expect(
                    std::abs(gpu_result.probabilities[column] -
                             varied_probabilities[column]) < 1.0e-6,
                    "FP32 GPU probabilities must approximate the reference");
            }
        }

        const auto public_gpu_fp32_probabilities =
            compute_l2_importance_probabilities(varied_representatives,
                                                ExecutionPolicy::gpu_fp32);
        passed &= expect(
            public_gpu_fp32_probabilities.size() == varied_probabilities.size(),
            "the public FP32 GPU policy must preserve the coordinate count");
        if (public_gpu_fp32_probabilities.size() ==
            varied_probabilities.size()) {
            for (std::size_t column = 0; column < varied_probabilities.size();
                 ++column) {
                passed &=
                    expect(std::abs(public_gpu_fp32_probabilities[column] -
                                    varied_probabilities[column]) < 1.0e-6,
                           "the public FP32 GPU policy must approximate the "
                           "reference");
            }
        }

        const auto gpu_cublas_fp32_result =
            ultrahigh_ann::compute_l2_importance_probabilities_gpu(
                varied_representatives, 0, 3,
                ultrahigh_ann::L2GpuDistanceBackend::cublas);
        passed &= expect(gpu_cublas_fp32_result.distance_backend ==
                             ultrahigh_ann::L2GpuDistanceBackend::cublas,
                         "the cuBLAS result must identify its backend");
        passed &=
            expect(gpu_cublas_fp32_result.device_working_set_bytes >
                       gpu_result.device_working_set_bytes,
                   "the cuBLAS computation must account for its Gram matrix");
        passed &= expect(
            gpu_cublas_fp32_result.probabilities.size() ==
                varied_probabilities.size(),
            "cuBLAS FP32 computation must preserve the coordinate count");
        if (gpu_cublas_fp32_result.probabilities.size() ==
            varied_probabilities.size()) {
            for (std::size_t column = 0; column < varied_probabilities.size();
                 ++column) {
                passed &= expect(
                    std::abs(gpu_cublas_fp32_result.probabilities[column] -
                             varied_probabilities[column]) < 1.0e-6,
                    "cuBLAS FP32 probabilities must approximate the reference");
            }
        }

        const auto public_gpu_cublas_fp32_probabilities =
            compute_l2_importance_probabilities(
                varied_representatives, ExecutionPolicy::gpu_cublas_fp32);
        passed &= expect(
            public_gpu_cublas_fp32_probabilities ==
                gpu_cublas_fp32_result.probabilities,
            "the public cuBLAS FP32 policy must dispatch to the cuBLAS path");

        std::vector<float> cancellation_values{
            10000.0F, 10000.0F, 10000.0F, 10000.0F, 10000.5F, 10000.0F,
            10000.0F, 10000.0F, 10000.0F, 9999.5F,  10000.0F, 10000.0F,
        };
        const DenseMatrix cancellation_representatives(
            std::move(cancellation_values), 3, 4);
        const auto cancellation_reference =
            compute_l2_sampling_probabilities(cancellation_representatives);
        const auto cancellation_cublas =
            ultrahigh_ann::compute_l2_importance_probabilities_gpu(
                cancellation_representatives, 0, 2,
                ultrahigh_ann::L2GpuDistanceBackend::cublas);
        passed &=
            expect(cancellation_cublas.probabilities.size() ==
                       cancellation_reference.size(),
                   "cuBLAS cancellation refinement must preserve dimensions");
        if (cancellation_cublas.probabilities.size() ==
            cancellation_reference.size()) {
            for (std::size_t column = 0; column < cancellation_reference.size();
                 ++column) {
                passed &= expect(
                    std::abs(cancellation_cublas.probabilities[column] -
                             cancellation_reference[column]) < 1.0e-6,
                    "cuBLAS cancellation refinement must match direct FP32");
            }
        }
    } else {
        bool unavailable_gpu_threw = false;
        try {
            static_cast<void>(
                ultrahigh_ann::compute_l2_importance_probabilities_gpu(
                    varied_representatives));
        } catch (const std::runtime_error&) {
            unavailable_gpu_threw = true;
        }
        passed &= expect(unavailable_gpu_threw,
                         "an unavailable GPU backend must fail explicitly");

        bool unavailable_gpu_fp32_policy_threw = false;
        try {
            static_cast<void>(compute_l2_importance_probabilities(
                varied_representatives, ExecutionPolicy::gpu_fp32));
        } catch (const std::runtime_error&) {
            unavailable_gpu_fp32_policy_threw = true;
        }
        passed &= expect(
            unavailable_gpu_fp32_policy_threw,
            "the public FP32 GPU policy must fail when CUDA is unavailable");

        bool unavailable_gpu_cublas_fp32_policy_threw = false;
        try {
            static_cast<void>(compute_l2_importance_probabilities(
                varied_representatives, ExecutionPolicy::gpu_cublas_fp32));
        } catch (const std::runtime_error&) {
            unavailable_gpu_cublas_fp32_policy_threw = true;
        }
        passed &= expect(
            unavailable_gpu_cublas_fp32_policy_threw,
            "the public cuBLAS FP32 policy must fail when CUDA is unavailable");
    }

    std::vector<float> duplicate_values(9, 1.0F);
    const DenseMatrix duplicate_representatives(std::move(duplicate_values), 3,
                                                3);
    const auto duplicate_probabilities =
        compute_l2_sampling_probabilities_cpu_parallel(
            duplicate_representatives);
    passed &= expect(duplicate_probabilities == std::vector<double>(3, 0.0),
                     "duplicate representatives must have zero probabilities");

    std::vector<float> single_values{1.0F, 2.0F, 3.0F};
    const DenseMatrix single_representative(std::move(single_values), 1, 3);
    const auto single_probabilities =
        compute_l2_sampling_probabilities_cpu_parallel(single_representative);
    passed &= expect(single_probabilities == std::vector<double>(3, 0.0),
                     "one representative must have zero probabilities");

    const DenseMatrix zero_column_representatives(std::vector<float>{}, 4, 0);
    const auto zero_column_probabilities =
        compute_l2_sampling_probabilities_cpu_parallel(
            zero_column_representatives);
    passed &=
        expect(zero_column_probabilities.empty(),
               "zero-column representatives must produce no probabilities");

    std::vector<float> non_finite_values{
        0.0F, std::numeric_limits<float>::quiet_NaN()};
    const DenseMatrix non_finite_representatives(std::move(non_finite_values),
                                                 2, 1);
    bool non_finite_threw = false;
    try {
        static_cast<void>(compute_l2_importance_probabilities(
            non_finite_representatives, ExecutionPolicy::cpu_parallel));
    } catch (const std::invalid_argument&) {
        non_finite_threw = true;
    }
    passed &= expect(
        non_finite_threw,
        "parallel probability computation must reject non-finite values");

    constexpr std::size_t repetitions = 16;
    const std::vector<double> deterministic_probabilities{1.0, 0.0};
    std::mt19937_64 random_engine(42);
    const auto coordinate_sample = build_l2_importance_sample(
        deterministic_probabilities, repetitions, random_engine);
    const auto& columns = coordinate_sample.columns;
    passed &= expect(columns.size() == 1,
                     "exactly one probability-one coordinate must be sampled");
    if (columns.size() == 1) {
        passed &= expect(columns[0].source_column == 0,
                         "the first coordinate must be sampled");
        passed &=
            expect(columns[0].multiplicity == repetitions,
                   "a probability-one coordinate must occur every repetition");
        passed &= expect(
            std::abs(columns[0].inverse_probability - 1.0) < 1.0e-12,
            "a probability-one coordinate must have inverse probability one");
    }

    std::mt19937_64 sequential_engine(1234);
    std::mt19937_64 parallel_engine(1234);
    const auto sequential_sample = build_l2_importance_sample(
        varied_representatives, repetitions, sequential_engine,
        ExecutionPolicy::sequential);
    const auto parallel_sample = build_l2_importance_sample(
        varied_representatives, repetitions, parallel_engine,
        ExecutionPolicy::cpu_parallel);
    passed &= expect(
        samples_equal(sequential_sample, parallel_sample),
        "parallel representative sampling must match sequential sampling");

    return passed ? 0 : 1;
}
