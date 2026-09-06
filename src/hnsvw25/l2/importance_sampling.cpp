#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include "core/finite_values.hpp"
#include "core/l2_distance.hpp"
#include "hnsvw25/l2/importance_sampling_detail.hpp"
#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <span>
#include <vector>

namespace ultrahigh_ann {
namespace detail {

std::vector<double>
compute_l2_sampling_probabilities(const DenseMatrix& representatives)
{
    validate_finite_values(representatives.values(), "representatives");
    std::vector<double> probabilities(representatives.cols(), 0.0);

    for (std::size_t first = 0; first < representatives.rows(); ++first) {
        const auto first_row = representatives.row(first);
        for (std::size_t second = first + 1; second < representatives.rows();
             ++second) {
            const auto second_row = representatives.row(second);
            const double squared_distance =
                squared_l2_dist(first_row, second_row);
            if (squared_distance == 0.0) {
                continue;
            }
            const double inverse_distance = 1.0 / squared_distance;
            for (std::size_t column = 0; column < representatives.cols();
                 ++column) {
                const double difference =
                    static_cast<double>(first_row[column]) -
                    static_cast<double>(second_row[column]);
                const double probability =
                    std::min(1.0, difference * difference * inverse_distance);
                probabilities[column] =
                    std::max(probabilities[column], probability);
            }
        }
    }
    return probabilities;
}

std::vector<double> compute_l2_sampling_probabilities_cpu_parallel(
    const DenseMatrix& representatives)
{
    validate_finite_values(representatives.values(), "representatives");

    const std::size_t row_count = representatives.rows();
    const std::size_t column_count = representatives.cols();
    std::vector<double> probabilities(column_count, 0.0);
    if (row_count < 2 || column_count == 0) {
        return probabilities;
    }

    const int worker_count = static_cast<int>(std::min<std::size_t>(
        row_count - 1, static_cast<std::size_t>(omp_get_max_threads())));
    std::vector<std::vector<double>> worker_probabilities(
        static_cast<std::size_t>(worker_count),
        std::vector<double>(column_count, 0.0));

#pragma omp parallel num_threads(worker_count)
    {
        auto& local_probabilities =
            worker_probabilities[static_cast<std::size_t>(
                omp_get_thread_num())];

#pragma omp for schedule(guided)
        for (std::size_t first = 0; first < row_count - 1; ++first) {
            const auto first_row = representatives.row(first);
            for (std::size_t second = first + 1; second < row_count; ++second) {
                const auto second_row = representatives.row(second);
                const double squared_distance =
                    squared_l2_dist(first_row, second_row);
                if (squared_distance == 0.0) {
                    continue;
                }
                const double inverse_distance = 1.0 / squared_distance;
                for (std::size_t column = 0; column < column_count; ++column) {
                    const double difference =
                        static_cast<double>(first_row[column]) -
                        static_cast<double>(second_row[column]);
                    const double probability = std::min(
                        1.0, difference * difference * inverse_distance);
                    local_probabilities[column] =
                        std::max(local_probabilities[column], probability);
                }
            }
        }
    }

#pragma omp parallel for schedule(static) num_threads(worker_count)
    for (std::size_t column = 0; column < column_count; ++column) {
        for (const auto& local_probabilities : worker_probabilities) {
            probabilities[column] =
                std::max(probabilities[column], local_probabilities[column]);
        }
    }

    return probabilities;
}

}  // namespace detail

std::vector<double>
compute_l2_importance_probabilities(const DenseMatrix& representatives,
                                    ExecutionPolicy execution_policy)
{
    if (execution_policy == ExecutionPolicy::gpu_fp32) {
        return compute_l2_importance_probabilities_gpu(representatives)
            .probabilities;
    }
    if (execution_policy == ExecutionPolicy::gpu_cublas_fp32) {
        return compute_l2_importance_probabilities_gpu(
                   representatives, 0, 128, L2GpuDistanceBackend::cublas)
            .probabilities;
    }
    if (execution_policy == ExecutionPolicy::cpu_parallel) {
        return detail::compute_l2_sampling_probabilities_cpu_parallel(
            representatives);
    }
    return detail::compute_l2_sampling_probabilities(representatives);
}

CoordinateSample
build_l2_importance_sample(std::span<const double> probabilities,
                           std::size_t repetitions,
                           std::mt19937_64& random_engine)
{
    return build_coordinate_sample(probabilities, repetitions, random_engine);
}

CoordinateSample build_l2_importance_sample(const DenseMatrix& representatives,
                                            std::size_t repetitions,
                                            std::mt19937_64& random_engine,
                                            ExecutionPolicy execution_policy)
{
    const auto probabilities =
        compute_l2_importance_probabilities(representatives, execution_policy);
    return build_l2_importance_sample(probabilities, repetitions,
                                      random_engine);
}

}  // namespace ultrahigh_ann
