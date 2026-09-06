#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

#include "core/finite_values.hpp"
#include "core/l1_distance.hpp"
#include "hnsvw25/l1/importance_sampling_detail.hpp"
#include "ultrahigh_ann/coordinate_sampling/coordinate_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace ultrahigh_ann {
namespace detail {

std::vector<double>
compute_l1_sampling_probabilities(const DenseMatrix& representatives)
{
    validate_finite_values(representatives.values(), "representatives");

    const std::size_t row_count = representatives.rows();
    const std::size_t column_count = representatives.cols();
    std::vector<double> probabilities(column_count, 0.0);
    for (std::size_t first = 0; first < row_count; ++first) {
        const auto first_row = representatives.row(first);
        for (std::size_t second = first + 1; second < row_count; ++second) {
            const auto second_row = representatives.row(second);
            const double distance = detail::l1_dist(first_row, second_row);
            if (distance == 0.0) {
                continue;
            }
            const double inverse_distance = 1.0 / distance;
            for (std::size_t column = 0; column < column_count; ++column) {
                const double probability =
                    std::abs(static_cast<double>(first_row[column]) -
                             static_cast<double>(second_row[column])) *
                    inverse_distance;
                probabilities[column] =
                    std::max(probabilities[column], probability);
            }
        }
    }
    return probabilities;
}

std::vector<double> compute_l1_sampling_probabilities_cpu_parallel(
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
                const double distance = detail::l1_dist(first_row, second_row);
                if (distance == 0.0) {
                    continue;
                }
                const double inverse_distance = 1.0 / distance;
                for (std::size_t column = 0; column < column_count; ++column) {
                    const double probability =
                        std::abs(static_cast<double>(first_row[column]) -
                                 static_cast<double>(second_row[column])) *
                        inverse_distance;
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
compute_l1_importance_probabilities(const DenseMatrix& representatives)
{
    return compute_l1_importance_probabilities(representatives,
                                               ExecutionPolicy::sequential);
}

std::vector<double>
compute_l1_importance_probabilities(const DenseMatrix& representatives,
                                    ExecutionPolicy execution_policy)
{
    switch (execution_policy) {
    case ExecutionPolicy::sequential:
        return detail::compute_l1_sampling_probabilities(representatives);
    case ExecutionPolicy::cpu_parallel:
        return detail::compute_l1_sampling_probabilities_cpu_parallel(
            representatives);
    case ExecutionPolicy::gpu_fp32:
        return compute_l1_importance_probabilities_gpu(representatives)
            .probabilities;
    case ExecutionPolicy::gpu_cublas_fp32:
        throw std::invalid_argument(
            "gpu_cublas_fp32 is unavailable for L1 probability computation; "
            "L1 distance has no matrix-multiplication identity");
    }
    throw std::invalid_argument("unknown L1 probability execution policy");
}

CoordinateSample
build_l1_importance_sample(std::span<const double> probabilities,
                           std::size_t repetitions,
                           std::mt19937_64& random_engine)
{
    return build_coordinate_sample(probabilities, repetitions, random_engine);
}

CoordinateSample build_l1_importance_sample(const DenseMatrix& representatives,
                                            std::size_t repetitions,
                                            std::mt19937_64& random_engine,
                                            ExecutionPolicy execution_policy)
{
    const auto probabilities =
        compute_l1_importance_probabilities(representatives, execution_policy);
    return build_l1_importance_sample(probabilities, repetitions,
                                      random_engine);
}

CoordinateSample build_importance_sample(std::span<const double> probabilities,
                                         std::size_t repetitions,
                                         std::mt19937_64& random_engine)
{
    return build_l1_importance_sample(probabilities, repetitions,
                                      random_engine);
}

CoordinateSample build_importance_sample(const DenseMatrix& representatives,
                                         std::size_t repetitions,
                                         std::mt19937_64& random_engine)
{
    return build_l1_importance_sample(representatives, repetitions,
                                      random_engine,
                                      ExecutionPolicy::sequential);
}

CoordinateSample build_importance_sample(const DenseMatrix& representatives,
                                         std::size_t repetitions,
                                         std::mt19937_64& random_engine,
                                         ExecutionPolicy execution_policy)
{
    return build_l1_importance_sample(representatives, repetitions,
                                      random_engine, execution_policy);
}

}  // namespace ultrahigh_ann
