#include "ultrahigh_ann/core/dense_l2_scan_index.hpp"

#include "core/finite_values.hpp"
#include "core/l2_distance.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <omp.h>
#include <span>
#include <stdexcept>

namespace ultrahigh_ann {
namespace {

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error("dense L2 query batch size overflows");
    }
    return left * right;
}

[[nodiscard]] std::size_t nearest_representative(
    const DenseMatrix& representatives,
    std::span<const float> query)
{
    std::size_t nearest_index = 0;
    double nearest_distance =
        detail::squared_l2_dist(query, representatives.row(0));
    for (std::size_t row = 1; row < representatives.rows(); ++row) {
        const double distance =
            detail::squared_l2_dist(query, representatives.row(row));
        if (distance < nearest_distance) {
            nearest_distance = distance;
            nearest_index = row;
        }
    }
    return nearest_index;
}

struct L2Candidate {
    double distance{std::numeric_limits<double>::infinity()};
    std::size_t index{std::numeric_limits<std::size_t>::max()};
};

[[nodiscard]] L2Candidate better_candidate(const L2Candidate& candidate,
                                           const L2Candidate& current) noexcept
{
    if (candidate.distance < current.distance ||
        (candidate.distance == current.distance &&
         candidate.index < current.index)) {
        return candidate;
    }
    return current;
}

#pragma omp declare reduction(ultrahigh_ann_l2_candidate_min:L2Candidate       \
                              : omp_out = better_candidate(omp_in, omp_out))   \
    initializer(omp_priv = L2Candidate{})

[[nodiscard]] std::size_t nearest_representative_parallel(
    const DenseMatrix& representatives,
    std::span<const float> query)
{
    L2Candidate nearest;
#pragma omp parallel for schedule(static)                                      \
    reduction(ultrahigh_ann_l2_candidate_min                                   \
              : nearest)
    for (std::size_t row = 0; row < representatives.rows(); ++row) {
        const L2Candidate candidate{
            .distance =
                detail::squared_l2_dist(query, representatives.row(row)),
            .index = row,
        };
        nearest = better_candidate(candidate, nearest);
    }
    return nearest.index;
}

[[nodiscard]] CpuDenseL2QueryStrategy automatic_strategy(
    std::size_t query_count,
    std::size_t representative_count) noexcept
{
    if (query_count == 0 || omp_in_parallel() != 0) {
        return CpuDenseL2QueryStrategy::sequential;
    }
    const std::size_t maximum_threads =
        static_cast<std::size_t>(omp_get_max_threads());
    if (maximum_threads <= 1) {
        return CpuDenseL2QueryStrategy::sequential;
    }
    const std::size_t useful_threads =
        std::min(maximum_threads, representative_count);
    const std::size_t query_parallel_threshold =
        std::max(std::size_t{1}, (useful_threads + 1) / 2);
    return query_count >= query_parallel_threshold
               ? CpuDenseL2QueryStrategy::parallel_queries
               : CpuDenseL2QueryStrategy::parallel_representatives;
}

}  // namespace

DenseL2ScanIndex::DenseL2ScanIndex(const DenseMatrix& representatives)
    : representatives_(representatives)
{
    if (representatives_.rows() == 0) {
        throw std::invalid_argument(
            "DenseL2ScanIndex expects at least one representative");
    }
    detail::validate_finite_values(representatives_.values(),
                                   "representatives");
}

std::size_t DenseL2ScanIndex::query(std::span<const float> query,
                                    CpuDenseL2QueryStrategy strategy) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, strategy);
    return output.front();
}

void DenseL2ScanIndex::query_batch(std::span<const float> queries,
                                   std::size_t query_count,
                                   std::span<std::size_t> output,
                                   CpuDenseL2QueryStrategy strategy) const
{
    const std::size_t expected_values =
        checked_product(query_count, representatives_.cols());
    if (queries.size() != expected_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match dense L2 representation");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "dense L2 output size does not match query count");
    }
    detail::validate_finite_values(queries, "queries");

    if (strategy == CpuDenseL2QueryStrategy::automatic) {
        strategy = automatic_strategy(query_count, representatives_.rows());
    }
    if (strategy == CpuDenseL2QueryStrategy::sequential) {
        for (std::size_t query_index = 0; query_index < query_count;
             ++query_index) {
            output[query_index] = nearest_representative(
                representatives_,
                queries.subspan(query_index * representatives_.cols(),
                                representatives_.cols()));
        }
    } else if (strategy == CpuDenseL2QueryStrategy::parallel_queries) {
#pragma omp parallel for schedule(static)
        for (std::size_t query_index = 0; query_index < query_count;
             ++query_index) {
            output[query_index] = nearest_representative(
                representatives_,
                queries.subspan(query_index * representatives_.cols(),
                                representatives_.cols()));
        }
    } else if (strategy == CpuDenseL2QueryStrategy::parallel_representatives) {
        for (std::size_t query_index = 0; query_index < query_count;
             ++query_index) {
            output[query_index] = nearest_representative_parallel(
                representatives_,
                queries.subspan(query_index * representatives_.cols(),
                                representatives_.cols()));
        }
    } else {
        throw std::invalid_argument("unknown CPU dense L2 query strategy");
    }
}

IndexSpaceUsage DenseL2ScanIndex::space_usage() const noexcept
{
    return IndexSpaceUsage{
        .index_payload_bytes = representatives_.values().size_bytes(),
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates = representatives_.cols(),
        .sampled_multiplicity = representatives_.cols(),
    };
}

}  // namespace ultrahigh_ann
