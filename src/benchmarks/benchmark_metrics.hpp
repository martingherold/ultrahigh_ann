#pragma once

#include "benchmark_setup.hpp"
#include "core/dense_matrix.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ultrahigh_ann::benchmark {

inline constexpr std::array<double, 4> approximation_epsilons{
    0.01,
    0.05,
    0.10,
    0.20,
};

struct DistributionSummary {
    std::size_t count{};
    std::optional<double> mean{};
    std::optional<double> median{};
    std::optional<double> percentile_95{};
    std::optional<double> percentile_99{};
    std::optional<double> maximum{};
};

struct MarginBucketMetrics {
    double lower_inclusive{};
    std::optional<double> upper_exclusive{};
    std::size_t query_count{};
    std::size_t non_optimal_count{};
    std::size_t ratio_eligible_count{};
    std::optional<double> mean_distance_ratio{};
};

struct ApproximationFailureMetrics {
    double epsilon{};
    std::size_t violation_count{};
};

struct ApproximationMetrics {
    std::size_t query_count{};
    std::size_t optimal_representative_count{};
    std::size_t non_optimal_count{};
    DistributionSummary distance_ratio;
    std::size_t zero_optimum_query_count{};
    std::size_t zero_optimum_non_optimal_count{};
    std::size_t conditional_non_optimal_ratio_count{};
    std::optional<double> conditional_mean_distance_ratio{};
    std::optional<double> conditional_mean_relative_excess{};
    std::array<ApproximationFailureMetrics, approximation_epsilons.size()>
        approximation_failures;
    DistributionSummary returned_representative_rank;
    std::vector<MarginBucketMetrics> margin_buckets;
};

struct QueryGeometryMetrics {
    std::size_t query_count{};
    std::size_t zero_optimum_query_count{};
    std::size_t non_unique_optimum_query_count{};
    DistributionSummary finite_multiplicative_margin;
    std::size_t infinite_multiplicative_margin_count{};
    std::vector<MarginBucketMetrics> margin_buckets;
};

// Stores one exact distance for every query/representative pair. Building the
// table is diagnostic work and must remain outside measured index query time.
// It lets every approximate run in a batch reuse the same exact geometry.
class ExactDistanceTable {
public:
    ExactDistanceTable(
        DistanceMetric metric,
        const DenseMatrix& representatives,
        const DenseMatrix& queries,
        std::size_t query_count);

    [[nodiscard]] const QueryGeometryMetrics& query_geometry() const noexcept;

    [[nodiscard]] ApproximationMetrics evaluate(
        std::span<const std::size_t> predictions) const;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;

private:
    std::size_t representative_count_{};
    std::vector<double> distances_;
    std::vector<double> optimum_distances_;
    std::vector<double> margins_;
    QueryGeometryMetrics query_geometry_;
};

}  // namespace ultrahigh_ann::benchmark
