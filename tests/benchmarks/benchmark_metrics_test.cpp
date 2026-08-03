#include "benchmark_metrics.hpp"

#include "core/dense_matrix.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace {

void expect(bool condition, std::string_view message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

void expect_near(
    std::optional<double> actual,
    double expected,
    std::string_view message)
{
    expect(actual.has_value(), message);
    expect(std::abs(*actual - expected) < 1e-12, message);
}

}  // namespace

int main()
{
    using ultrahigh_ann::DenseMatrix;
    using namespace ultrahigh_ann::benchmark;

    const DenseMatrix representatives(
        std::vector<float>{0.0F, 2.0F, 5.0F},
        3,
        1);
    const DenseMatrix queries(
        std::vector<float>{0.0F, 1.0F, 3.0F, 10.0F},
        4,
        1);
    const ExactDistanceTable distances(
        DistanceMetric::l1,
        representatives,
        queries,
        queries.rows());

    const QueryGeometryMetrics& geometry = distances.query_geometry();
    expect(geometry.query_count == 4, "geometry query count");
    expect(geometry.zero_optimum_query_count == 1, "zero optimum count");
    expect(
        geometry.non_unique_optimum_query_count == 1,
        "tied optimum count");
    expect(
        geometry.infinite_multiplicative_margin_count == 1,
        "infinite margin count");
    expect_near(
        geometry.finite_multiplicative_margin.mean,
        (1.0 + 2.0 + 1.6) / 3.0,
        "finite mean margin");
    expect_near(
        geometry.finite_multiplicative_margin.median,
        1.6,
        "finite median margin");
    expect_near(
        geometry.finite_multiplicative_margin.percentile_95,
        2.0,
        "finite p95 margin");
    expect(geometry.margin_buckets[0].query_count == 1, "unit margin bucket");
    expect(geometry.margin_buckets[4].query_count == 3, "large margin bucket");

    // Query 1 returns the other exact-distance tie and must count as optimal.
    const ApproximationMetrics metrics = distances.evaluate(
        std::vector<std::size_t>{0, 1, 0, 1});
    expect(metrics.query_count == 4, "approximation query count");
    expect(
        metrics.optimal_representative_count == 2,
        "distance-based agreement accepts ties");
    expect(metrics.non_optimal_count == 2, "non-optimal count");
    expect(metrics.distance_ratio.count == 3, "positive optimum ratio count");
    expect_near(
        metrics.distance_ratio.mean,
        (1.0 + 3.0 + 1.6) / 3.0,
        "mean approximation ratio");
    expect_near(metrics.distance_ratio.median, 1.6, "median ratio");
    expect_near(metrics.distance_ratio.percentile_95, 3.0, "p95 ratio");
    expect_near(metrics.distance_ratio.percentile_99, 3.0, "p99 ratio");
    expect_near(metrics.distance_ratio.maximum, 3.0, "maximum ratio");
    expect(
        metrics.conditional_non_optimal_ratio_count == 2,
        "conditional ratio count");
    expect_near(
        metrics.conditional_mean_distance_ratio,
        2.3,
        "conditional mean ratio");
    expect_near(
        metrics.conditional_mean_relative_excess,
        1.3,
        "conditional relative excess");
    expect(
        metrics.zero_optimum_query_count == 1 &&
            metrics.zero_optimum_non_optimal_count == 0,
        "zero optimum queries are reported separately");
    for (const auto& failure : metrics.approximation_failures) {
        expect(failure.violation_count == 2, "epsilon violation count");
    }
    expect_near(
        metrics.returned_representative_rank.mean,
        1.75,
        "mean returned rank");
    expect_near(
        metrics.returned_representative_rank.percentile_95,
        3.0,
        "p95 returned rank");
    expect(metrics.margin_buckets[0].non_optimal_count == 0, "tie bucket errors");
    expect(metrics.margin_buckets[4].non_optimal_count == 2, "large margin errors");

    const ApproximationMetrics zero_miss = distances.evaluate(
        std::vector<std::size_t>{1, 0, 1, 2});
    expect(
        zero_miss.zero_optimum_non_optimal_count == 1,
        "zero-distance miss count");
    for (const auto& failure : zero_miss.approximation_failures) {
        expect(
            failure.violation_count >= 1,
            "zero-distance miss violates every finite approximation factor");
    }

    const ExactDistanceTable l2_distances(
        DistanceMetric::l2,
        representatives,
        queries,
        queries.rows());
    const ApproximationMetrics l2_metrics = l2_distances.evaluate(
        std::vector<std::size_t>{0, 1, 0, 1});
    expect_near(
        l2_metrics.distance_ratio.mean,
        *metrics.distance_ratio.mean,
        "one-dimensional L1 and L2 ratios agree");

    return 0;
}
