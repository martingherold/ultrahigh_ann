#include "benchmark_metrics.hpp"

#include "../core/l1_distance.hpp"
#include "../core/l2_distance.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ultrahigh_ann::benchmark {
namespace {

struct MarginBucketDefinition {
    double lower_inclusive;
    std::optional<double> upper_exclusive;
};

constexpr std::array<MarginBucketDefinition, 5> margin_bucket_definitions{{
    {1.00, 1.01},
    {1.01, 1.05},
    {1.05, 1.10},
    {1.10, 1.25},
    {1.25, std::nullopt},
}};

[[nodiscard]] double exact_distance(
    DistanceMetric metric,
    std::span<const float> query,
    std::span<const float> representative)
{
    if (metric == DistanceMetric::l1) {
        return detail::l1_dist(query, representative);
    }
    return std::sqrt(detail::squared_l2_dist(query, representative));
}

[[nodiscard]] double nearest_rank_quantile(
    const std::vector<double>& sorted_values,
    double probability)
{
    if (sorted_values.empty()) {
        throw std::logic_error("cannot take a quantile of an empty sample");
    }
    const std::size_t one_based_rank = static_cast<std::size_t>(
        std::ceil(probability * static_cast<double>(sorted_values.size())));
    return sorted_values[std::max<std::size_t>(one_based_rank, 1) - 1];
}

[[nodiscard]] DistributionSummary summarize(std::vector<double> values)
{
    DistributionSummary result{.count = values.size()};
    if (values.empty()) {
        return result;
    }
    std::ranges::sort(values);
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                  static_cast<double>(values.size());
    result.median = nearest_rank_quantile(values, 0.50);
    result.percentile_95 = nearest_rank_quantile(values, 0.95);
    result.percentile_99 = nearest_rank_quantile(values, 0.99);
    result.maximum = values.back();
    return result;
}

[[nodiscard]] std::size_t margin_bucket_index(double margin)
{
    for (std::size_t index = 0;
         index < margin_bucket_definitions.size();
         ++index) {
        const auto& definition = margin_bucket_definitions[index];
        if (margin >= definition.lower_inclusive &&
            (!definition.upper_exclusive.has_value() ||
             margin < *definition.upper_exclusive)) {
            return index;
        }
    }
    throw std::logic_error("multiplicative margin is smaller than one");
}

[[nodiscard]] std::vector<MarginBucketMetrics> make_margin_buckets()
{
    std::vector<MarginBucketMetrics> result;
    result.reserve(margin_bucket_definitions.size());
    for (const auto& definition : margin_bucket_definitions) {
        result.push_back(MarginBucketMetrics{
            .lower_inclusive = definition.lower_inclusive,
            .upper_exclusive = definition.upper_exclusive,
        });
    }
    return result;
}

[[nodiscard]] double multiplicative_margin(
    double optimum,
    double second)
{
    if (optimum > 0.0) {
        return second / optimum;
    }
    if (second == 0.0) {
        return 1.0;
    }
    return std::numeric_limits<double>::infinity();
}

}  // namespace

ExactDistanceTable::ExactDistanceTable(
    DistanceMetric metric,
    const DenseMatrix& representatives,
    const DenseMatrix& queries,
    std::size_t query_count)
    : representative_count_(representatives.rows())
{
    if (representative_count_ == 0) {
        throw std::invalid_argument(
            "distance diagnostics expect at least one representative");
    }
    if (query_count == 0 || query_count > queries.rows()) {
        throw std::invalid_argument(
            "distance diagnostics expect a valid positive query count");
    }
    if (representatives.cols() != queries.cols()) {
        throw std::invalid_argument(
            "representative and query dimensions differ");
    }
    if (query_count > distances_.max_size() / representative_count_) {
        throw std::length_error("exact distance table dimensions overflow");
    }

    distances_.resize(query_count * representative_count_);
    optimum_distances_.reserve(query_count);
    margins_.reserve(query_count);
    query_geometry_.query_count = query_count;
    query_geometry_.margin_buckets = make_margin_buckets();
    std::vector<double> finite_margins;
    finite_margins.reserve(query_count);

    for (std::size_t query_index = 0;
         query_index < query_count;
         ++query_index) {
        const auto query = queries.row(query_index);
        auto row = std::span<double>{distances_}.subspan(
            query_index * representative_count_,
            representative_count_);
        for (std::size_t representative_index = 0;
             representative_index < representative_count_;
             ++representative_index) {
            row[representative_index] = exact_distance(
                metric,
                query,
                representatives.row(representative_index));
        }

        const auto minimum = std::ranges::min_element(row);
        const double optimum = *minimum;
        const std::size_t optimum_count = static_cast<std::size_t>(
            std::ranges::count(row, optimum));
        double second = std::numeric_limits<double>::infinity();
        if (representative_count_ > 1) {
            const std::size_t minimum_index = static_cast<std::size_t>(
                std::distance(row.begin(), minimum));
            for (std::size_t index = 0; index < row.size(); ++index) {
                if (index != minimum_index) {
                    second = std::min(second, row[index]);
                }
            }
        }

        const double margin = multiplicative_margin(optimum, second);
        optimum_distances_.push_back(optimum);
        margins_.push_back(margin);
        query_geometry_.zero_optimum_query_count +=
            static_cast<std::size_t>(optimum == 0.0);
        query_geometry_.non_unique_optimum_query_count +=
            static_cast<std::size_t>(optimum_count > 1);
        ++query_geometry_.margin_buckets[margin_bucket_index(margin)]
              .query_count;
        if (std::isinf(margin)) {
            ++query_geometry_.infinite_multiplicative_margin_count;
        } else {
            finite_margins.push_back(margin);
        }
    }

    query_geometry_.finite_multiplicative_margin =
        summarize(std::move(finite_margins));
}

const QueryGeometryMetrics& ExactDistanceTable::query_geometry() const noexcept
{
    return query_geometry_;
}

ApproximationMetrics ExactDistanceTable::evaluate(
    std::span<const std::size_t> predictions) const
{
    if (predictions.size() != optimum_distances_.size()) {
        throw std::invalid_argument(
            "prediction count does not match exact distance table");
    }

    ApproximationMetrics result;
    result.query_count = predictions.size();
    result.margin_buckets = make_margin_buckets();
    for (std::size_t index = 0;
         index < approximation_epsilons.size();
         ++index) {
        result.approximation_failures[index].epsilon =
            approximation_epsilons[index];
    }

    std::vector<double> ratios;
    ratios.reserve(predictions.size());
    std::vector<double> ranks;
    ranks.reserve(predictions.size());
    double conditional_ratio_sum = 0.0;
    std::vector<double> bucket_ratio_sums(result.margin_buckets.size(), 0.0);

    for (std::size_t query_index = 0;
         query_index < predictions.size();
         ++query_index) {
        const std::size_t prediction = predictions[query_index];
        if (prediction >= representative_count_) {
            throw std::invalid_argument(
                "prediction contains an out-of-range representative row");
        }
        const auto row = std::span<const double>{distances_}.subspan(
            query_index * representative_count_,
            representative_count_);
        const double optimum = optimum_distances_[query_index];
        const double returned = row[prediction];
        const bool optimal = returned == optimum;
        const std::size_t bucket_index =
            margin_bucket_index(margins_[query_index]);
        auto& bucket = result.margin_buckets[bucket_index];
        ++bucket.query_count;
        result.optimal_representative_count +=
            static_cast<std::size_t>(optimal);
        result.non_optimal_count += static_cast<std::size_t>(!optimal);
        bucket.non_optimal_count += static_cast<std::size_t>(!optimal);

        const std::size_t rank = 1 + static_cast<std::size_t>(
            std::ranges::count_if(
                row,
                [returned](double distance) {
                    return distance < returned;
                }));
        ranks.push_back(static_cast<double>(rank));

        if (optimum == 0.0) {
            ++result.zero_optimum_query_count;
            if (!optimal) {
                ++result.zero_optimum_non_optimal_count;
            }
        } else {
            const double ratio = returned / optimum;
            ratios.push_back(ratio);
            ++bucket.ratio_eligible_count;
            bucket_ratio_sums[bucket_index] += ratio;
            if (!optimal) {
                conditional_ratio_sum += ratio;
                ++result.conditional_non_optimal_ratio_count;
            }
        }

        for (std::size_t index = 0;
             index < approximation_epsilons.size();
             ++index) {
            const bool violation = optimum == 0.0
                ? !optimal
                : returned / optimum > 1.0 + approximation_epsilons[index];
            result.approximation_failures[index].violation_count +=
                static_cast<std::size_t>(violation);
        }
    }

    result.distance_ratio = summarize(std::move(ratios));
    result.returned_representative_rank = summarize(std::move(ranks));
    if (result.conditional_non_optimal_ratio_count != 0) {
        const double mean_ratio =
            conditional_ratio_sum /
            static_cast<double>(result.conditional_non_optimal_ratio_count);
        result.conditional_mean_distance_ratio = mean_ratio;
        result.conditional_mean_relative_excess = mean_ratio - 1.0;
    }
    for (std::size_t index = 0; index < result.margin_buckets.size(); ++index) {
        auto& bucket = result.margin_buckets[index];
        if (bucket.ratio_eligible_count != 0) {
            bucket.mean_distance_ratio =
                bucket_ratio_sums[index] /
                static_cast<double>(bucket.ratio_eligible_count);
        }
    }
    return result;
}

ApproximationMetrics evaluate_selected_distances(
    DistanceMetric metric,
    const DenseMatrix& representatives,
    const DenseMatrix& queries,
    std::size_t query_count,
    std::span<const std::size_t> reference_predictions,
    std::span<const std::size_t> candidate_predictions)
{
    if (query_count == 0 || query_count > queries.rows() ||
        reference_predictions.size() != query_count ||
        candidate_predictions.size() != query_count) {
        throw std::invalid_argument(
            "selected-distance diagnostics received incompatible query data");
    }
    if (representatives.rows() == 0 ||
        representatives.cols() != queries.cols()) {
        throw std::invalid_argument(
            "selected-distance diagnostics received incompatible matrices");
    }

    ApproximationMetrics result;
    result.query_count = query_count;
    for (std::size_t index = 0;
         index < approximation_epsilons.size();
         ++index) {
        result.approximation_failures[index].epsilon =
            approximation_epsilons[index];
    }

    std::vector<double> ratios;
    ratios.reserve(query_count);
    double conditional_ratio_sum{};
    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        const std::size_t reference = reference_predictions[query_index];
        const std::size_t candidate = candidate_predictions[query_index];
        if (reference >= representatives.rows() ||
            candidate >= representatives.rows()) {
            throw std::invalid_argument(
                "selected-distance diagnostics received an invalid prediction");
        }
        const auto query = queries.row(query_index);
        const double optimum = exact_distance(
            metric, query, representatives.row(reference));
        const double returned = exact_distance(
            metric, query, representatives.row(candidate));
        const bool no_worse_than_reference = returned <= optimum;
        const bool worse_than_reference = returned > optimum;
        const bool improved = returned < optimum;
        result.optimal_representative_count +=
            static_cast<std::size_t>(no_worse_than_reference);
        result.non_optimal_count +=
            static_cast<std::size_t>(worse_than_reference);
        result.reference_improvement_count +=
            static_cast<std::size_t>(improved);

        if (optimum == 0.0) {
            ++result.zero_optimum_query_count;
            if (worse_than_reference) {
                ++result.zero_optimum_non_optimal_count;
            }
        } else {
            const double ratio = returned / optimum;
            ratios.push_back(ratio);
            if (worse_than_reference) {
                conditional_ratio_sum += ratio;
                ++result.conditional_non_optimal_ratio_count;
            }
        }
        for (std::size_t index = 0;
             index < approximation_epsilons.size();
            ++index) {
            const bool violation = optimum == 0.0
                ? worse_than_reference
                : returned / optimum > 1.0 + approximation_epsilons[index];
            result.approximation_failures[index].violation_count +=
                static_cast<std::size_t>(violation);
        }
    }

    result.distance_ratio = summarize(std::move(ratios));
    if (result.conditional_non_optimal_ratio_count != 0) {
        const double mean_ratio =
            conditional_ratio_sum /
            static_cast<double>(result.conditional_non_optimal_ratio_count);
        result.conditional_mean_distance_ratio = mean_ratio;
        result.conditional_mean_relative_excess = mean_ratio - 1.0;
    }
    return result;
}

std::size_t ExactDistanceTable::payload_bytes() const noexcept
{
    return distances_.size() * sizeof(double) +
           optimum_distances_.size() * sizeof(double) +
           margins_.size() * sizeof(double);
}

}  // namespace ultrahigh_ann::benchmark
