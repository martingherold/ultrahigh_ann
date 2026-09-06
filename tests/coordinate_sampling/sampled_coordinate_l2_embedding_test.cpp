#include "ultrahigh_ann/coordinate_sampling/sampled_coordinate_l2_ann_index.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
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

[[nodiscard]] std::size_t weighted_reference_query(
    const ultrahigh_ann::DenseMatrix& representatives,
    const ultrahigh_ann::CoordinateSample& sample,
    std::span<const float> query)
{
    std::size_t result = 0;
    double minimum_distance = std::numeric_limits<double>::infinity();
    for (std::size_t row = 0; row < representatives.rows(); ++row) {
        double distance = 0.0;
        for (const auto& column : sample.columns) {
            const double difference =
                static_cast<double>(query[column.source_column]) -
                static_cast<double>(
                    representatives.row(row)[column.source_column]);
            distance += static_cast<double>(column.multiplicity) *
                        column.inverse_probability * difference * difference;
        }
        if (distance < minimum_distance) {
            minimum_distance = distance;
            result = row;
        }
    }
    return result;
}

}  // namespace

int main()
{
    using ultrahigh_ann::CoordinateSample;
    using ultrahigh_ann::CpuDenseL2QueryStrategy;
    using ultrahigh_ann::DenseMatrix;
    using ultrahigh_ann::SampledColumn;
    using ultrahigh_ann::SampledCoordinateL2AnnIndex;

    static_assert(!std::is_copy_constructible_v<SampledCoordinateL2AnnIndex>);
    static_assert(std::is_move_constructible_v<SampledCoordinateL2AnnIndex>);

    std::vector<float> values{
        0.0F, 100.0F, 0.0F, 0.0F, -100.0F, 0.0F,
        4.0F, 0.0F,   2.0F, 1.0F, 0.0F,    8.0F,
    };
    const DenseMatrix representatives(std::move(values), 4, 3);
    const CoordinateSample sample{
        .columns =
            {
                SampledColumn{
                    .source_column = 0,
                    .multiplicity = 3,
                    .inverse_probability = 0.7,
                },
                SampledColumn{
                    .source_column = 2,
                    .multiplicity = 2,
                    .inverse_probability = 1.3,
                },
            },
    };

    SampledCoordinateL2AnnIndex original(representatives, sample);
    const SampledCoordinateL2AnnIndex index(std::move(original));

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 12> queries{
        0.2F, nan,      0.1F, 3.8F, 1000.0F, 1.9F,
        1.0F, -1000.0F, 7.5F, 0.0F, 42.0F,   0.0F,
    };
    std::array<std::size_t, 4> results{};
    index.query_batch(queries, 4, results);

    bool passed = true;
    constexpr std::array strategies{
        CpuDenseL2QueryStrategy::sequential,
        CpuDenseL2QueryStrategy::parallel_queries,
        CpuDenseL2QueryStrategy::parallel_representatives,
        CpuDenseL2QueryStrategy::automatic,
    };
    for (const CpuDenseL2QueryStrategy strategy : strategies) {
        std::array<std::size_t, 4> strategy_results{};
        index.query_batch(queries, 4, strategy_results, strategy);
        passed &=
            expect(strategy_results == results,
                   "every sampled CPU strategy must preserve batch results");
    }
    for (std::size_t query_index = 0; query_index < results.size();
         ++query_index) {
        const auto query =
            std::span<const float>{queries}.subspan(query_index * 3, 3);
        passed &=
            expect(results[query_index] ==
                       weighted_reference_query(representatives, sample, query),
                   "the reduced dense scan must match weighted sampled L2 away "
                   "from numerical boundary cases");
        passed &= expect(index.query(query) == results[query_index],
                         "sampled single and batch queries must agree");
        passed &= expect(
            index.query(query,
                        CpuDenseL2QueryStrategy::parallel_representatives) ==
                results[query_index],
            "sampled single-query representative parallelism must agree");
    }
    passed &=
        expect(results[3] == 0,
               "identical sampled representatives must tie at the lowest row");

    const auto usage = index.space_usage();
    passed &= expect(
        usage.index_payload_bytes ==
            4 * 2 * sizeof(float) + 2 * sizeof(std::size_t) + 2 * sizeof(float),
        "sampled payload must contain one reduced matrix and embedding "
        "metadata");
    passed &= expect(usage.query_workspace_payload_bytes == 2 * sizeof(float),
                     "sampled workspace must contain one packed reduced query");
    passed &= expect(
        usage.unique_query_coordinates == 2 && usage.sampled_multiplicity == 5,
        "sampled usage must retain coordinate and multiplicity semantics");

    return passed ? 0 : 1;
}
