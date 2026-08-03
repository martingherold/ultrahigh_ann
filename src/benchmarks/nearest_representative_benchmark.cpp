#include "benchmark_metrics.hpp"
#include "benchmark_setup.hpp"
#include "datasets/representative_query_dataset.hpp"
#include "exact/l1/exact_l1_index.hpp"
#include "exact/l2/exact_l2_index.hpp"
#include "hnsvw25/l1/flat/l1_ann_index.hpp"
#include "coordinate_sampling/uniform_l1_ann_index.hpp"
#include "hnsvw25/l1/hierarchical/hierarchical_l1_ann_index.hpp"
#include "hnsvw25/l1/importance_sampling.hpp"
#include "hnsvw25/l2/flat/l2_ann_index.hpp"
#include "coordinate_sampling/uniform_l2_ann_index.hpp"
#include "hnsvw25/l2/hierarchical/hierarchical_l2_ann_index.hpp"
#include "hnsvw25/l2/importance_sampling.hpp"
#include "coordinate_sampling/uniform_probabilities.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using DistanceMetric = ultrahigh_ann::benchmark::DistanceMetric;

constexpr DistanceMetric default_distance{DistanceMetric::l1};
constexpr std::string_view default_output_path{
    "results/raw/nearest_representative_benchmark.json"};

struct Options {
    std::filesystem::path dataset_directory{
        "data/processed/tcga_kidney_v1"};
    std::size_t repetitions{64};
    std::size_t projection_dimension{31};
    std::uint64_t seed{42};
    std::size_t maximum_queries{};
    DistanceMetric distance{default_distance};
    std::filesystem::path output_path{default_output_path};
    std::optional<std::filesystem::path> setup_path;
};

struct QueryResult {
    std::vector<std::size_t> predictions;
    std::size_t correct{};
    std::size_t exact_agreements{};
    std::size_t exact_choice_agreements{};
    std::size_t exact_label_agreements{};
    double elapsed_ms{};
    std::optional<ultrahigh_ann::benchmark::ApproximationMetrics>
        approximation_metrics;
};

struct ProbabilityExecution {
    std::vector<double> probabilities;
    double sampling_mass{};
    double build_ms{};
};

void print_usage(std::string_view program)
{
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Load a transformed representative/query dataset and run exact, "
           "flat, and hierarchical L1 or L2 queries.\n\n"
        << "Batch mode:\n"
        << "  --setup FILE            Run a versioned setup file; no other "
           "options may be supplied\n\n"
        << "Options:\n"
        << "  --dataset DIR           Dataset directory (default: "
           "data/processed/tcga_kidney_v1)\n"
        << "  --repetitions N         Importance-sampling repetitions "
           "(default: 64)\n"
        << "  --projection-dimension N  Hierarchical Cauchy/JL dimension "
           "(default: 31)\n"
        << "  --seed N                Random seed (default: 42)\n"
        << "  --distance METRIC      l1 or l2 (default: "
        << ultrahigh_ann::benchmark::distance_name(default_distance)
        << ")\n"
        << "  --max-queries N         Run at most N queries; 0 means all "
           "(default: 0)\n"
        << "  --output FILE           JSON report path (default: "
        << default_output_path << ")\n"
        << "  -h, --help              Show this help\n";
}

[[nodiscard]] DistanceMetric parse_distance(std::string_view value)
{
    if (value == "l1") {
        return DistanceMetric::l1;
    }
    if (value == "l2") {
        return DistanceMetric::l2;
    }
    throw std::invalid_argument("--distance expects l1 or l2");
}

template<class Integer>
[[nodiscard]] Integer parse_integer(
    std::string_view text,
    std::string_view option)
{
    Integer value{};
    const auto [position, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        value);
    if (error != std::errc{} || position != text.data() + text.size()) {
        throw std::invalid_argument(
            std::string(option) + " expects a nonnegative integer");
    }
    return value;
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options options;
    bool setup_seen = false;
    bool legacy_option_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(
                std::string(argument) + " expects a value");
        }
        const std::string_view value{argv[++index]};
        if (argument == "--setup") {
            if (setup_seen) {
                throw std::invalid_argument(
                    "--setup may only be specified once");
            }
            options.setup_path = value;
            setup_seen = true;
        } else if (argument == "--dataset") {
            options.dataset_directory = value;
            legacy_option_seen = true;
        } else if (argument == "--repetitions") {
            options.repetitions =
                parse_integer<std::size_t>(value, argument);
            legacy_option_seen = true;
        } else if (argument == "--projection-dimension") {
            options.projection_dimension =
                parse_integer<std::size_t>(value, argument);
            legacy_option_seen = true;
        } else if (argument == "--seed") {
            options.seed = parse_integer<std::uint64_t>(value, argument);
            legacy_option_seen = true;
        } else if (argument == "--distance") {
            options.distance = parse_distance(value);
            legacy_option_seen = true;
        } else if (argument == "--max-queries") {
            options.maximum_queries =
                parse_integer<std::size_t>(value, argument);
            legacy_option_seen = true;
        } else if (argument == "--output") {
            options.output_path = value;
            legacy_option_seen = true;
        } else {
            throw std::invalid_argument(
                "unknown option: " + std::string(argument));
        }
    }

    if (setup_seen && legacy_option_seen) {
        throw std::invalid_argument(
            "--setup cannot be combined with single-run options");
    }
    if (setup_seen) {
        return options;
    }
    if (options.repetitions == 0) {
        throw std::invalid_argument("--repetitions must be positive");
    }
    if (options.projection_dimension == 0) {
        throw std::invalid_argument(
            "--projection-dimension must be positive");
    }
    return options;
}

template<class QueryFunction>
[[nodiscard]] QueryResult run_queries(
    const ultrahigh_ann::DenseMatrix& queries,
    std::span<const std::size_t> query_labels,
    std::span<const std::size_t> representative_labels,
    std::size_t query_count,
    std::span<const std::size_t> exact_predictions,
    QueryFunction&& query)
{
    QueryResult result;
    result.predictions.reserve(query_count);
    const auto start = Clock::now();
    for (std::size_t row = 0; row < query_count; ++row) {
        const std::size_t prediction = query(queries.row(row));
        if (prediction >= representative_labels.size()) {
            throw std::logic_error(
                "index returned an out-of-range representative row");
        }
        result.predictions.push_back(prediction);
        result.correct += static_cast<std::size_t>(
            representative_labels[prediction] == query_labels[row]);
        if (!exact_predictions.empty()) {
            if (exact_predictions[row] >= representative_labels.size()) {
                throw std::logic_error(
                    "exact prediction contains an out-of-range representative row");
            }
            result.exact_choice_agreements += static_cast<std::size_t>(
                prediction == exact_predictions[row]);
            result.exact_label_agreements += static_cast<std::size_t>(
                representative_labels[prediction] ==
                representative_labels[exact_predictions[row]]);
        }
    }
    const auto end = Clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

[[nodiscard]] double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(
               Clock::now() - start)
        .count();
}

[[nodiscard]] double accuracy(const QueryResult& result)
{
    return static_cast<double>(result.correct) /
           static_cast<double>(result.predictions.size());
}

[[nodiscard]] double agreement_with_exact(const QueryResult& result)
{
    return static_cast<double>(result.exact_agreements) /
           static_cast<double>(result.predictions.size());
}

[[nodiscard]] double agreement_with_exact_choice(const QueryResult& result)
{
    return static_cast<double>(result.exact_choice_agreements) /
           static_cast<double>(result.predictions.size());
}

[[nodiscard]] double label_agreement_with_exact(const QueryResult& result)
{
    return static_cast<double>(result.exact_label_agreements) /
           static_cast<double>(result.predictions.size());
}

[[nodiscard]] std::size_t representative_class_count(
    const ultrahigh_ann::RepresentativeQueryDataset& dataset)
{
    return std::unordered_set<std::size_t>{
        dataset.representative_labels.begin(),
        dataset.representative_labels.end()}
        .size();
}

[[nodiscard]] std::string utc_timestamp()
{
    const std::time_t time = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    if (gmtime_s(&utc, &time) != 0) {
        throw std::runtime_error("failed to create UTC timestamp");
    }
#else
    if (gmtime_r(&time, &utc) == nullptr) {
        throw std::runtime_error("failed to create UTC timestamp");
    }
#endif
    std::array<char, 32> buffer{};
    if (std::strftime(
            buffer.data(),
            buffer.size(),
            "%Y-%m-%dT%H:%M:%SZ",
            &utc) == 0) {
        throw std::runtime_error("failed to format UTC timestamp");
    }
    return buffer.data();
}

void write_json_string(std::ostream& output, std::string_view value)
{
    constexpr std::string_view hexadecimal{"0123456789abcdef"};
    output.put('"');
    for (const char raw_character : value) {
        const auto character =
            static_cast<unsigned char>(raw_character);
        switch (character) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20U) {
                output << "\\u00"
                       << hexadecimal[character >> 4U]
                       << hexadecimal[character & 0x0fU];
            } else {
                output.put(static_cast<char>(character));
            }
        }
    }
    output.put('"');
}

[[nodiscard]] std::filesystem::path portable_report_path(
    const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path absolute =
        std::filesystem::absolute(path, error).lexically_normal();
    if (error) {
        return path.lexically_normal();
    }

    const std::filesystem::path working_directory =
        std::filesystem::current_path(error);
    if (error) {
        return absolute;
    }

    const std::filesystem::path relative =
        absolute.lexically_relative(working_directory);
    if (!relative.empty() && relative.begin()->generic_string() != "..") {
        return relative;
    }
    return absolute;
}

void write_optional_json_number(
    std::ostream& output,
    const std::optional<double>& value)
{
    if (value.has_value()) {
        output << *value;
    } else {
        output << "null";
    }
}

void write_distribution_json(
    std::ostream& output,
    const ultrahigh_ann::benchmark::DistributionSummary& summary,
    std::size_t indentation)
{
    const std::string field(indentation + 2, ' ');
    const std::string outer(indentation, ' ');
    output << "{\n"
           << field << "\"quantile_method\": \"nearest_rank\",\n"
           << field << "\"count\": " << summary.count << ",\n"
           << field << "\"mean\": ";
    write_optional_json_number(output, summary.mean);
    output << ",\n" << field << "\"median\": ";
    write_optional_json_number(output, summary.median);
    output << ",\n" << field << "\"percentile_95\": ";
    write_optional_json_number(output, summary.percentile_95);
    output << ",\n" << field << "\"percentile_99\": ";
    write_optional_json_number(output, summary.percentile_99);
    output << ",\n" << field << "\"maximum\": ";
    write_optional_json_number(output, summary.maximum);
    output << '\n' << outer << '}';
}

void write_margin_bounds_json(
    std::ostream& output,
    const ultrahigh_ann::benchmark::MarginBucketMetrics& bucket,
    const std::string& field)
{
    output << field << "\"lower_inclusive\": "
           << bucket.lower_inclusive << ",\n"
           << field << "\"upper_exclusive\": ";
    write_optional_json_number(output, bucket.upper_exclusive);
}

void write_query_geometry_json(
    std::ostream& output,
    const ultrahigh_ann::benchmark::QueryGeometryMetrics& geometry,
    std::size_t indentation)
{
    const std::string outer(indentation, ' ');
    const std::string field(indentation + 2, ' ');
    const std::string nested(indentation + 4, ' ');
    output << "{\n"
           << field << "\"query_count\": " << geometry.query_count
           << ",\n"
           << field << "\"zero_optimum_query_count\": "
           << geometry.zero_optimum_query_count << ",\n"
           << field << "\"non_unique_optimum_query_count\": "
           << geometry.non_unique_optimum_query_count << ",\n"
           << field << "\"multiplicative_margin\": {\n"
           << nested << "\"definition\": \"second_nearest_distance / nearest_distance\",\n"
           << nested << "\"finite_distribution\": ";
    write_distribution_json(
        output,
        geometry.finite_multiplicative_margin,
        indentation + 4);
    output << ",\n"
           << nested << "\"infinite_count\": "
           << geometry.infinite_multiplicative_margin_count << ",\n"
           << nested << "\"buckets\": [\n";
    for (std::size_t index = 0;
         index < geometry.margin_buckets.size();
         ++index) {
        const auto& bucket = geometry.margin_buckets[index];
        output << nested << "  {\n";
        write_margin_bounds_json(output, bucket, nested + "    ");
        output << ",\n" << nested << "    \"query_count\": "
               << bucket.query_count << "\n"
               << nested << "  }"
               << (index + 1 == geometry.margin_buckets.size()
                       ? "\n"
                       : ",\n");
    }
    output << nested << "]\n"
           << field << "}\n"
           << outer << '}';
}

void write_approximation_json(
    std::ostream& output,
    const QueryResult& result,
    std::size_t indentation)
{
    if (!result.approximation_metrics.has_value()) {
        output << "null";
        return;
    }
    const auto& metrics = *result.approximation_metrics;
    const std::string outer(indentation, ' ');
    const std::string field(indentation + 2, ' ');
    const std::string nested(indentation + 4, ' ');
    const double query_count = static_cast<double>(metrics.query_count);
    output << "{\n"
           << field << "\"optimal_representative_count\": "
           << metrics.optimal_representative_count << ",\n"
           << field << "\"optimal_representative_rate\": "
           << static_cast<double>(metrics.optimal_representative_count) /
                  query_count
           << ",\n"
           << field << "\"non_optimal_count\": "
           << metrics.non_optimal_count << ",\n"
           << field << "\"non_optimal_rate\": "
           << static_cast<double>(metrics.non_optimal_count) / query_count
           << ",\n"
           << field << "\"distance_ratio\": {\n"
           << nested << "\"definition\": \"returned_distance / exact_distance\",\n"
           << nested << "\"zero_optimum_queries_excluded\": "
           << metrics.zero_optimum_query_count << ",\n"
           << nested << "\"distribution\": ";
    write_distribution_json(output, metrics.distance_ratio, indentation + 4);
    output << "\n" << field << "},\n"
           << field << "\"non_optimal_distance_ratio\": {\n"
           << nested << "\"eligible_non_optimal_count\": "
           << metrics.conditional_non_optimal_ratio_count << ",\n"
           << nested << "\"zero_optimum_non_optimal_count\": "
           << metrics.zero_optimum_non_optimal_count << ",\n"
           << nested << "\"mean\": ";
    write_optional_json_number(
        output,
        metrics.conditional_mean_distance_ratio);
    output << ",\n" << nested << "\"mean_relative_excess\": ";
    write_optional_json_number(
        output,
        metrics.conditional_mean_relative_excess);
    output << "\n" << field << "},\n"
           << field << "\"approximation_guarantee_failures\": [\n";
    for (std::size_t index = 0;
         index < metrics.approximation_failures.size();
         ++index) {
        const auto& failure = metrics.approximation_failures[index];
        output << nested << "{\"epsilon\": " << failure.epsilon
               << ", \"violation_count\": " << failure.violation_count
               << ", \"violation_rate\": "
               << static_cast<double>(failure.violation_count) / query_count
               << '}'
               << (index + 1 == metrics.approximation_failures.size()
                       ? "\n"
                       : ",\n");
    }
    output << field << "],\n"
           << field << "\"returned_representative_rank\": {\n"
           << nested << "\"definition\": \"one plus the number of representatives at strictly smaller exact distance\",\n"
           << nested << "\"distribution\": ";
    write_distribution_json(
        output,
        metrics.returned_representative_rank,
        indentation + 4);
    output << "\n" << field << "},\n"
           << field << "\"by_multiplicative_margin\": [\n";
    for (std::size_t index = 0;
         index < metrics.margin_buckets.size();
         ++index) {
        const auto& bucket = metrics.margin_buckets[index];
        output << nested << "{\n";
        write_margin_bounds_json(output, bucket, nested + "  ");
        output << ",\n"
               << nested << "  \"query_count\": " << bucket.query_count
               << ",\n"
               << nested << "  \"non_optimal_count\": "
               << bucket.non_optimal_count << ",\n"
               << nested << "  \"non_optimal_rate\": ";
        if (bucket.query_count == 0) {
            output << "null";
        } else {
            output << static_cast<double>(bucket.non_optimal_count) /
                          static_cast<double>(bucket.query_count);
        }
        output << ",\n"
               << nested << "  \"ratio_eligible_count\": "
               << bucket.ratio_eligible_count << ",\n"
               << nested << "  \"mean_distance_ratio\": ";
        write_optional_json_number(output, bucket.mean_distance_ratio);
        output << "\n" << nested << '}'
               << (index + 1 == metrics.margin_buckets.size()
                       ? "\n"
                       : ",\n");
    }
    output << field << "]\n"
           << outer << '}';
}

void write_method_json(
    std::ostream& output,
    std::string_view method,
    double build_ms,
    const QueryResult& result,
    const ultrahigh_ann::IndexSpaceUsage& space_usage,
    std::size_t dimension,
    bool has_exact_baseline,
    bool final_entry)
{
    const double query_count =
        static_cast<double>(result.predictions.size());
    output << "    {\n      \"method\": ";
    write_json_string(output, method);
    output << ",\n"
           << "      \"build_ms\": " << build_ms << ",\n"
           << "      \"query_total_ms\": " << result.elapsed_ms << ",\n"
           << "      \"microseconds_per_query\": "
           << 1000.0 * result.elapsed_ms / query_count << ",\n"
           << "      \"correct\": " << result.correct << ",\n"
           << "      \"query_count\": " << result.predictions.size()
           << ",\n"
           << "      \"accuracy\": " << accuracy(result) << ",\n"
           << "      \"agreement_with_exact\": ";
    if (has_exact_baseline) {
        output << agreement_with_exact(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << "      \"agreement_with_exact_index_choice\": ";
    if (has_exact_baseline) {
        output << agreement_with_exact_choice(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << "      \"label_agreement_with_exact\": ";
    if (has_exact_baseline) {
        output << label_agreement_with_exact(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << "      \"space\": {\n"
           << "        \"index_payload_bytes\": "
           << space_usage.index_payload_bytes << ",\n"
           << "        \"query_workspace_payload_bytes\": "
           << space_usage.query_workspace_payload_bytes << "\n"
           << "      },\n"
           << "      \"coordinate_access\": {\n"
           << "        \"unique_coordinates\": "
           << space_usage.unique_query_coordinates << ",\n"
           << "        \"dimension_fraction\": "
           << static_cast<double>(space_usage.unique_query_coordinates) /
                  static_cast<double>(dimension)
           << ",\n"
           << "        \"sampled_multiplicity\": "
           << space_usage.sampled_multiplicity << "\n"
           << "      },\n"
           << "      \"approximation\": ";
    write_approximation_json(output, result, 6);
    output << "\n"
           << "    }" << (final_entry ? "\n" : ",\n");
}

void write_json_report(
    const Options& options,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::size_t query_count,
    double load_ms,
    double exact_build_ms,
    const QueryResult& exact,
    const ultrahigh_ann::IndexSpaceUsage& exact_space_usage,
    double diagnostic_build_ms,
    double diagnostic_evaluation_ms,
    const ultrahigh_ann::benchmark::ExactDistanceTable& distance_table,
    const ProbabilityExecution& probability_execution,
    double flat_build_ms,
    const QueryResult& flat,
    const ultrahigh_ann::IndexSpaceUsage& flat_space_usage,
    double hierarchical_build_ms,
    const QueryResult& hierarchical,
    const ultrahigh_ann::IndexSpaceUsage& hierarchical_space_usage)
{
    const std::filesystem::path parent = options.output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(options.output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open JSON report: " + options.output_path.string());
    }
    output << std::setprecision(10);
    output << "{\n"
           << "  \"schema_version\": 4,\n"
           << "  \"generated_at_utc\": ";
    write_json_string(output, utc_timestamp());
    output << ",\n  \"dataset\": {\n"
           << "    \"directory\": ";
    write_json_string(
        output,
        portable_report_path(options.dataset_directory).generic_string());
    output << ",\n"
           << "    \"representatives_file\": \"representatives.npy\",\n"
           << "    \"representative_labels_file\": ";
    if (std::filesystem::is_regular_file(
            options.dataset_directory / "representative_labels.npy")) {
        output << "\"representative_labels.npy\"";
    } else {
        output << "null";
    }
    output << ",\n"
           << "    \"queries_file\": \"queries.npy\",\n"
           << "    \"query_labels_file\": \"query_labels.npy\",\n"
           << "    \"representative_count\": "
           << dataset.representatives.rows() << ",\n"
           << "    \"representative_class_count\": "
           << representative_class_count(dataset) << ",\n"
           << "    \"query_count_available\": " << dataset.queries.rows()
           << ",\n"
           << "    \"dimension\": " << dataset.representatives.cols()
           << ",\n"
           << "    \"load_ms\": " << load_ms << "\n"
           << "  },\n"
           << "  \"settings\": {\n"
           << "    \"distance\": ";
    write_json_string(
        output,
        ultrahigh_ann::benchmark::distance_name(options.distance));
    output << ",\n"
           << "    \"queries_run\": " << query_count << ",\n"
           << "    \"query_limit\": ";
    if (options.maximum_queries == 0) {
        output << "null";
    } else {
        output << options.maximum_queries;
    }
    output << ",\n"
           << "    \"seed\": " << options.seed << ",\n"
           << "    \"flat\": {\n"
           << "      \"repetitions\": " << options.repetitions << "\n"
           << "    },\n"
           << "    \"hierarchical\": {\n"
           << "      \"repetitions\": " << options.repetitions << ",\n"
           << "      \"projection_dimension\": "
           << options.projection_dimension << "\n"
           << "    }\n"
           << "  },\n"
           << "  \"shared_preprocessing\": {\n"
           << "    \"sampling_probabilities\": {\n"
           << "      \"computed_once\": true,\n"
           << "      \"build_ms\": "
           << probability_execution.build_ms << ",\n"
           << "      \"coordinate_count\": "
           << probability_execution.probabilities.size() << ",\n"
           << "      \"sampling_mass\": "
           << probability_execution.sampling_mass << ",\n"
           << "      \"sampling_mass_per_representative\": "
           << probability_execution.sampling_mass /
                  static_cast<double>(dataset.representatives.rows())
           << ",\n"
           << "      \"mass_matched_uniform_probability\": "
           << probability_execution.sampling_mass /
                  static_cast<double>(dataset.representatives.cols())
           << ",\n"
           << "      \"payload_bytes\": "
           << probability_execution.probabilities.size() * sizeof(double)
           << "\n"
           << "    }\n"
           << "  },\n"
           << "  \"diagnostics\": {\n"
           << "    \"included_in_query_timings\": false,\n"
           << "    \"distance_table_build_ms\": "
           << diagnostic_build_ms << ",\n"
           << "    \"run_evaluation_total_ms\": "
           << diagnostic_evaluation_ms << ",\n"
           << "    \"distance_table_payload_bytes\": "
           << distance_table.payload_bytes() << ",\n"
           << "    \"query_geometry\": ";
    write_query_geometry_json(output, distance_table.query_geometry(), 4);
    output << "\n  },\n"
           << "  \"results\": [\n";
    write_method_json(
        output,
        "exact",
        exact_build_ms,
        exact,
        exact_space_usage,
        dataset.representatives.cols(),
        false,
        false);
    write_method_json(
        output,
        "flat",
        flat_build_ms,
        flat,
        flat_space_usage,
        dataset.representatives.cols(),
        true,
        false);
    write_method_json(
        output,
        "hierarchical",
        hierarchical_build_ms,
        hierarchical,
        hierarchical_space_usage,
        dataset.representatives.cols(),
        true,
        true);
    output << "  ]\n}\n";
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed to write JSON report: " + options.output_path.string());
    }
}

void print_result(
    std::string_view method,
    double build_ms,
    const QueryResult& result,
    bool has_exact_baseline)
{
    const double query_count =
        static_cast<double>(result.predictions.size());
    const double accuracy_percent = 100.0 * accuracy(result);
    const double microseconds_per_query =
        1000.0 * result.elapsed_ms / query_count;

    std::cout << std::left << std::setw(14) << method << std::right
              << std::setw(12) << std::fixed << std::setprecision(3)
              << build_ms << std::setw(13) << result.elapsed_ms
              << std::setw(13) << microseconds_per_query
              << std::setw(11) << std::setprecision(2)
              << accuracy_percent << '%';
    if (has_exact_baseline) {
        const double agreement = 100.0 * agreement_with_exact(result);
        std::cout << std::setw(12) << agreement << '%';
    } else {
        std::cout << std::setw(13) << "-";
    }
    std::cout << '\n';
}

void print_space_usage(
    std::string_view method,
    const ultrahigh_ann::IndexSpaceUsage& usage,
    std::size_t dimension)
{
    constexpr double bytes_per_kib = 1024.0;
    const double coordinate_percent =
        100.0 * static_cast<double>(usage.unique_query_coordinates) /
        static_cast<double>(dimension);
    std::cout << std::left << std::setw(14) << method << std::right
              << std::setw(14) << std::fixed << std::setprecision(2)
              << static_cast<double>(usage.index_payload_bytes) /
                     bytes_per_kib
              << std::setw(16)
              << static_cast<double>(usage.query_workspace_payload_bytes) /
                     bytes_per_kib
              << std::setw(14) << usage.unique_query_coordinates
              << std::setw(12) << coordinate_percent << '%'
              << std::setw(16) << usage.sampled_multiplicity << '\n';
}

struct ApproximateExecution {
    ultrahigh_ann::benchmark::ApproximateRun run;
    double build_ms{};
    QueryResult query_result;
    ultrahigh_ann::IndexSpaceUsage space_usage;
};

struct ExactExecution {
    double build_ms{};
    QueryResult query_result;
    ultrahigh_ann::IndexSpaceUsage space_usage;
};

void add_approximation_metrics(
    QueryResult& result,
    const ultrahigh_ann::benchmark::ExactDistanceTable& distance_table)
{
    result.approximation_metrics =
        distance_table.evaluate(result.predictions);
    result.exact_agreements =
        result.approximation_metrics->optimal_representative_count;
}

[[nodiscard]] ProbabilityExecution compute_probabilities(
    DistanceMetric distance,
    const ultrahigh_ann::DenseMatrix& representatives)
{
    const auto build_start = Clock::now();
    std::vector<double> probabilities =
        distance == DistanceMetric::l1
            ? ultrahigh_ann::compute_l1_importance_probabilities(
                  representatives)
            : ultrahigh_ann::compute_l2_importance_probabilities(
                  representatives);
    const double sampling_mass =
        ultrahigh_ann::compute_sampling_mass(probabilities);
    return ProbabilityExecution{
        .probabilities = std::move(probabilities),
        .sampling_mass = sampling_mass,
        .build_ms = elapsed_ms(build_start),
    };
}

template<class Index>
[[nodiscard]] ExactExecution execute_exact_index(
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::size_t query_count)
{
    const auto build_start = Clock::now();
    const Index index(dataset.representatives);
    const double build_ms = elapsed_ms(build_start);
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    QueryResult result = run_queries(
        dataset.queries,
        dataset.query_labels,
        dataset.representative_labels,
        query_count,
        {},
        [&index](std::span<const float> query) {
            return index.query(query);
        });
    return ExactExecution{
        .build_ms = build_ms,
        .query_result = std::move(result),
        .space_usage = space_usage,
    };
}

[[nodiscard]] ExactExecution execute_exact(
    DistanceMetric distance,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::size_t query_count)
{
    if (distance == DistanceMetric::l1) {
        return execute_exact_index<ultrahigh_ann::ExactL1Index>(
            dataset,
            query_count);
    }
    return execute_exact_index<ultrahigh_ann::ExactL2Index>(
        dataset,
        query_count);
}

template<class Index>
[[nodiscard]] ApproximateExecution execute_flat_run(
    const ultrahigh_ann::benchmark::ApproximateRun& run,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::span<const double> importance_probabilities,
    std::size_t query_count,
    std::span<const std::size_t> exact_predictions)
{
    std::mt19937_64 random_engine(run.seed);
    const auto build_start = Clock::now();
    const Index index(
        dataset.representatives,
        importance_probabilities,
        run.repetitions,
        random_engine);
    const double build_ms = elapsed_ms(build_start);
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    QueryResult result = run_queries(
        dataset.queries,
        dataset.query_labels,
        dataset.representative_labels,
        query_count,
        exact_predictions,
        [&index](std::span<const float> query) {
            return index.query(query);
        });
    return ApproximateExecution{
        .run = run,
        .build_ms = build_ms,
        .query_result = std::move(result),
        .space_usage = space_usage,
    };
}

template<class Index>
[[nodiscard]] ApproximateExecution execute_hierarchical_run(
    const ultrahigh_ann::benchmark::ApproximateRun& run,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::span<const double> importance_probabilities,
    std::size_t query_count,
    std::span<const std::size_t> exact_predictions)
{
    std::mt19937_64 random_engine(run.seed);
    const auto build_start = Clock::now();
    const Index index(
        dataset.representatives,
        importance_probabilities,
        run.repetitions,
        run.projection_dimension,
        random_engine);
    auto workspace = index.make_query_workspace();
    const double build_ms = elapsed_ms(build_start);
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    QueryResult result = run_queries(
        dataset.queries,
        dataset.query_labels,
        dataset.representative_labels,
        query_count,
        exact_predictions,
        [&index, &workspace](std::span<const float> query) {
            return index.query(query, workspace);
        });
    return ApproximateExecution{
        .run = run,
        .build_ms = build_ms,
        .query_result = std::move(result),
        .space_usage = space_usage,
    };
}

template<class Index>
[[nodiscard]] ApproximateExecution execute_uniform_run(
    const ultrahigh_ann::benchmark::ApproximateRun& run,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    double sampling_mass,
    std::size_t query_count,
    std::span<const std::size_t> exact_predictions)
{
    std::mt19937_64 random_engine(run.seed);
    const auto build_start = Clock::now();
    const Index index(
        dataset.representatives,
        sampling_mass,
        run.repetitions,
        random_engine);
    const double build_ms = elapsed_ms(build_start);
    const ultrahigh_ann::IndexSpaceUsage space_usage = index.space_usage();
    QueryResult result = run_queries(
        dataset.queries,
        dataset.query_labels,
        dataset.representative_labels,
        query_count,
        exact_predictions,
        [&index](std::span<const float> query) {
            return index.query(query);
        });
    return ApproximateExecution{
        .run = run,
        .build_ms = build_ms,
        .query_result = std::move(result),
        .space_usage = space_usage,
    };
}

[[nodiscard]] ApproximateExecution execute_approximate_run(
    DistanceMetric distance,
    const ultrahigh_ann::benchmark::ApproximateRun& run,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::span<const double> importance_probabilities,
    double sampling_mass,
    std::size_t query_count,
    std::span<const std::size_t> exact_predictions)
{
    if (run.method == ultrahigh_ann::benchmark::ApproximateMethod::flat) {
        if (distance == DistanceMetric::l1) {
            return execute_flat_run<ultrahigh_ann::FlatL1AnnIndex>(
                run,
                dataset,
                importance_probabilities,
                query_count,
                exact_predictions);
        }
        return execute_flat_run<ultrahigh_ann::FlatL2AnnIndex>(
            run,
            dataset,
            importance_probabilities,
            query_count,
            exact_predictions);
    }
    if (run.method ==
        ultrahigh_ann::benchmark::ApproximateMethod::uniform) {
        if (distance == DistanceMetric::l1) {
            return execute_uniform_run<ultrahigh_ann::UniformL1AnnIndex>(
                run,
                dataset,
                sampling_mass,
                query_count,
                exact_predictions);
        }
        return execute_uniform_run<ultrahigh_ann::UniformL2AnnIndex>(
            run,
            dataset,
            sampling_mass,
            query_count,
            exact_predictions);
    }
    if (distance == DistanceMetric::l1) {
        return execute_hierarchical_run<
            ultrahigh_ann::HierarchicalL1AnnIndex>(
            run,
            dataset,
            importance_probabilities,
            query_count,
            exact_predictions);
    }
    return execute_hierarchical_run<
        ultrahigh_ann::HierarchicalL2AnnIndex>(
        run,
        dataset,
        importance_probabilities,
        query_count,
        exact_predictions);
}

void write_indented_result_json(
    std::ostream& output,
    std::size_t indentation,
    double build_ms,
    const QueryResult& result,
    const ultrahigh_ann::IndexSpaceUsage& space_usage,
    std::size_t dimension,
    bool has_exact_baseline)
{
    const std::string outer(indentation, ' ');
    const std::string field(indentation + 2, ' ');
    const std::string nested(indentation + 4, ' ');
    const double query_count =
        static_cast<double>(result.predictions.size());
    output << "{\n"
           << field << "\"build_ms\": " << build_ms << ",\n"
           << field << "\"query_total_ms\": " << result.elapsed_ms
           << ",\n"
           << field << "\"microseconds_per_query\": "
           << 1000.0 * result.elapsed_ms / query_count << ",\n"
           << field << "\"correct\": " << result.correct << ",\n"
           << field << "\"query_count\": " << result.predictions.size()
           << ",\n"
           << field << "\"accuracy\": " << accuracy(result) << ",\n"
           << field << "\"agreement_with_exact\": ";
    if (has_exact_baseline) {
        output << agreement_with_exact(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << field << "\"agreement_with_exact_index_choice\": ";
    if (has_exact_baseline) {
        output << agreement_with_exact_choice(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << field << "\"label_agreement_with_exact\": ";
    if (has_exact_baseline) {
        output << label_agreement_with_exact(result);
    } else {
        output << "null";
    }
    output << ",\n"
           << field << "\"space\": {\n"
           << nested << "\"index_payload_bytes\": "
           << space_usage.index_payload_bytes << ",\n"
           << nested << "\"query_workspace_payload_bytes\": "
           << space_usage.query_workspace_payload_bytes << "\n"
           << field << "},\n"
           << field << "\"coordinate_access\": {\n"
           << nested << "\"unique_coordinates\": "
           << space_usage.unique_query_coordinates << ",\n"
           << nested << "\"dimension_fraction\": "
           << static_cast<double>(space_usage.unique_query_coordinates) /
                  static_cast<double>(dimension)
           << ",\n"
           << nested << "\"sampled_multiplicity\": "
           << space_usage.sampled_multiplicity << "\n"
           << field << "},\n"
           << field << "\"approximation\": ";
    write_approximation_json(output, result, indentation + 2);
    output << '\n'
           << outer << '}';
}

void write_batch_json_report(
    const ultrahigh_ann::benchmark::BenchmarkSetup& setup,
    const ultrahigh_ann::RepresentativeQueryDataset& dataset,
    std::size_t query_count,
    double load_ms,
    double exact_build_ms,
    const QueryResult& exact,
    const ultrahigh_ann::IndexSpaceUsage& exact_space_usage,
    double diagnostic_build_ms,
    double diagnostic_evaluation_ms,
    const ultrahigh_ann::benchmark::ExactDistanceTable& distance_table,
    const ProbabilityExecution& probability_execution,
    const std::vector<ApproximateExecution>& executions)
{
    const std::filesystem::path parent = setup.output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(setup.output_path, std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open JSON report: " + setup.output_path.string());
    }
    output << std::setprecision(10);
    output << "{\n"
           << "  \"schema_version\": 4,\n"
           << "  \"generated_at_utc\": ";
    write_json_string(output, utc_timestamp());
    output << ",\n  \"setup_file\": ";
    write_json_string(
        output,
        portable_report_path(setup.setup_path).generic_string());
    output << ",\n  \"dataset\": {\n"
           << "    \"directory\": ";
    write_json_string(
        output,
        portable_report_path(setup.dataset_directory).generic_string());
    output << ",\n"
           << "    \"representatives_file\": \"representatives.npy\",\n"
           << "    \"representative_labels_file\": ";
    if (std::filesystem::is_regular_file(
            setup.dataset_directory / "representative_labels.npy")) {
        output << "\"representative_labels.npy\"";
    } else {
        output << "null";
    }
    output << ",\n"
           << "    \"queries_file\": \"queries.npy\",\n"
           << "    \"query_labels_file\": \"query_labels.npy\",\n"
           << "    \"representative_count\": "
           << dataset.representatives.rows() << ",\n"
           << "    \"representative_class_count\": "
           << representative_class_count(dataset) << ",\n"
           << "    \"query_count_available\": " << dataset.queries.rows()
           << ",\n"
           << "    \"dimension\": " << dataset.representatives.cols()
           << ",\n"
           << "    \"load_ms\": " << load_ms << "\n"
           << "  },\n"
           << "  \"settings\": {\n"
           << "    \"distance\": ";
    write_json_string(
        output,
        ultrahigh_ann::benchmark::distance_name(setup.distance));
    output << ",\n"
           << "    \"queries_run\": " << query_count << ",\n"
           << "    \"query_limit\": ";
    if (setup.maximum_queries == 0) {
        output << "null";
    } else {
        output << setup.maximum_queries;
    }
    output << "\n  },\n"
           << "  \"shared_preprocessing\": {\n"
           << "    \"sampling_probabilities\": {\n"
           << "      \"computed_once\": true,\n"
           << "      \"build_ms\": "
           << probability_execution.build_ms << ",\n"
           << "      \"coordinate_count\": "
           << probability_execution.probabilities.size() << ",\n"
           << "      \"sampling_mass\": "
           << probability_execution.sampling_mass << ",\n"
           << "      \"sampling_mass_per_representative\": "
           << probability_execution.sampling_mass /
                  static_cast<double>(dataset.representatives.rows())
           << ",\n"
           << "      \"mass_matched_uniform_probability\": "
           << probability_execution.sampling_mass /
                  static_cast<double>(dataset.representatives.cols())
           << ",\n"
           << "      \"payload_bytes\": "
           << probability_execution.probabilities.size() * sizeof(double)
           << "\n"
           << "    }\n"
           << "  },\n"
           << "  \"diagnostics\": {\n"
           << "    \"included_in_query_timings\": false,\n"
           << "    \"distance_table_build_ms\": "
           << diagnostic_build_ms << ",\n"
           << "    \"run_evaluation_total_ms\": "
           << diagnostic_evaluation_ms << ",\n"
           << "    \"distance_table_payload_bytes\": "
           << distance_table.payload_bytes() << ",\n"
           << "    \"query_geometry\": ";
    write_query_geometry_json(output, distance_table.query_geometry(), 4);
    output << "\n  },\n"
           << "  \"exact\": ";
    write_indented_result_json(
        output,
        2,
        exact_build_ms,
        exact,
        exact_space_usage,
        dataset.representatives.cols(),
        false);
    output << ",\n  \"runs\": [\n";

    for (std::size_t index = 0; index < executions.size(); ++index) {
        const ApproximateExecution& execution = executions[index];
        output << "    {\n      \"name\": ";
        write_json_string(output, execution.run.name);
        output << ",\n      \"method\": ";
        write_json_string(
            output,
            ultrahigh_ann::benchmark::method_name(execution.run.method));
        output << ",\n      \"settings\": {\n"
               << "        \"repetitions\": "
               << execution.run.repetitions << ",\n"
               << "        \"seed\": " << execution.run.seed;
        if (execution.run.method ==
            ultrahigh_ann::benchmark::ApproximateMethod::hierarchical) {
            output << ",\n        \"projection_dimension\": "
                   << execution.run.projection_dimension;
        }
        output << "\n      },\n      \"result\": ";
        write_indented_result_json(
            output,
            6,
            execution.build_ms,
            execution.query_result,
            execution.space_usage,
            dataset.representatives.cols(),
            true);
        output << "\n    }"
               << (index + 1 == executions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed to write JSON report: " + setup.output_path.string());
    }
}

void print_batch_result(
    std::string_view name,
    std::string_view method,
    double build_ms,
    const QueryResult& result,
    bool has_exact_baseline)
{
    const double query_count =
        static_cast<double>(result.predictions.size());
    std::cout << std::left << std::setw(25) << name
              << std::setw(14) << method << std::right
              << std::setw(12) << std::fixed << std::setprecision(3)
              << build_ms << std::setw(13) << result.elapsed_ms
              << std::setw(13) << 1000.0 * result.elapsed_ms / query_count
              << std::setw(11) << std::setprecision(2)
              << 100.0 * accuracy(result) << '%';
    if (has_exact_baseline) {
        std::cout << std::setw(12)
                  << 100.0 * agreement_with_exact(result) << '%';
    } else {
        std::cout << std::setw(13) << '-';
    }
    std::cout << '\n';
}

void print_batch_space_usage(
    std::string_view name,
    std::string_view method,
    const ultrahigh_ann::IndexSpaceUsage& usage,
    std::size_t dimension)
{
    constexpr double bytes_per_kib = 1024.0;
    const double coordinate_percent =
        100.0 * static_cast<double>(usage.unique_query_coordinates) /
        static_cast<double>(dimension);
    std::cout << std::left << std::setw(25) << name
              << std::setw(14) << method << std::right
              << std::setw(14) << std::fixed << std::setprecision(2)
              << static_cast<double>(usage.index_payload_bytes) /
                     bytes_per_kib
              << std::setw(16)
              << static_cast<double>(usage.query_workspace_payload_bytes) /
                     bytes_per_kib
              << std::setw(14) << usage.unique_query_coordinates
              << std::setw(12) << coordinate_percent << '%'
              << std::setw(16) << usage.sampled_multiplicity << '\n';
}

int run_batch_setup(const std::filesystem::path& setup_path)
{
    const ultrahigh_ann::benchmark::BenchmarkSetup setup =
        ultrahigh_ann::benchmark::load_benchmark_setup(setup_path);
    const auto load_start = Clock::now();
    ultrahigh_ann::RepresentativeQueryDataset dataset =
        ultrahigh_ann::load_representative_query_dataset(
            setup.dataset_directory);
    const double load_ms = elapsed_ms(load_start);
    const std::size_t query_count =
        setup.maximum_queries == 0
            ? dataset.queries.rows()
            : std::min(setup.maximum_queries, dataset.queries.rows());

    std::cout << "Loaded " << dataset.representatives.rows() << " x "
              << dataset.representatives.cols() << " representatives and "
              << dataset.queries.rows() << " x " << dataset.queries.cols()
              << " queries in " << std::fixed << std::setprecision(3)
              << load_ms << " ms.\nRunning exact once and "
              << setup.runs.size() << " approximate configurations over "
              << query_count << " queries using "
              << ultrahigh_ann::benchmark::distance_name(setup.distance)
              << ".\n\n";

    const ExactExecution exact = execute_exact(
        setup.distance,
        dataset,
        query_count);
    const ProbabilityExecution probability_execution =
        compute_probabilities(
            setup.distance,
            dataset.representatives);
    std::cout << "Computed "
              << probability_execution.probabilities.size()
              << " shared sampling probabilities with S(C)="
              << probability_execution.sampling_mass << " once in "
              << std::fixed << std::setprecision(3)
              << probability_execution.build_ms << " ms.\n\n";

    std::vector<ApproximateExecution> executions;
    executions.reserve(setup.runs.size());
    for (std::size_t index = 0; index < setup.runs.size(); ++index) {
        const auto& run = setup.runs[index];
        std::cout << '[' << index + 1 << '/' << setup.runs.size() << "] "
                  << run.name << " ("
                  << ultrahigh_ann::benchmark::method_name(run.method)
                  << ")\n";
        executions.push_back(execute_approximate_run(
            setup.distance,
            run,
            dataset,
            probability_execution.probabilities,
            probability_execution.sampling_mass,
            query_count,
            exact.query_result.predictions));
    }

    const auto diagnostic_build_start = Clock::now();
    const ultrahigh_ann::benchmark::ExactDistanceTable distance_table(
        setup.distance,
        dataset.representatives,
        dataset.queries,
        query_count);
    const double diagnostic_build_ms = elapsed_ms(diagnostic_build_start);
    const auto diagnostic_evaluation_start = Clock::now();
    for (ApproximateExecution& execution : executions) {
        add_approximation_metrics(execution.query_result, distance_table);
    }
    const double diagnostic_evaluation_ms =
        elapsed_ms(diagnostic_evaluation_start);
    std::cout << "Computed reusable exact distance diagnostics in "
              << diagnostic_build_ms << " ms and evaluated all runs in "
              << diagnostic_evaluation_ms
              << " ms (excluded from query timings).\n";

    std::cout << "\n"
              << std::left << std::setw(25) << "run"
              << std::setw(14) << "method" << std::right
              << std::setw(12) << "index ms" << std::setw(13) << "query ms"
              << std::setw(13) << "us/query" << std::setw(12) << "accuracy"
              << std::setw(13) << "vs exact" << '\n';
    print_batch_result(
        "exact",
        "exact",
        exact.build_ms,
        exact.query_result,
        false);
    for (const ApproximateExecution& execution : executions) {
        print_batch_result(
            execution.run.name,
            ultrahigh_ann::benchmark::method_name(execution.run.method),
            execution.build_ms,
            execution.query_result,
            true);
    }

    std::cout << "\nLogical standalone index space "
                 "(allocator metadata excluded):\n"
              << std::left << std::setw(25) << "run"
              << std::setw(14) << "method" << std::right
              << std::setw(14) << "index KiB"
              << std::setw(16) << "workspace KiB"
              << std::setw(14) << "coordinates"
              << std::setw(13) << "dimension"
              << std::setw(16) << "multiplicity" << '\n';
    print_batch_space_usage(
        "exact",
        "exact",
        exact.space_usage,
        dataset.representatives.cols());
    for (const ApproximateExecution& execution : executions) {
        print_batch_space_usage(
            execution.run.name,
            ultrahigh_ann::benchmark::method_name(execution.run.method),
            execution.space_usage,
            dataset.representatives.cols());
    }

    write_batch_json_report(
        setup,
        dataset,
        query_count,
        load_ms,
        exact.build_ms,
        exact.query_result,
        exact.space_usage,
        diagnostic_build_ms,
        diagnostic_evaluation_ms,
        distance_table,
        probability_execution,
        executions);
    std::cout << "\nWrote JSON report to " << setup.output_path << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = parse_options(argc, argv);
        if (options.setup_path.has_value()) {
            return run_batch_setup(*options.setup_path);
        }

        const auto load_start = Clock::now();
        ultrahigh_ann::RepresentativeQueryDataset dataset =
            ultrahigh_ann::load_representative_query_dataset(
                options.dataset_directory);
        const double load_ms = elapsed_ms(load_start);

        const std::size_t query_count =
            options.maximum_queries == 0
                ? dataset.queries.rows()
                : std::min(
                      options.maximum_queries,
                      dataset.queries.rows());

        std::cout
            << "Loaded " << dataset.representatives.rows() << " x "
            << dataset.representatives.cols() << " representatives and "
            << dataset.queries.rows() << " x " << dataset.queries.cols()
            << " queries in " << std::fixed << std::setprecision(3)
            << load_ms << " ms.\n"
            << "Running " << query_count << " queries with repetitions="
            << options.repetitions << ", projection_dimension="
            << options.projection_dimension << ", seed=" << options.seed
            << ", distance="
            << ultrahigh_ann::benchmark::distance_name(options.distance)
            << ".\n\n";

        const ExactExecution exact = execute_exact(
            options.distance,
            dataset,
            query_count);
        const ProbabilityExecution probability_execution =
            compute_probabilities(
                options.distance,
                dataset.representatives);
        std::cout << "Computed "
                  << probability_execution.probabilities.size()
                  << " shared sampling probabilities with S(C)="
                  << probability_execution.sampling_mass << " once in "
                  << std::fixed << std::setprecision(3)
                  << probability_execution.build_ms << " ms.\n\n";
        const ultrahigh_ann::benchmark::ApproximateRun flat_run{
            .name = "flat",
            .method = ultrahigh_ann::benchmark::ApproximateMethod::flat,
            .repetitions = options.repetitions,
            .seed = options.seed,
        };
        ApproximateExecution flat = execute_approximate_run(
            options.distance,
            flat_run,
            dataset,
            probability_execution.probabilities,
            probability_execution.sampling_mass,
            query_count,
            exact.query_result.predictions);
        const ultrahigh_ann::benchmark::ApproximateRun hierarchical_run{
            .name = "hierarchical",
            .method =
                ultrahigh_ann::benchmark::ApproximateMethod::hierarchical,
            .repetitions = options.repetitions,
            .seed = options.seed,
            .projection_dimension = options.projection_dimension,
        };
        ApproximateExecution hierarchical = execute_approximate_run(
            options.distance,
            hierarchical_run,
            dataset,
            probability_execution.probabilities,
            probability_execution.sampling_mass,
            query_count,
            exact.query_result.predictions);

        const auto diagnostic_build_start = Clock::now();
        const ultrahigh_ann::benchmark::ExactDistanceTable distance_table(
            options.distance,
            dataset.representatives,
            dataset.queries,
            query_count);
        const double diagnostic_build_ms = elapsed_ms(diagnostic_build_start);
        const auto diagnostic_evaluation_start = Clock::now();
        add_approximation_metrics(flat.query_result, distance_table);
        add_approximation_metrics(hierarchical.query_result, distance_table);
        const double diagnostic_evaluation_ms =
            elapsed_ms(diagnostic_evaluation_start);
        std::cout << "Computed reusable exact distance diagnostics in "
                  << diagnostic_build_ms
                  << " ms and evaluated both approximate runs in "
                  << diagnostic_evaluation_ms
                  << " ms (excluded from query timings).\n\n";

        std::cout
            << std::left << std::setw(14) << "method" << std::right
            << std::setw(12) << "index ms" << std::setw(13) << "query ms"
            << std::setw(13) << "us/query" << std::setw(12) << "accuracy"
            << std::setw(13) << "vs exact" << '\n';
        print_result("exact", exact.build_ms, exact.query_result, false);
        print_result("flat", flat.build_ms, flat.query_result, true);
        print_result(
            "hierarchical",
            hierarchical.build_ms,
            hierarchical.query_result,
            true);

        std::cout
            << "\nLogical standalone index space "
               "(allocator metadata excluded):\n"
            << std::left << std::setw(14) << "method" << std::right
            << std::setw(14) << "index KiB"
            << std::setw(16) << "workspace KiB"
            << std::setw(14) << "coordinates"
            << std::setw(13) << "dimension"
            << std::setw(16) << "multiplicity" << '\n';
        print_space_usage(
            "exact",
            exact.space_usage,
            dataset.representatives.cols());
        print_space_usage(
            "flat",
            flat.space_usage,
            dataset.representatives.cols());
        print_space_usage(
            "hierarchical",
            hierarchical.space_usage,
            dataset.representatives.cols());

        write_json_report(
            options,
            dataset,
            query_count,
            load_ms,
            exact.build_ms,
            exact.query_result,
            exact.space_usage,
            diagnostic_build_ms,
            diagnostic_evaluation_ms,
            distance_table,
            probability_execution,
            flat.build_ms,
            flat.query_result,
            flat.space_usage,
            hierarchical.build_ms,
            hierarchical.query_result,
            hierarchical.space_usage);
        std::cout << "\nWrote JSON report to "
                  << options.output_path << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
