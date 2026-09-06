#include "benchmark_setup.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ultrahigh_ann::benchmark {
namespace {

constexpr std::string_view format_header{"ultrahigh_ann_benchmark_setup_v2"};

[[nodiscard]] std::string_view trim(std::string_view value)
{
    constexpr std::string_view whitespace{" \t\r\n"};
    const std::size_t first = value.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    const std::size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

[[nodiscard]] std::vector<std::string_view> split(std::string_view value,
                                                  char separator)
{
    std::vector<std::string_view> fields;
    std::size_t start{};
    while (true) {
        const std::size_t end = value.find(separator, start);
        fields.push_back(trim(value.substr(start, end - start)));
        if (end == std::string_view::npos) {
            return fields;
        }
        start = end + 1;
    }
}

[[nodiscard]] std::runtime_error parse_error(const std::filesystem::path& path,
                                             std::size_t line,
                                             std::string_view message)
{
    return std::runtime_error(path.string() + ":" + std::to_string(line) +
                              ": " + std::string(message));
}

template <class Integer>
[[nodiscard]] Integer
parse_integer(std::string_view text, const std::filesystem::path& path,
              std::size_t line, std::string_view field, bool allow_zero)
{
    Integer value{};
    const auto [position, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    bool invalid = error != std::errc{} ||
                   position != text.data() + text.size() ||
                   (!allow_zero && value == 0);
    if constexpr (std::is_signed_v<Integer>) {
        invalid = invalid || value < 0;
    }
    if (invalid) {
        throw parse_error(path, line,
                          std::string(field) +
                              (allow_zero ? " must be a nonnegative integer"
                                          : " must be a positive integer"));
    }
    return value;
}

[[nodiscard]] std::filesystem::path
resolve_path(const std::filesystem::path& setup_path,
             std::string_view configured)
{
    const std::filesystem::path value{configured};
    if (value.is_absolute()) {
        return value.lexically_normal();
    }
    return (setup_path.parent_path() / value).lexically_normal();
}

[[nodiscard]] bool valid_run_name(std::string_view value)
{
    return !value.empty() && std::ranges::all_of(value, [](char character) {
        const auto byte = static_cast<unsigned char>(character);
        return (byte >= static_cast<unsigned char>('a') &&
                byte <= static_cast<unsigned char>('z')) ||
               (byte >= static_cast<unsigned char>('A') &&
                byte <= static_cast<unsigned char>('Z')) ||
               (byte >= static_cast<unsigned char>('0') &&
                byte <= static_cast<unsigned char>('9')) ||
               character == '_' || character == '-' || character == '.';
    });
}

[[nodiscard]] std::pair<std::string_view, std::string_view>
parse_parameter(std::string_view field, const std::filesystem::path& path,
                std::size_t line)
{
    const std::size_t separator = field.find('=');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 == field.size()) {
        throw parse_error(path, line,
                          "run parameters must use key=value syntax");
    }
    return {trim(field.substr(0, separator)),
            trim(field.substr(separator + 1))};
}

[[nodiscard]] DistanceMetric parse_distance(std::string_view value,
                                            const std::filesystem::path& path,
                                            std::size_t line)
{
    if (value == "l1") {
        return DistanceMetric::l1;
    }
    if (value == "l2") {
        return DistanceMetric::l2;
    }
    throw parse_error(path, line, "distance must be l1 or l2");
}

[[nodiscard]] IndexKind parse_index(std::string_view value,
                                    const std::filesystem::path& path,
                                    std::size_t line)
{
    if (value == "exact") {
        return IndexKind::exact;
    }
    if (value == "flat") {
        return IndexKind::flat;
    }
    if (value == "uniform") {
        return IndexKind::uniform;
    }
    if (value == "hierarchical") {
        return IndexKind::hierarchical;
    }
    throw parse_error(path, line,
                      "index must be exact, flat, uniform, or hierarchical");
}

[[nodiscard]] ExecutionBackend parse_backend(std::string_view value,
                                             const std::filesystem::path& path,
                                             std::size_t line)
{
    if (value == "cpu") {
        return ExecutionBackend::cpu;
    }
    if (value == "cuda") {
        return ExecutionBackend::cuda;
    }
    throw parse_error(path, line, "backend must be cpu or cuda");
}

[[nodiscard]] QueryStrategy parse_strategy(std::string_view value,
                                           const std::filesystem::path& path,
                                           std::size_t line)
{
    if (value == "sequential") {
        return QueryStrategy::sequential;
    }
    if (value == "parallel_queries") {
        return QueryStrategy::parallel_queries;
    }
    if (value == "parallel_representatives") {
        return QueryStrategy::parallel_representatives;
    }
    if (value == "automatic") {
        return QueryStrategy::automatic;
    }
    if (value == "direct") {
        return QueryStrategy::direct;
    }
    if (value == "gemm") {
        return QueryStrategy::gemm;
    }
    throw parse_error(path, line, "unknown query strategy");
}

[[nodiscard]] ProbabilityPolicy
parse_probability_policy(std::string_view value,
                         const std::filesystem::path& path, std::size_t line)
{
    if (value == "sequential") {
        return ProbabilityPolicy::sequential;
    }
    if (value == "cpu_parallel") {
        return ProbabilityPolicy::cpu_parallel;
    }
    if (value == "gpu_fp32") {
        return ProbabilityPolicy::gpu_fp32;
    }
    if (value == "gpu_cublas_fp32") {
        return ProbabilityPolicy::gpu_cublas_fp32;
    }
    if (value == "load") {
        return ProbabilityPolicy::load;
    }
    throw parse_error(path, line, "unknown probability policy");
}

[[nodiscard]] DiagnosticMode
parse_diagnostics(std::string_view value, const std::filesystem::path& path,
                  std::size_t line)
{
    if (value == "none") {
        return DiagnosticMode::none;
    }
    if (value == "selected_distances") {
        return DiagnosticMode::selected_distances;
    }
    if (value == "full_distance_table") {
        return DiagnosticMode::full_distance_table;
    }
    throw parse_error(
        path, line,
        "diagnostics must be none, selected_distances, or full_distance_table");
}

[[nodiscard]] std::vector<std::size_t>
parse_batch_sizes(std::string_view value, const std::filesystem::path& path,
                  std::size_t line)
{
    std::vector<std::size_t> result;
    std::unordered_set<std::size_t> seen;
    for (const std::string_view field : split(value, ',')) {
        const std::size_t batch =
            parse_integer<std::size_t>(field, path, line, "batch_sizes", false);
        if (!seen.insert(batch).second) {
            throw parse_error(path, line,
                              "batch_sizes must not contain duplicates");
        }
        result.push_back(batch);
    }
    return result;
}

[[nodiscard]] BenchmarkRun
parse_run(const std::vector<std::string_view>& fields,
          const BenchmarkSetup& setup, const std::filesystem::path& path,
          std::size_t line)
{
    if (fields.size() < 3) {
        throw parse_error(path, line, "run requires a name and index kind");
    }
    if (!valid_run_name(fields[1])) {
        throw parse_error(
            path, line,
            "run name may contain only letters, digits, '.', '_', and '-'");
    }

    BenchmarkRun run{
        .name = std::string(fields[1]),
        .index = parse_index(fields[2], path, line),
        .batch_sizes = setup.batch_sizes,
        .warmups = setup.warmups,
        .trials = setup.trials,
    };
    std::unordered_map<std::string, std::string_view> parameters;
    for (std::size_t index = 3; index < fields.size(); ++index) {
        const auto [key, value] = parse_parameter(fields[index], path, line);
        if (!parameters.emplace(std::string(key), value).second) {
            throw parse_error(path, line,
                              "duplicate run parameter: " + std::string(key));
        }
    }

    if (const auto found = parameters.find("backend");
        found != parameters.end()) {
        run.backend = parse_backend(found->second, path, line);
    }
    run.strategy =
        run.backend == ExecutionBackend::cuda
            ? (setup.distance == DistanceMetric::l1 ? QueryStrategy::direct
                                                    : QueryStrategy::gemm)
            : QueryStrategy::sequential;
    if (const auto found = parameters.find("strategy");
        found != parameters.end()) {
        run.strategy = parse_strategy(found->second, path, line);
    }
    if (const auto found = parameters.find("batch_sizes");
        found != parameters.end()) {
        run.batch_sizes = parse_batch_sizes(found->second, path, line);
    }
    if (const auto found = parameters.find("warmups");
        found != parameters.end()) {
        run.warmups = parse_integer<std::size_t>(found->second, path, line,
                                                 "warmups", true);
    }
    if (const auto found = parameters.find("trials");
        found != parameters.end()) {
        run.trials = parse_integer<std::size_t>(found->second, path, line,
                                                "trials", false);
    }

    const bool approximate = run.index != IndexKind::exact;
    const auto repetitions = parameters.find("repetitions");
    const auto seed = parameters.find("seed");
    if (approximate &&
        (repetitions == parameters.end() || seed == parameters.end())) {
        throw parse_error(path, line,
                          "approximate runs require repetitions and seed");
    }
    if (!approximate &&
        (repetitions != parameters.end() || seed != parameters.end())) {
        throw parse_error(path, line,
                          "exact runs do not accept repetitions or seed");
    }
    if (approximate) {
        run.repetitions = parse_integer<std::size_t>(
            repetitions->second, path, line, "repetitions", false);
        run.seed = parse_integer<std::uint64_t>(seed->second, path, line,
                                                "seed", true);
    }
    const auto projection = parameters.find("projection_dimension");
    if (run.index == IndexKind::hierarchical) {
        if (projection == parameters.end()) {
            throw parse_error(path, line,
                              "hierarchical runs require projection_dimension");
        }
        run.projection_dimension = parse_integer<std::size_t>(
            projection->second, path, line, "projection_dimension", false);
    } else if (projection != parameters.end()) {
        throw parse_error(path, line,
                          "projection_dimension requires a hierarchical run");
    }

    const std::unordered_set<std::string> allowed{
        "backend", "strategy",    "batch_sizes", "warmups",
        "trials",  "repetitions", "seed",        "projection_dimension"};
    for (const auto& [key, value] : parameters) {
        static_cast<void>(value);
        if (!allowed.contains(key)) {
            throw parse_error(path, line, "unsupported run parameter: " + key);
        }
    }

    if (run.backend == ExecutionBackend::cuda) {
        if (run.strategy != QueryStrategy::direct &&
            run.strategy != QueryStrategy::gemm) {
            throw parse_error(path, line,
                              "CUDA strategy must be direct or gemm");
        }
        if (setup.distance == DistanceMetric::l1 &&
            run.index != IndexKind::hierarchical &&
            run.strategy != QueryStrategy::direct) {
            throw parse_error(path, line,
                              "CUDA L1 exact, flat, and uniform strategies "
                              "must be direct");
        }
    } else {
        if (run.strategy == QueryStrategy::direct ||
            run.strategy == QueryStrategy::gemm) {
            throw parse_error(path, line,
                              "CPU runs cannot use direct or gemm strategy");
        }
        if ((setup.distance == DistanceMetric::l1 ||
             run.index == IndexKind::hierarchical) &&
            run.strategy != QueryStrategy::sequential) {
            throw parse_error(
                path, line,
                "CPU L1 and hierarchical runs require sequential strategy");
        }
    }
    return run;
}

}  // namespace

std::string_view distance_name(DistanceMetric value) noexcept
{
    return value == DistanceMetric::l1 ? "l1" : "l2";
}

std::string_view index_name(IndexKind value) noexcept
{
    switch (value) {
    case IndexKind::exact:
        return "exact";
    case IndexKind::flat:
        return "flat";
    case IndexKind::uniform:
        return "uniform";
    case IndexKind::hierarchical:
        return "hierarchical";
    }
    return "unknown";
}

std::string_view backend_name(ExecutionBackend value) noexcept
{
    return value == ExecutionBackend::cpu ? "cpu" : "cuda";
}

std::string_view strategy_name(QueryStrategy value) noexcept
{
    switch (value) {
    case QueryStrategy::sequential:
        return "sequential";
    case QueryStrategy::parallel_queries:
        return "parallel_queries";
    case QueryStrategy::parallel_representatives:
        return "parallel_representatives";
    case QueryStrategy::automatic:
        return "automatic";
    case QueryStrategy::direct:
        return "direct";
    case QueryStrategy::gemm:
        return "gemm";
    }
    return "unknown";
}

std::string_view probability_policy_name(ProbabilityPolicy value) noexcept
{
    switch (value) {
    case ProbabilityPolicy::sequential:
        return "sequential";
    case ProbabilityPolicy::cpu_parallel:
        return "cpu_parallel";
    case ProbabilityPolicy::gpu_fp32:
        return "gpu_fp32";
    case ProbabilityPolicy::gpu_cublas_fp32:
        return "gpu_cublas_fp32";
    case ProbabilityPolicy::load:
        return "load";
    }
    return "unknown";
}

std::string_view diagnostic_mode_name(DiagnosticMode value) noexcept
{
    switch (value) {
    case DiagnosticMode::none:
        return "none";
    case DiagnosticMode::selected_distances:
        return "selected_distances";
    case DiagnosticMode::full_distance_table:
        return "full_distance_table";
    }
    return "unknown";
}

BenchmarkSetup load_benchmark_setup(const std::filesystem::path& input_path)
{
    const std::filesystem::path path =
        std::filesystem::absolute(input_path).lexically_normal();
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open benchmark setup: " +
                                 path.string());
    }

    BenchmarkSetup setup;
    setup.setup_path = path;
    bool header_seen{};
    std::unordered_map<std::string, std::pair<std::string, std::size_t>>
        directives;
    std::vector<std::pair<std::vector<std::string>, std::size_t>> raw_runs;
    std::string text;
    std::size_t line_number{};
    while (std::getline(input, text)) {
        ++line_number;
        const std::string_view line = trim(text);
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        if (!header_seen) {
            if (line != format_header) {
                throw parse_error(path, line_number,
                                  "expected format header " +
                                      std::string(format_header));
            }
            header_seen = true;
            continue;
        }
        const auto fields = split(line, '\t');
        if (fields[0] == "run") {
            std::vector<std::string> owned;
            owned.reserve(fields.size());
            for (const auto field : fields) {
                owned.emplace_back(field);
            }
            raw_runs.emplace_back(std::move(owned), line_number);
            continue;
        }
        if (fields.size() != 2 || fields[1].empty()) {
            throw parse_error(path, line_number,
                              std::string(fields[0]) +
                                  " requires exactly one value");
        }
        if (!directives
                 .emplace(std::string(fields[0]),
                          std::pair{std::string(fields[1]), line_number})
                 .second) {
            throw parse_error(path, line_number,
                              "duplicate directive: " + std::string(fields[0]));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed to read benchmark setup: " +
                                 path.string());
    }
    if (!header_seen) {
        throw std::runtime_error("benchmark setup is empty: " + path.string());
    }

    auto require = [&](std::string_view key) -> const std::string& {
        const auto found = directives.find(std::string(key));
        if (found == directives.end()) {
            throw std::runtime_error("benchmark setup does not define " +
                                     std::string(key));
        }
        return found->second.first;
    };
    setup.dataset_directory = resolve_path(path, require("dataset"));
    setup.json_output_path = resolve_path(path, require("json_output"));
    setup.csv_output_path = resolve_path(path, require("csv_output"));
    setup.reference_run = require("reference");
    if (setup.json_output_path == setup.csv_output_path) {
        throw std::runtime_error("json_output and csv_output must differ");
    }

    auto value = [&](std::string_view key) -> std::optional<std::string_view> {
        const auto found = directives.find(std::string(key));
        if (found == directives.end()) {
            return std::nullopt;
        }
        return found->second.first;
    };
    auto line_of = [&](std::string_view key) {
        return directives.at(std::string(key)).second;
    };

    if (const auto configured = value("distance")) {
        setup.distance = parse_distance(*configured, path, line_of("distance"));
    }
    if (const auto configured = value("max_queries")) {
        setup.maximum_queries = parse_integer<std::size_t>(
            *configured, path, line_of("max_queries"), "max_queries", true);
    }
    if (const auto configured = value("warmups")) {
        setup.warmups = parse_integer<std::size_t>(
            *configured, path, line_of("warmups"), "warmups", true);
    }
    if (const auto configured = value("trials")) {
        setup.trials = parse_integer<std::size_t>(
            *configured, path, line_of("trials"), "trials", false);
    }
    if (const auto configured = value("batch_sizes")) {
        setup.batch_sizes =
            parse_batch_sizes(*configured, path, line_of("batch_sizes"));
    }
    if (const auto configured = value("device")) {
        setup.device = parse_integer<int>(*configured, path, line_of("device"),
                                          "device", true);
    }
    if (const auto configured = value("diagnostics")) {
        setup.diagnostics =
            parse_diagnostics(*configured, path, line_of("diagnostics"));
    }
    if (const auto configured = value("probability_policy")) {
        setup.probability_policy = parse_probability_policy(
            *configured, path, line_of("probability_policy"));
    }
    if (const auto configured = value("probabilities")) {
        setup.probabilities_path = resolve_path(path, *configured);
        if (!value("probability_policy")) {
            setup.probability_policy = ProbabilityPolicy::load;
        }
    }
    if (setup.probability_policy == ProbabilityPolicy::load &&
        !setup.probabilities_path.has_value()) {
        throw std::runtime_error(
            "probability_policy=load requires probabilities");
    }
    if (setup.distance == DistanceMetric::l1 &&
        setup.probability_policy == ProbabilityPolicy::gpu_cublas_fp32) {
        throw parse_error(
            path, line_of("probability_policy"),
            "gpu_cublas_fp32 is unavailable for L1 probability computation");
    }

    const auto dataset_file = [&](std::string_view directive,
                                  std::string_view default_name) {
        const std::string_view filename =
            value(directive).value_or(default_name);
        const std::filesystem::path configured{filename};
        return configured.is_absolute()
                   ? configured.lexically_normal()
                   : (setup.dataset_directory / configured).lexically_normal();
    };
    setup.representatives_path =
        dataset_file("representatives_file", "representatives.npy");
    setup.representative_labels_path =
        dataset_file("representative_labels_file", "representative_labels.npy");
    setup.queries_path = dataset_file("queries_file", "queries.npy");
    setup.query_labels_path =
        dataset_file("query_labels_file", "query_labels.npy");

    const std::unordered_set<std::string> known_directives{
        "dataset",
        "json_output",
        "csv_output",
        "reference",
        "distance",
        "max_queries",
        "warmups",
        "trials",
        "batch_sizes",
        "device",
        "diagnostics",
        "probability_policy",
        "probabilities",
        "representatives_file",
        "representative_labels_file",
        "queries_file",
        "query_labels_file"};
    for (const auto& [key, configured] : directives) {
        static_cast<void>(configured);
        if (!known_directives.contains(key)) {
            throw parse_error(path, directives.at(key).second,
                              "unknown directive: " + key);
        }
    }

    std::unordered_set<std::string> run_names;
    for (const auto& [owned_fields, run_line] : raw_runs) {
        std::vector<std::string_view> fields;
        fields.reserve(owned_fields.size());
        for (const auto& field : owned_fields) {
            fields.push_back(field);
        }
        BenchmarkRun run = parse_run(fields, setup, path, run_line);
        if (!run_names.insert(run.name).second) {
            throw parse_error(path, run_line,
                              "duplicate run name: " + run.name);
        }
        setup.runs.push_back(std::move(run));
    }
    if (setup.runs.empty()) {
        throw std::runtime_error("benchmark setup does not define any runs");
    }
    const auto reference =
        std::ranges::find_if(setup.runs, [&](const BenchmarkRun& run) {
            return run.name == setup.reference_run;
        });
    if (reference == setup.runs.end()) {
        throw std::runtime_error("reference does not name a configured run");
    }
    if (reference->index != IndexKind::exact) {
        throw std::runtime_error("reference run must use the exact index");
    }
    return setup;
}

}  // namespace ultrahigh_ann::benchmark
