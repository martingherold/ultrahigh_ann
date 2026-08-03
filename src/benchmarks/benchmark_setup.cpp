#include "benchmark_setup.hpp"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ultrahigh_ann::benchmark {
namespace {

constexpr std::string_view format_header{
    "ultrahigh_ann_benchmark_setup_v1"};

[[nodiscard]] std::string_view trim(std::string_view value)
{
    constexpr std::string_view whitespace{" \t\r\n"};
    const std::size_t start = value.find_first_not_of(whitespace);
    if (start == std::string_view::npos) {
        return {};
    }
    const std::size_t end = value.find_last_not_of(whitespace);
    return value.substr(start, end - start + 1);
}

[[nodiscard]] std::vector<std::string_view> split_tabs(
    std::string_view line)
{
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t separator = line.find('\t', start);
        fields.push_back(trim(line.substr(start, separator - start)));
        if (separator == std::string_view::npos) {
            return fields;
        }
        start = separator + 1;
    }
}

[[nodiscard]] std::runtime_error parse_error(
    const std::filesystem::path& path,
    std::size_t line_number,
    std::string_view message)
{
    return std::runtime_error(
        path.string() + ":" + std::to_string(line_number) + ": " +
        std::string(message));
}

template<class Integer>
[[nodiscard]] Integer parse_integer(
    std::string_view text,
    const std::filesystem::path& path,
    std::size_t line_number,
    std::string_view field,
    bool allow_zero)
{
    Integer value{};
    const auto [position, error] = std::from_chars(
        text.data(),
        text.data() + text.size(),
        value);
    if (error != std::errc{} || position != text.data() + text.size() ||
        (!allow_zero && value == 0)) {
        throw parse_error(
            path,
            line_number,
            std::string(field) +
                (allow_zero ? " must be a nonnegative integer"
                            : " must be a positive integer"));
    }
    return value;
}

[[nodiscard]] bool valid_run_name(std::string_view name)
{
    if (name.empty()) {
        return false;
    }
    return std::ranges::all_of(name, [](char character) {
        const auto value = static_cast<unsigned char>(character);
        return (value >= static_cast<unsigned char>('a') &&
                value <= static_cast<unsigned char>('z')) ||
               (value >= static_cast<unsigned char>('A') &&
                value <= static_cast<unsigned char>('Z')) ||
               (value >= static_cast<unsigned char>('0') &&
                value <= static_cast<unsigned char>('9')) ||
               character == '_' || character == '-' || character == '.';
    });
}

[[nodiscard]] std::pair<std::string_view, std::string_view>
parse_parameter(
    std::string_view field,
    const std::filesystem::path& path,
    std::size_t line_number)
{
    const std::size_t separator = field.find('=');
    if (separator == std::string_view::npos || separator == 0 ||
        separator + 1 == field.size()) {
        throw parse_error(
            path,
            line_number,
            "run parameters must use key=value syntax");
    }
    return {
        trim(field.substr(0, separator)),
        trim(field.substr(separator + 1)),
    };
}

[[nodiscard]] ApproximateMethod parse_method(
    std::string_view value,
    const std::filesystem::path& path,
    std::size_t line_number)
{
    if (value == "flat") {
        return ApproximateMethod::flat;
    }
    if (value == "uniform") {
        return ApproximateMethod::uniform;
    }
    if (value == "hierarchical") {
        return ApproximateMethod::hierarchical;
    }
    throw parse_error(
        path,
        line_number,
        "unknown approximate method: " + std::string(value));
}

[[nodiscard]] DistanceMetric parse_distance(
    std::string_view value,
    const std::filesystem::path& path,
    std::size_t line_number)
{
    if (value == "l1") {
        return DistanceMetric::l1;
    }
    if (value == "l2") {
        return DistanceMetric::l2;
    }
    throw parse_error(
        path,
        line_number,
        "unknown distance metric: " + std::string(value));
}

[[nodiscard]] ApproximateRun parse_run(
    const std::vector<std::string_view>& fields,
    const std::filesystem::path& path,
    std::size_t line_number)
{
    if (fields.size() < 5) {
        throw parse_error(
            path,
            line_number,
            "run requires a name, method, repetitions, and seed");
    }
    if (!valid_run_name(fields[1])) {
        throw parse_error(
            path,
            line_number,
            "run name may contain only letters, digits, '.', '_', and '-'");
    }

    const ApproximateMethod method =
        parse_method(fields[2], path, line_number);
    std::unordered_map<std::string, std::string_view> parameters;
    for (std::size_t index = 3; index < fields.size(); ++index) {
        const auto [key, value] =
            parse_parameter(fields[index], path, line_number);
        if (!parameters.emplace(std::string(key), value).second) {
            throw parse_error(
                path,
                line_number,
                "duplicate run parameter: " + std::string(key));
        }
    }

    const auto repetitions = parameters.find("repetitions");
    const auto seed = parameters.find("seed");
    if (repetitions == parameters.end() || seed == parameters.end()) {
        throw parse_error(
            path,
            line_number,
            "every run requires repetitions and seed parameters");
    }

    ApproximateRun run{
        .name = std::string(fields[1]),
        .method = method,
        .repetitions = parse_integer<std::size_t>(
            repetitions->second,
            path,
            line_number,
            "repetitions",
            false),
        .seed = parse_integer<std::uint64_t>(
            seed->second,
            path,
            line_number,
            "seed",
            true),
    };

    std::unordered_set<std::string_view> allowed{
        "repetitions",
        "seed",
    };
    if (method == ApproximateMethod::hierarchical) {
        const auto projection = parameters.find("projection_dimension");
        if (projection == parameters.end()) {
            throw parse_error(
                path,
                line_number,
                "hierarchical run requires projection_dimension");
        }
        run.projection_dimension = parse_integer<std::size_t>(
            projection->second,
            path,
            line_number,
            "projection_dimension",
            false);
        allowed.insert("projection_dimension");
    }
    for (const auto& [key, value] : parameters) {
        static_cast<void>(value);
        if (!allowed.contains(key)) {
            throw parse_error(
                path,
                line_number,
                "unsupported parameter for " +
                    std::string(method_name(method)) + ": " + key);
        }
    }
    return run;
}

[[nodiscard]] std::filesystem::path resolve_path(
    const std::filesystem::path& setup_path,
    std::string_view configured)
{
    const std::filesystem::path value{configured};
    if (value.is_absolute()) {
        return value.lexically_normal();
    }
    return (setup_path.parent_path() / value).lexically_normal();
}

}  // namespace

std::string_view method_name(ApproximateMethod method) noexcept
{
    switch (method) {
    case ApproximateMethod::flat:
        return "flat";
    case ApproximateMethod::uniform:
        return "uniform";
    case ApproximateMethod::hierarchical:
        return "hierarchical";
    }
    return "unknown";
}

std::string_view distance_name(DistanceMetric distance) noexcept
{
    switch (distance) {
    case DistanceMetric::l1:
        return "l1";
    case DistanceMetric::l2:
        return "l2";
    }
    return "unknown";
}

BenchmarkSetup load_benchmark_setup(const std::filesystem::path& input_path)
{
    const std::filesystem::path path =
        std::filesystem::absolute(input_path).lexically_normal();
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open benchmark setup: " + path.string());
    }

    BenchmarkSetup setup;
    setup.setup_path = path;
    bool header_seen = false;
    bool dataset_seen = false;
    bool output_seen = false;
    bool distance_seen = false;
    bool maximum_queries_seen = false;
    std::unordered_set<std::string> run_names;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string_view trimmed = trim(line);
        if (trimmed.empty() || trimmed.starts_with('#')) {
            continue;
        }
        if (!header_seen) {
            if (trimmed != format_header) {
                throw parse_error(
                    path,
                    line_number,
                    "expected format header " + std::string(format_header));
            }
            header_seen = true;
            continue;
        }

        const std::vector<std::string_view> fields = split_tabs(trimmed);
        if (fields[0] == "dataset" || fields[0] == "output" ||
            fields[0] == "distance" || fields[0] == "max_queries") {
            if (fields.size() != 2 || fields[1].empty()) {
                throw parse_error(
                    path,
                    line_number,
                    std::string(fields[0]) + " requires exactly one value");
            }
            if (fields[0] == "dataset") {
                if (std::exchange(dataset_seen, true)) {
                    throw parse_error(path, line_number, "duplicate dataset");
                }
                setup.dataset_directory = resolve_path(path, fields[1]);
            } else if (fields[0] == "output") {
                if (std::exchange(output_seen, true)) {
                    throw parse_error(path, line_number, "duplicate output");
                }
                setup.output_path = resolve_path(path, fields[1]);
            } else if (fields[0] == "distance") {
                if (std::exchange(distance_seen, true)) {
                    throw parse_error(path, line_number, "duplicate distance");
                }
                setup.distance = parse_distance(
                    fields[1],
                    path,
                    line_number);
            } else {
                if (std::exchange(maximum_queries_seen, true)) {
                    throw parse_error(
                        path,
                        line_number,
                        "duplicate max_queries");
                }
                setup.maximum_queries = parse_integer<std::size_t>(
                    fields[1],
                    path,
                    line_number,
                    "max_queries",
                    true);
            }
        } else if (fields[0] == "run") {
            ApproximateRun run = parse_run(fields, path, line_number);
            if (!run_names.insert(run.name).second) {
                throw parse_error(
                    path,
                    line_number,
                    "duplicate run name: " + run.name);
            }
            setup.runs.push_back(std::move(run));
        } else {
            throw parse_error(
                path,
                line_number,
                "unknown directive: " + std::string(fields[0]));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error("failed to read benchmark setup: " + path.string());
    }
    if (!header_seen) {
        throw std::runtime_error("benchmark setup is empty: " + path.string());
    }
    if (!dataset_seen) {
        throw std::runtime_error("benchmark setup does not define dataset");
    }
    if (!output_seen) {
        throw std::runtime_error("benchmark setup does not define output");
    }
    if (setup.runs.empty()) {
        throw std::runtime_error("benchmark setup does not define any runs");
    }
    return setup;
}

}  // namespace ultrahigh_ann::benchmark
