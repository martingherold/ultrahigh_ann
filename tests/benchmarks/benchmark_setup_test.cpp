#include "benchmark_setup.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto suffix = std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count();
        path_ = std::filesystem::temp_directory_path() /
                ("ultrahigh-ann-setup-test-" + std::to_string(suffix));
        std::filesystem::create_directories(path_);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void expect(bool condition, std::string_view message)
{
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void write_text(const std::filesystem::path& path, std::string_view text)
{
    std::ofstream output(path);
    output << text;
    if (!output) {
        throw std::runtime_error("failed to write test setup");
    }
}

template<class Function>
void expect_failure(Function&& function, std::string_view expected_message)
{
    try {
        function();
    } catch (const std::runtime_error& error) {
        expect(
            std::string_view(error.what()).find(expected_message) !=
                std::string_view::npos,
            "unexpected setup-parser error");
        return;
    }
    throw std::runtime_error("expected setup parsing to fail");
}

void test_loads_flat_uniform_and_hierarchical_runs()
{
    const TemporaryDirectory temporary;
    const std::filesystem::path setup_path = temporary.path() / "suite.tsv";
    write_text(
        setup_path,
        "# benchmark suite\n"
        "ultrahigh_ann_benchmark_setup_v1\n"
        "dataset\t../dataset\n"
        "output\tresults/report.json\n"
        "distance\tl2\n"
        "max_queries\t123\n"
        "run\tflat_t128_seed42\tflat\trepetitions=128\tseed=42\n"
        "run\tuniform_t128_seed42\tuniform\trepetitions=128\tseed=42\n"
        "run\thier_t256_p31\thierarchical\trepetitions=256\tseed=7\t"
        "projection_dimension=31\n");

    const auto setup =
        ultrahigh_ann::benchmark::load_benchmark_setup(setup_path);
    expect(setup.maximum_queries == 123, "maximum query count changed");
    expect(
        setup.distance == ultrahigh_ann::benchmark::DistanceMetric::l2,
        "distance metric changed");
    expect(setup.runs.size() == 3, "run count changed");
    expect(
        setup.dataset_directory ==
            (temporary.path() / "../dataset").lexically_normal(),
        "dataset path was not resolved relative to setup");
    expect(
        setup.output_path ==
            (temporary.path() / "results/report.json").lexically_normal(),
        "output path was not resolved relative to setup");
    expect(
        setup.runs[0].method ==
            ultrahigh_ann::benchmark::ApproximateMethod::flat,
        "flat method changed");
    expect(setup.runs[0].repetitions == 128, "flat repetitions changed");
    expect(setup.runs[0].seed == 42, "flat seed changed");
    expect(
        setup.runs[1].method ==
            ultrahigh_ann::benchmark::ApproximateMethod::uniform,
        "uniform method changed");
    expect(
        setup.runs[1].repetitions == 128,
        "uniform repetitions changed");
    expect(setup.runs[1].seed == 42, "uniform seed changed");
    expect(
        setup.runs[2].method ==
            ultrahigh_ann::benchmark::ApproximateMethod::hierarchical,
        "hierarchical method changed");
    expect(
        setup.runs[2].projection_dimension == 31,
        "projection dimension changed");
}

void test_rejects_duplicate_names_and_wrong_parameters()
{
    const TemporaryDirectory temporary;
    const std::filesystem::path duplicate = temporary.path() / "duplicate.tsv";
    write_text(
        duplicate,
        "ultrahigh_ann_benchmark_setup_v1\n"
        "dataset\tdataset\n"
        "output\treport.json\n"
        "run\tsame\tflat\trepetitions=128\tseed=1\n"
        "run\tsame\tflat\trepetitions=256\tseed=2\n");
    expect_failure(
        [&duplicate] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(duplicate));
        },
        "duplicate run name");

    const std::filesystem::path wrong = temporary.path() / "wrong.tsv";
    write_text(
        wrong,
        "ultrahigh_ann_benchmark_setup_v1\n"
        "dataset\tdataset\n"
        "output\treport.json\n"
        "run\tflat\tflat\trepetitions=128\tseed=1\t"
        "projection_dimension=31\n");
    expect_failure(
        [&wrong] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(wrong));
        },
        "unsupported parameter for flat");
}

void test_defaults_to_l1_and_rejects_unknown_distance()
{
    const TemporaryDirectory temporary;
    const std::filesystem::path default_l1 = temporary.path() / "l1.tsv";
    write_text(
        default_l1,
        "ultrahigh_ann_benchmark_setup_v1\n"
        "dataset\tdataset\n"
        "output\treport.json\n"
        "run\tflat\tflat\trepetitions=1\tseed=1\n");
    const auto setup =
        ultrahigh_ann::benchmark::load_benchmark_setup(default_l1);
    expect(
        setup.distance == ultrahigh_ann::benchmark::DistanceMetric::l1,
        "omitting distance must preserve the version-one L1 default");

    const std::filesystem::path unknown = temporary.path() / "unknown.tsv";
    write_text(
        unknown,
        "ultrahigh_ann_benchmark_setup_v1\n"
        "dataset\tdataset\n"
        "output\treport.json\n"
        "distance\tcosine\n"
        "run\tflat\tflat\trepetitions=1\tseed=1\n");
    expect_failure(
        [&unknown] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(unknown));
        },
        "unknown distance metric");
}

}  // namespace

int main()
{
    test_loads_flat_uniform_and_hierarchical_runs();
    test_rejects_duplicate_names_and_wrong_parameters();
    test_defaults_to_l1_and_rejects_unknown_distance();
    return 0;
}
