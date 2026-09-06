#include "benchmark_setup.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
        expect(std::string_view(error.what()).find(expected_message) !=
                   std::string_view::npos,
               "unexpected setup-parser error");
        return;
    }
    throw std::runtime_error("expected setup parsing to fail");
}

void test_loads_cpu_and_cuda_runs()
{
    using namespace ultrahigh_ann::benchmark;
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "suite.tsv";
    write_text(
        path,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\t../dataset\n"
        "json_output\tresults/report.json\n"
        "csv_output\tresults/trials.csv\n"
        "probabilities\tcache/l2.uap\n"
        "distance\tl2\n"
        "reference\texact_cuda\n"
        "max_queries\t123\n"
        "warmups\t1\n"
        "trials\t5\n"
        "batch_sizes\t32,128\n"
        "diagnostics\tselected_distances\n"
        "run\texact_cuda\texact\tbackend=cuda\tstrategy=gemm\n"
        "run\tflat_cpu\tflat\tbackend=cpu\tstrategy=parallel_queries\t"
        "repetitions=64\tseed=42\tbatch_sizes=123\n"
        "run\thier\thierarchical\trepetitions=128\tseed=7\t"
        "projection_dimension=31\n");

    const BenchmarkSetup setup = load_benchmark_setup(path);
    expect(setup.distance == DistanceMetric::l2, "distance changed");
    expect(setup.maximum_queries == 123, "query limit changed");
    expect(setup.reference_run == "exact_cuda", "reference changed");
    expect(setup.probability_policy == ProbabilityPolicy::load,
           "probability file should imply load policy");
    expect(setup.runs.size() == 3, "run count changed");
    expect(setup.runs[0].backend == ExecutionBackend::cuda,
           "CUDA backend changed");
    expect(setup.runs[0].strategy == QueryStrategy::gemm,
           "GEMM strategy changed");
    expect(setup.runs[0].batch_sizes.size() == 2,
           "global batch sizes not inherited");
    expect(setup.runs[1].strategy == QueryStrategy::parallel_queries,
           "CPU strategy changed");
    expect(setup.runs[1].batch_sizes == std::vector<std::size_t>{123},
           "run batch override changed");
    expect(setup.runs[2].projection_dimension == 31,
           "projection dimension changed");
    expect(setup.json_output_path ==
               (temporary.path() / "results/report.json").lexically_normal(),
           "JSON output path resolution changed");
    expect(setup.representatives_path ==
               (temporary.path() / "../dataset/reference_vectors.npy")
                   .lexically_normal(),
           "dataset file path resolution changed");
}

void test_reference_vector_configuration()
{
    using namespace ultrahigh_ann::benchmark;
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "reference-vectors.tsv";
    const std::string base =
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "reference\texact\n";
    write_text(
        path,
        base +
            "reference_vectors_file\tinputs/vectors.npy\n"
            "reference_labels_file\tinputs/labels.npy\n"
            "run\texact\texact\tstrategy=parallel_reference_vectors\n");
    const BenchmarkSetup setup = load_benchmark_setup(path);
    expect(setup.representatives_path ==
               temporary.path() / "dataset/inputs/vectors.npy",
           "reference vector override must resolve from the dataset directory");
    expect(setup.representative_labels_path ==
               temporary.path() / "dataset/inputs/labels.npy",
           "reference label override must resolve from the dataset directory");
    expect(setup.runs[0].strategy == QueryStrategy::parallel_representatives,
           "reference-vector strategy must select parallel reference scanning");
    expect(strategy_name(setup.runs[0].strategy) == "parallel_reference_vectors",
           "the reported strategy must use the canonical CLI spelling");

    for (const std::string_view old_directive :
         {"representatives_file", "representative_labels_file"}) {
        write_text(path, base + std::string(old_directive) +
                             "\told.npy\nrun\texact\texact\n");
        expect_failure([&] { static_cast<void>(load_benchmark_setup(path)); },
                       "unknown directive");
    }
    write_text(path, base +
                         "run\texact\texact\tstrategy=parallel_representatives\n");
    expect_failure([&] { static_cast<void>(load_benchmark_setup(path)); },
                   "unknown query strategy");
}

void test_rejects_invalid_reference_and_backend_strategy()
{
    const TemporaryDirectory temporary;
    const auto reference = temporary.path() / "reference.tsv";
    write_text(
        reference,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "reference\tflat\n"
        "run\tflat\tflat\trepetitions=1\tseed=1\n");
    expect_failure(
        [&] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(reference));
        },
        "reference run must use the exact index");

    const auto strategy = temporary.path() / "strategy.tsv";
    write_text(
        strategy,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "reference\texact\n"
        "run\texact\texact\tbackend=cpu\tstrategy=gemm\n");
    expect_failure(
        [&] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(strategy));
        },
        "CPU runs cannot use direct or gemm");
}

void test_rejects_version_one_and_duplicate_outputs()
{
    const TemporaryDirectory temporary;
    const auto old = temporary.path() / "old.tsv";
    write_text(old, "ultrahigh_ann_benchmark_setup_v1\n");
    expect_failure(
        [&] {
            static_cast<void>(
                ultrahigh_ann::benchmark::load_benchmark_setup(old));
        },
        "expected format header ultrahigh_ann_benchmark_setup_v2");

    const auto same_output = temporary.path() / "same-output.tsv";
    write_text(
        same_output,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.out\n"
        "csv_output\treport.out\n"
        "reference\texact\n"
        "run\texact\texact\n");
    expect_failure(
        [&] {
            static_cast<void>(ultrahigh_ann::benchmark::load_benchmark_setup(
                same_output));
        },
        "json_output and csv_output must differ");
}

void test_loads_cublas_probability_policy()
{
    using namespace ultrahigh_ann::benchmark;
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "cublas.tsv";
    write_text(
        path,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl2\n"
        "probability_policy\tgpu_cublas_fp32\n"
        "reference\texact\n"
        "run\texact\texact\n"
        "run\tflat\tflat\trepetitions=8\tseed=42\n");

    const BenchmarkSetup setup = load_benchmark_setup(path);
    expect(setup.probability_policy == ProbabilityPolicy::gpu_cublas_fp32,
           "cuBLAS probability policy changed");
    expect(probability_policy_name(setup.probability_policy) ==
               "gpu_cublas_fp32",
           "cuBLAS probability policy name changed");

    const auto l1_path = temporary.path() / "l1-cublas.tsv";
    write_text(
        l1_path,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl1\n"
        "probability_policy\tgpu_cublas_fp32\n"
        "reference\texact\n"
        "run\texact\texact\n"
        "run\tflat\tflat\trepetitions=8\tseed=42\n");
    expect_failure(
        [&] { static_cast<void>(load_benchmark_setup(l1_path)); },
        "gpu_cublas_fp32 is unavailable for L1");
}

void test_loads_l1_cuda_runs_and_rejects_nonhierarchical_gemm()
{
    using namespace ultrahigh_ann::benchmark;
    const TemporaryDirectory temporary;
    const auto valid = temporary.path() / "l1-cuda.tsv";
    write_text(
        valid,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl1\n"
        "reference\texact_cuda\n"
        "run\texact_cuda\texact\tbackend=cuda\n"
        "run\tflat_cuda\tflat\tbackend=cuda\tstrategy=direct\t"
        "repetitions=8\tseed=42\n");

    const BenchmarkSetup setup = load_benchmark_setup(valid);
    expect(setup.runs.size() == 2, "L1 CUDA run count changed");
    expect(setup.runs[0].strategy == QueryStrategy::direct,
           "L1 CUDA must default to direct queries");
    expect(setup.runs[1].backend == ExecutionBackend::cuda,
           "L1 CUDA backend changed");

    const auto invalid = temporary.path() / "l1-gemm.tsv";
    write_text(
        invalid,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl1\n"
        "reference\texact_cuda\n"
        "run\texact_cuda\texact\tbackend=cuda\tstrategy=gemm\n");
    expect_failure(
        [&] {
            static_cast<void>(load_benchmark_setup(invalid));
        },
        "CUDA L1 exact, flat, and uniform strategies must be direct");
}

void test_loads_cuda_hierarchies()
{
    using namespace ultrahigh_ann::benchmark;
    const TemporaryDirectory temporary;
    const auto valid = temporary.path() / "l2-cuda-hierarchy.tsv";
    write_text(
        valid,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl2\n"
        "reference\texact_cuda\n"
        "run\texact_cuda\texact\tbackend=cuda\tstrategy=direct\n"
        "run\thier_direct\thierarchical\tbackend=cuda\tstrategy=direct\t"
        "repetitions=8\tseed=42\tprojection_dimension=17\n"
        "run\thier_gemm\thierarchical\tbackend=cuda\tstrategy=gemm\t"
        "repetitions=8\tseed=43\tprojection_dimension=17\n");

    const BenchmarkSetup setup = load_benchmark_setup(valid);
    expect(setup.runs.size() == 3, "CUDA hierarchy run count changed");
    expect(setup.runs[1].strategy == QueryStrategy::direct,
           "CUDA hierarchy direct strategy changed");
    expect(setup.runs[2].strategy == QueryStrategy::gemm,
           "CUDA hierarchy GEMM strategy changed");

    const auto l1 = temporary.path() / "l1-cuda-hierarchy.tsv";
    write_text(
        l1,
        "ultrahigh_ann_benchmark_setup_v2\n"
        "dataset\tdataset\n"
        "json_output\treport.json\n"
        "csv_output\treport.csv\n"
        "distance\tl1\n"
        "reference\texact_cuda\n"
        "run\texact_cuda\texact\tbackend=cuda\n"
        "run\thier_direct\thierarchical\tbackend=cuda\tstrategy=direct\t"
        "repetitions=8\tseed=42\tprojection_dimension=17\n"
        "run\thier_gemm\thierarchical\tbackend=cuda\tstrategy=gemm\t"
        "repetitions=8\tseed=43\tprojection_dimension=17\n");
    const BenchmarkSetup l1_setup = load_benchmark_setup(l1);
    expect(l1_setup.runs.size() == 3, "CUDA L1 hierarchy run count changed");
    expect(l1_setup.runs[1].strategy == QueryStrategy::direct,
           "CUDA L1 hierarchy direct strategy changed");
    expect(l1_setup.runs[2].strategy == QueryStrategy::gemm,
           "CUDA L1 hierarchy GEMM strategy changed");
}

}  // namespace

int main()
{
    test_loads_cpu_and_cuda_runs();
    test_reference_vector_configuration();
    test_rejects_invalid_reference_and_backend_strategy();
    test_rejects_version_one_and_duplicate_outputs();
    test_loads_cublas_probability_policy();
    test_loads_l1_cuda_runs_and_rejects_nonhierarchical_gemm();
    test_loads_cuda_hierarchies();
    return 0;
}
