#include "ultrahigh_ann/io/importance_probability_io.hpp"

#include <chrono>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
    {
        const auto suffix =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("ultrahigh-ann-probability-io-test-" + std::to_string(suffix));
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

template<class Function>
void expect_failure(Function&& function, std::string_view expected_message)
{
    try {
        function();
    } catch (const std::exception& error) {
        expect(std::string_view(error.what()).find(expected_message) !=
                   std::string_view::npos,
               "unexpected probability-I/O error");
        return;
    }
    throw std::runtime_error("expected probability I/O to fail");
}

void test_round_trip_preserves_metadata_and_values()
{
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "nested" / "probabilities.uap";
    const std::vector<double> expected{0.0, 0.125, 0.5, 1.0};
    const ultrahigh_ann::io::Sha256Digest expected_digest{
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    };
    ultrahigh_ann::io::save_importance_probabilities(
        path, ultrahigh_ann::io::ImportanceProbabilityKind::l2, 7'028,
        expected_digest, expected);

    const auto loaded = ultrahigh_ann::io::load_importance_probabilities(path);
    expect(loaded.kind == ultrahigh_ann::io::ImportanceProbabilityKind::l2,
           "probability kind changed during round trip");
    expect(loaded.representative_count == 7'028,
           "representative count changed during round trip");
    expect(loaded.representatives_sha256 == expected_digest,
           "representative digest changed during round trip");
    expect(loaded.probabilities == expected,
           "probabilities changed during round trip");
    expect(std::filesystem::file_size(path) == 64 + expected.size() * 8,
           "probability file has an unexpected size");
    expect(ultrahigh_ann::io::importance_probability_source_matches(
               loaded, ultrahigh_ann::io::ImportanceProbabilityKind::l2,
               7'028, expected.size(), expected_digest),
           "valid source binding was rejected");
    auto wrong_digest = expected_digest;
    wrong_digest[0] ^= 0xffU;
    expect(!ultrahigh_ann::io::importance_probability_source_matches(
               loaded, ultrahigh_ann::io::ImportanceProbabilityKind::l2,
               7'028, expected.size(), wrong_digest),
           "same-shape wrong-matrix source binding was accepted");
    expect(!ultrahigh_ann::io::importance_probability_source_matches(
               loaded, ultrahigh_ann::io::ImportanceProbabilityKind::l1,
               7'028, expected.size(), expected_digest),
           "wrong probability metric was accepted");
    expect(!ultrahigh_ann::io::importance_probability_source_matches(
               loaded, ultrahigh_ann::io::ImportanceProbabilityKind::l2,
               33, expected.size(), expected_digest),
           "wrong representative count was accepted");
    expect(!ultrahigh_ann::io::importance_probability_source_matches(
               loaded, ultrahigh_ann::io::ImportanceProbabilityKind::l2,
               7'028, expected.size() + 1, expected_digest),
           "wrong coordinate count was accepted");
}

void test_sha256_matches_standard_vector()
{
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "abc.bin";
    {
        std::ofstream output(path, std::ios::binary);
        output << "abc";
    }
    const auto digest = ultrahigh_ann::io::sha256_file(path);
    expect(ultrahigh_ann::io::sha256_hex(digest) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "SHA-256 implementation failed the standard abc test vector");
}

void test_rejects_invalid_probability_values()
{
    const TemporaryDirectory temporary;
    const auto path = temporary.path() / "invalid.uap";
    const std::vector<double> too_large{1.1};
    const ultrahigh_ann::io::Sha256Digest digest{};
    expect_failure(
        [&] {
            ultrahigh_ann::io::save_importance_probabilities(
                path, ultrahigh_ann::io::ImportanceProbabilityKind::l1, 2,
                digest, too_large);
        },
        "finite and in [0, 1]");

    const std::vector<double> non_finite{
        std::numeric_limits<double>::quiet_NaN()};
    expect_failure(
        [&] {
            ultrahigh_ann::io::save_importance_probabilities(
                path, ultrahigh_ann::io::ImportanceProbabilityKind::l1, 2,
                digest, non_finite);
        },
        "finite and in [0, 1]");
}

void test_rejects_wrong_magic_and_truncated_payload()
{
    const TemporaryDirectory temporary;
    const auto wrong_magic = temporary.path() / "wrong-magic.uap";
    {
        std::ofstream output(wrong_magic, std::ios::binary);
        output << "not a probability file";
    }
    expect_failure(
        [&] {
            static_cast<void>(
                ultrahigh_ann::io::load_importance_probabilities(wrong_magic));
        },
        "wrong magic");

    const auto truncated = temporary.path() / "truncated.uap";
    const std::vector<double> values{0.25, 0.75};
    const ultrahigh_ann::io::Sha256Digest digest{};
    ultrahigh_ann::io::save_importance_probabilities(
        truncated, ultrahigh_ann::io::ImportanceProbabilityKind::l2, 3,
        digest, values);
    std::filesystem::resize_file(truncated,
                                 std::filesystem::file_size(truncated) - 1);
    expect_failure(
        [&] {
            static_cast<void>(
                ultrahigh_ann::io::load_importance_probabilities(truncated));
        },
        "truncated");
}

}  // namespace

int main()
{
    test_round_trip_preserves_metadata_and_values();
    test_sha256_matches_standard_vector();
    test_rejects_invalid_probability_values();
    test_rejects_wrong_magic_and_truncated_payload();
    return 0;
}
