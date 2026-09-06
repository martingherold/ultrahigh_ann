#pragma once

#include "ultrahigh_ann/io/sha256.hpp"

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace ultrahigh_ann::io {

enum class ImportanceProbabilityKind {
    l1,
    l2,
};

struct ImportanceProbabilityFile {
    ImportanceProbabilityKind kind{};
    std::size_t representative_count{};
    Sha256Digest representatives_sha256{};
    std::vector<double> probabilities;
};

// Store a self-describing, versioned binary probability file bound to the
// exact representative-matrix file from which it was computed. Values are
// encoded as little-endian IEEE-754 binary64 numbers.
void save_importance_probabilities(const std::filesystem::path& path,
                                   ImportanceProbabilityKind kind,
                                   std::size_t representative_count,
                                   const Sha256Digest& representatives_sha256,
                                   std::span<const double> probabilities);

[[nodiscard]] ImportanceProbabilityFile
load_importance_probabilities(const std::filesystem::path& path);

[[nodiscard]] bool importance_probability_source_matches(
    const ImportanceProbabilityFile& file, ImportanceProbabilityKind kind,
    std::size_t representative_count, std::size_t coordinate_count,
    const Sha256Digest& representatives_sha256) noexcept;

}  // namespace ultrahigh_ann::io
