#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ultrahigh_ann::io {

using Sha256Digest = std::array<std::uint8_t, 32>;

// Hash the exact bytes stored in a file. The digest is suitable for binding a
// derived artifact, such as an importance-probability file, to its source.
[[nodiscard]] Sha256Digest sha256_file(const std::filesystem::path& path);

[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

}  // namespace ultrahigh_ann::io
