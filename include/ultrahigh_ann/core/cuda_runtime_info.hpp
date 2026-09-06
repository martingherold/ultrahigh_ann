#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace ultrahigh_ann {

struct CudaRuntimeInfo {
    int device{};
    std::string device_name;
    int compute_capability_major{};
    int compute_capability_minor{};
    std::size_t total_global_memory_bytes{};
    int compiled_runtime_version{};
    int runtime_version{};
    int driver_version{};
};

// Return metadata for a usable device. A CUDA-disabled build or a runtime with
// no usable device returns nullopt.
[[nodiscard]] std::optional<CudaRuntimeInfo>
cuda_runtime_info(int device = 0) noexcept;

}  // namespace ultrahigh_ann
