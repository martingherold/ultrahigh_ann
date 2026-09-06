#include "ultrahigh_ann/core/cuda_runtime_info.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>
#include <string>

namespace ultrahigh_ann {

std::optional<CudaRuntimeInfo> cuda_runtime_info(int device) noexcept
{
    int device_count{};
    if (device < 0 || cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device >= device_count) {
        return std::nullopt;
    }

    cudaDeviceProp properties{};
    int runtime_version{};
    int driver_version{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        return std::nullopt;
    }

    return CudaRuntimeInfo{
        .device = device,
        .device_name = properties.name,
        .compute_capability_major = properties.major,
        .compute_capability_minor = properties.minor,
        .total_global_memory_bytes = properties.totalGlobalMem,
        .compiled_runtime_version = CUDART_VERSION,
        .runtime_version = runtime_version,
        .driver_version = driver_version,
    };
}

}  // namespace ultrahigh_ann
