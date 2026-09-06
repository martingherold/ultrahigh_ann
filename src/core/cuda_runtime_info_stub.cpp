#include "ultrahigh_ann/core/cuda_runtime_info.hpp"

#include <optional>

namespace ultrahigh_ann {

std::optional<CudaRuntimeInfo> cuda_runtime_info(int) noexcept
{
    return std::nullopt;
}

}  // namespace ultrahigh_ann
