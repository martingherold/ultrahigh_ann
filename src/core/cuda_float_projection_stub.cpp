#include "core/cuda_float_projection.hpp"

#include <stdexcept>
#include <utility>

namespace ultrahigh_ann::detail {
namespace {

[[noreturn]] void throw_cuda_unavailable()
{
    throw std::runtime_error("CUDA support is not enabled; reconfigure with "
                             "-DULTRAHIGH_ANN_ENABLE_CUDA=ON");
}

}  // namespace

struct CudaFloatProjection::Impl {};
struct CudaFloatProjection::Workspace::Impl {};

CudaFloatProjection::Workspace::Workspace(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{}

CudaFloatProjection::Workspace::Workspace(Workspace&&) noexcept = default;

CudaFloatProjection::Workspace&
CudaFloatProjection::Workspace::operator=(Workspace&&) noexcept = default;

CudaFloatProjection::Workspace::~Workspace() = default;

std::size_t CudaFloatProjection::Workspace::maximum_batch_size() const noexcept
{
    return 0;
}

std::size_t CudaFloatProjection::Workspace::payload_bytes() const noexcept
{
    return 0;
}

CudaFloatProjection::CudaFloatProjection(const DenseMatrix& projection_matrix,
                                         int device)
{
    static_cast<void>(projection_matrix);
    static_cast<void>(device);
    throw_cuda_unavailable();
}

CudaFloatProjection::CudaFloatProjection(CudaFloatProjection&&) noexcept =
    default;

CudaFloatProjection&
CudaFloatProjection::operator=(CudaFloatProjection&&) noexcept = default;

CudaFloatProjection::~CudaFloatProjection() = default;

CudaFloatProjection::Workspace
CudaFloatProjection::make_workspace(std::size_t maximum_batch_size) const
{
    static_cast<void>(maximum_batch_size);
    throw_cuda_unavailable();
}

const float* CudaFloatProjection::project_batch_float(
    std::span<const float> sampled_queries, std::size_t query_count,
    Workspace& workspace, CudaFloatProjectionStrategy strategy) const
{
    static_cast<void>(sampled_queries);
    static_cast<void>(query_count);
    static_cast<void>(workspace);
    static_cast<void>(strategy);
    throw_cuda_unavailable();
}

std::size_t CudaFloatProjection::sampled_dimension() const noexcept
{
    return 0;
}

std::size_t CudaFloatProjection::projection_dimension() const noexcept
{
    return 0;
}

std::size_t CudaFloatProjection::payload_bytes() const noexcept
{
    return 0;
}

int CudaFloatProjection::device() const noexcept
{
    return 0;
}

}  // namespace ultrahigh_ann::detail
