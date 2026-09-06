#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"

#include <stdexcept>
#include <utility>

namespace ultrahigh_ann {
namespace {

[[noreturn]] void throw_cuda_unavailable()
{
    throw std::runtime_error("CUDA support is not enabled; reconfigure with "
                             "-DULTRAHIGH_ANN_ENABLE_CUDA=ON");
}

}  // namespace

struct CudaDenseL2ScanIndex::Impl {};

struct CudaDenseL2ScanIndex::QueryWorkspace::Impl {};

bool cuda_dense_l2_scan_available() noexcept
{
    return false;
}

CudaDenseL2ScanIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaDenseL2ScanIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaDenseL2ScanIndex::QueryWorkspace& CudaDenseL2ScanIndex::QueryWorkspace::
operator=(QueryWorkspace&&) noexcept = default;

CudaDenseL2ScanIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t CudaDenseL2ScanIndex::QueryWorkspace::maximum_batch_size()
    const noexcept
{
    return 0;
}

std::size_t CudaDenseL2ScanIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return 0;
}

CudaDenseL2ScanIndex::CudaDenseL2ScanIndex(const DenseMatrix& representatives,
                                           int device)
{
    static_cast<void>(representatives);
    static_cast<void>(device);
    throw_cuda_unavailable();
}

CudaDenseL2ScanIndex::CudaDenseL2ScanIndex(CudaDenseL2ScanIndex&&) noexcept =
    default;

CudaDenseL2ScanIndex& CudaDenseL2ScanIndex::operator=(
    CudaDenseL2ScanIndex&&) noexcept = default;

CudaDenseL2ScanIndex::~CudaDenseL2ScanIndex() = default;

CudaDenseL2ScanIndex::QueryWorkspace CudaDenseL2ScanIndex::make_query_workspace(
    std::size_t maximum_batch_size) const
{
    static_cast<void>(maximum_batch_size);
    throw_cuda_unavailable();
}

std::size_t CudaDenseL2ScanIndex::query(std::span<const float> query,
                                        QueryWorkspace& workspace) const
{
    static_cast<void>(query);
    static_cast<void>(workspace);
    throw_cuda_unavailable();
}

void CudaDenseL2ScanIndex::query_batch(std::span<const float> queries,
                                       std::size_t query_count,
                                       std::span<std::size_t> output,
                                       QueryWorkspace& workspace,
                                       CudaDenseL2QueryStrategy strategy) const
{
    static_cast<void>(queries);
    static_cast<void>(query_count);
    static_cast<void>(output);
    static_cast<void>(workspace);
    static_cast<void>(strategy);
    throw_cuda_unavailable();
}

void CudaDenseL2ScanIndex::query_device_batch(
    const float* device_queries,
    std::size_t query_count,
    std::span<std::size_t> output,
    QueryWorkspace& workspace,
    CudaDenseL2QueryStrategy strategy) const
{
    static_cast<void>(device_queries);
    static_cast<void>(query_count);
    static_cast<void>(output);
    static_cast<void>(workspace);
    static_cast<void>(strategy);
    throw_cuda_unavailable();
}

IndexSpaceUsage CudaDenseL2ScanIndex::space_usage() const noexcept
{
    return {};
}

int CudaDenseL2ScanIndex::device() const noexcept
{
    return 0;
}

const std::string& CudaDenseL2ScanIndex::device_name() const noexcept
{
    static const std::string unavailable{"unavailable"};
    return unavailable;
}

}  // namespace ultrahigh_ann
