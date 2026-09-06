#include "ultrahigh_ann/core/cuda_dense_l1_scan_index.hpp"

#include <stdexcept>
#include <utility>

namespace ultrahigh_ann {
namespace {

[[noreturn]] void throw_cuda_unavailable()
{
    throw std::runtime_error("CUDA support is not enabled; reconfigure with "
                             "-DULTRAHIGH_ANN_ENABLE_CUDA=ON");
}

} // namespace

struct CudaDenseL1ScanIndex::Impl {};

struct CudaDenseL1ScanIndex::QueryWorkspace::Impl {};

bool cuda_dense_l1_scan_available() noexcept
{
    return false;
}

CudaDenseL1ScanIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaDenseL1ScanIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaDenseL1ScanIndex::QueryWorkspace&
CudaDenseL1ScanIndex::QueryWorkspace::operator=(QueryWorkspace&&) noexcept =
    default;

CudaDenseL1ScanIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t
CudaDenseL1ScanIndex::QueryWorkspace::maximum_batch_size() const noexcept
{
    return 0;
}

std::size_t CudaDenseL1ScanIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return 0;
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(const DenseMatrix& representatives,
                                           int device)
{
    static_cast<void>(representatives);
    static_cast<void>(device);
    throw_cuda_unavailable();
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(
    const DenseMatrix& representatives,
    std::span<const float> coordinate_weights, int device)
{
    static_cast<void>(representatives);
    static_cast<void>(coordinate_weights);
    static_cast<void>(device);
    throw_cuda_unavailable();
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(CudaDenseL1ScanIndex&&) noexcept =
    default;

CudaDenseL1ScanIndex&
CudaDenseL1ScanIndex::operator=(CudaDenseL1ScanIndex&&) noexcept = default;

CudaDenseL1ScanIndex::~CudaDenseL1ScanIndex() = default;

CudaDenseL1ScanIndex::QueryWorkspace
CudaDenseL1ScanIndex::make_query_workspace(std::size_t maximum_batch_size) const
{
    static_cast<void>(maximum_batch_size);
    throw_cuda_unavailable();
}

std::size_t CudaDenseL1ScanIndex::query(std::span<const float> query,
                                        QueryWorkspace& workspace) const
{
    static_cast<void>(query);
    static_cast<void>(workspace);
    throw_cuda_unavailable();
}

void CudaDenseL1ScanIndex::query_batch(std::span<const float> queries,
                                       std::size_t query_count,
                                       std::span<std::size_t> output,
                                       QueryWorkspace& workspace) const
{
    static_cast<void>(queries);
    static_cast<void>(query_count);
    static_cast<void>(output);
    static_cast<void>(workspace);
    throw_cuda_unavailable();
}

IndexSpaceUsage CudaDenseL1ScanIndex::space_usage() const noexcept
{
    return {};
}

int CudaDenseL1ScanIndex::device() const noexcept
{
    return 0;
}

const std::string& CudaDenseL1ScanIndex::device_name() const noexcept
{
    static const std::string unavailable{"unavailable"};
    return unavailable;
}

} // namespace ultrahigh_ann
