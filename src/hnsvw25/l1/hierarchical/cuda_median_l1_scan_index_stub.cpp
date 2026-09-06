#include "hnsvw25/l1/hierarchical/cuda_median_l1_scan_index.hpp"

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

struct CudaMedianL1ScanIndex::Impl {};
struct CudaMedianL1ScanIndex::Workspace::Impl {};

CudaMedianL1ScanIndex::Workspace::Workspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{}

CudaMedianL1ScanIndex::Workspace::Workspace(Workspace&&) noexcept = default;

CudaMedianL1ScanIndex::Workspace&
CudaMedianL1ScanIndex::Workspace::operator=(Workspace&&) noexcept = default;

CudaMedianL1ScanIndex::Workspace::~Workspace() = default;

std::size_t
CudaMedianL1ScanIndex::Workspace::maximum_batch_size() const noexcept
{
    return 0;
}

std::size_t CudaMedianL1ScanIndex::Workspace::payload_bytes() const noexcept
{
    return 0;
}

CudaMedianL1ScanIndex::CudaMedianL1ScanIndex(
    const DenseMatrix& projected_representatives, int device)
{
    static_cast<void>(projected_representatives);
    static_cast<void>(device);
    throw_cuda_unavailable();
}

CudaMedianL1ScanIndex::CudaMedianL1ScanIndex(CudaMedianL1ScanIndex&&) noexcept =
    default;

CudaMedianL1ScanIndex&
CudaMedianL1ScanIndex::operator=(CudaMedianL1ScanIndex&&) noexcept = default;

CudaMedianL1ScanIndex::~CudaMedianL1ScanIndex() = default;

CudaMedianL1ScanIndex::Workspace
CudaMedianL1ScanIndex::make_workspace(std::size_t maximum_batch_size) const
{
    static_cast<void>(maximum_batch_size);
    throw_cuda_unavailable();
}

void CudaMedianL1ScanIndex::query_device_batch(const float* projected_queries,
                                               std::size_t query_count,
                                               std::span<std::size_t> output,
                                               Workspace& workspace) const
{
    static_cast<void>(projected_queries);
    static_cast<void>(query_count);
    static_cast<void>(output);
    static_cast<void>(workspace);
    throw_cuda_unavailable();
}

std::size_t CudaMedianL1ScanIndex::payload_bytes() const noexcept
{
    return 0;
}

std::size_t CudaMedianL1ScanIndex::representative_count() const noexcept
{
    return 0;
}

std::size_t CudaMedianL1ScanIndex::projection_dimension() const noexcept
{
    return 0;
}

int CudaMedianL1ScanIndex::device() const noexcept
{
    return 0;
}

const std::string& CudaMedianL1ScanIndex::device_name() const noexcept
{
    static const std::string unavailable{"unavailable"};
    return unavailable;
}

}  // namespace ultrahigh_ann::detail
