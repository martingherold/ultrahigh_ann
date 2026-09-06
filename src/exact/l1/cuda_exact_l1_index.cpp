#include "ultrahigh_ann/exact/l1/cuda_exact_l1_index.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

bool cuda_exact_l1_available() noexcept
{
    return cuda_dense_l1_scan_available();
}

CudaExactL1Index::CudaExactL1Index(const DenseMatrix& representatives,
                                   int device)
    : scan_(representatives, device)
{
}

CudaExactL1Index::CudaExactL1Index(CudaExactL1Index&&) noexcept = default;

CudaExactL1Index&
CudaExactL1Index::operator=(CudaExactL1Index&&) noexcept = default;

CudaExactL1Index::~CudaExactL1Index() = default;

CudaExactL1Index::QueryWorkspace
CudaExactL1Index::make_query_workspace(std::size_t maximum_batch_size) const
{
    return scan_.make_query_workspace(maximum_batch_size);
}

std::size_t CudaExactL1Index::query(std::span<const float> query,
                                    QueryWorkspace& workspace) const
{
    return scan_.query(query, workspace);
}

void CudaExactL1Index::query_batch(std::span<const float> queries,
                                   std::size_t query_count,
                                   std::span<std::size_t> output,
                                   QueryWorkspace& workspace) const
{
    scan_.query_batch(queries, query_count, output, workspace);
}

IndexSpaceUsage CudaExactL1Index::space_usage() const noexcept
{
    return scan_.space_usage();
}

int CudaExactL1Index::device() const noexcept
{
    return scan_.device();
}

const std::string& CudaExactL1Index::device_name() const noexcept
{
    return scan_.device_name();
}

} // namespace ultrahigh_ann
