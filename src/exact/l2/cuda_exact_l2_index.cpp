#include "ultrahigh_ann/exact/l2/cuda_exact_l2_index.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

bool cuda_exact_l2_available() noexcept
{
    return cuda_dense_l2_scan_available();
}

CudaExactL2Index::CudaExactL2Index(const DenseMatrix& representatives,
                                   int device)
    : scan_(representatives, device)
{
}

CudaExactL2Index::CudaExactL2Index(CudaExactL2Index&&) noexcept = default;

CudaExactL2Index& CudaExactL2Index::operator=(CudaExactL2Index&&) noexcept =
    default;

CudaExactL2Index::~CudaExactL2Index() = default;

CudaExactL2Index::QueryWorkspace CudaExactL2Index::make_query_workspace(
    std::size_t maximum_batch_size) const
{
    return scan_.make_query_workspace(maximum_batch_size);
}

std::size_t CudaExactL2Index::query(std::span<const float> query,
                                    QueryWorkspace& workspace) const
{
    return scan_.query(query, workspace);
}

void CudaExactL2Index::query_batch(std::span<const float> queries,
                                   std::size_t query_count,
                                   std::span<std::size_t> output,
                                   QueryWorkspace& workspace,
                                   CudaExactL2QueryStrategy strategy) const
{
    scan_.query_batch(queries, query_count, output, workspace, strategy);
}

IndexSpaceUsage CudaExactL2Index::space_usage() const noexcept
{
    return scan_.space_usage();
}

int CudaExactL2Index::device() const noexcept
{
    return scan_.device();
}

const std::string& CudaExactL2Index::device_name() const noexcept
{
    return scan_.device_name();
}

}  // namespace ultrahigh_ann
