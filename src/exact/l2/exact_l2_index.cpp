#include "ultrahigh_ann/exact/l2/exact_l2_index.hpp"

#include <cstddef>
#include <span>

namespace ultrahigh_ann {

ExactL2Index::ExactL2Index(const DenseMatrix& representatives)
    : scan_(representatives)
{
}

std::size_t ExactL2Index::query(std::span<const float> query,
                                CpuDenseL2QueryStrategy strategy) const
{
    return scan_.query(query, strategy);
}

void ExactL2Index::query_batch(std::span<const float> queries,
                               std::size_t query_count,
                               std::span<std::size_t> output,
                               CpuDenseL2QueryStrategy strategy) const
{
    scan_.query_batch(queries, query_count, output, strategy);
}

IndexSpaceUsage ExactL2Index::space_usage() const noexcept
{
    return scan_.space_usage();
}

}  // namespace ultrahigh_ann
