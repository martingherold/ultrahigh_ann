#include "ultrahigh_ann/hnsvw25/l2/flat/l2_ann_index.hpp"

#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

FlatL2AnnIndex::FlatL2AnnIndex(const DenseMatrix& input,
                               std::size_t repetitions,
                               std::mt19937_64& random_engine)
    : index_(input,
             compute_l2_importance_probabilities(input),
             repetitions,
             random_engine)
{
}

FlatL2AnnIndex::FlatL2AnnIndex(const DenseMatrix& input,
                               std::span<const double> importance_probabilities,
                               std::size_t repetitions,
                               std::mt19937_64& random_engine)
    : index_(input, importance_probabilities, repetitions, random_engine)
{
}

std::size_t FlatL2AnnIndex::query(std::span<const float> query,
                                  CpuDenseL2QueryStrategy strategy) const
{
    return index_.query(query, strategy);
}

void FlatL2AnnIndex::query_batch(std::span<const float> queries,
                                 std::size_t query_count,
                                 std::span<std::size_t> output,
                                 CpuDenseL2QueryStrategy strategy) const
{
    index_.query_batch(queries, query_count, output, strategy);
}

IndexSpaceUsage FlatL2AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
