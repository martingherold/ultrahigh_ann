#include "hnsvw25/l2/flat/l2_ann_index.hpp"

#include "hnsvw25/l2/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

FlatL2AnnIndex::FlatL2AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : index_(
          input,
          compute_l2_importance_probabilities(input),
          repetitions,
          random_engine)
{
}

FlatL2AnnIndex::FlatL2AnnIndex(
    const DenseMatrix& input,
    std::span<const double> importance_probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : index_(
          input,
          importance_probabilities,
          repetitions,
          random_engine)
{
}

std::size_t FlatL2AnnIndex::query(std::span<const float> query) const
{
    return index_.query(query);
}

IndexSpaceUsage FlatL2AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
