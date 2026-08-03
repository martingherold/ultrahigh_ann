#include "hnsvw25/l1/flat/l1_ann_index.hpp"

#include "hnsvw25/l1/importance_sampling.hpp"

#include <cstddef>
#include <random>
#include <span>
namespace ultrahigh_ann {

FlatL1AnnIndex::FlatL1AnnIndex(
    const DenseMatrix& input,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : index_(
          input,
          compute_l1_importance_probabilities(input),
          repetitions,
          random_engine)
{
}

FlatL1AnnIndex::FlatL1AnnIndex(
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

std::size_t FlatL1AnnIndex::query(
    std::span<const float> query) const
{
    return index_.query(query);
}

IndexSpaceUsage FlatL1AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
