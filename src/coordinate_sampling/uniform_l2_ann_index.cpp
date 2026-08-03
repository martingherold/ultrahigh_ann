#include "coordinate_sampling/uniform_l2_ann_index.hpp"

#include "coordinate_sampling/uniform_probabilities.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

UniformL2AnnIndex::UniformL2AnnIndex(
    const DenseMatrix& input,
    double sampling_mass,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
    : index_(
          input,
          make_mass_matched_uniform_probabilities(
              input.cols(),
              sampling_mass),
          repetitions,
          random_engine)
{
}

std::size_t UniformL2AnnIndex::query(
    std::span<const float> query) const
{
    return index_.query(query);
}

IndexSpaceUsage UniformL2AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
