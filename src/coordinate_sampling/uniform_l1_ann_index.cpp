#include "ultrahigh_ann/coordinate_sampling/uniform_l1_ann_index.hpp"

#include "ultrahigh_ann/coordinate_sampling/uniform_probabilities.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

UniformL1AnnIndex::UniformL1AnnIndex(
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

std::size_t UniformL1AnnIndex::query(
    std::span<const float> query) const
{
    return index_.query(query);
}

IndexSpaceUsage UniformL1AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
