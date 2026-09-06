#include "ultrahigh_ann/coordinate_sampling/uniform_l2_ann_index.hpp"

#include "ultrahigh_ann/coordinate_sampling/uniform_probabilities.hpp"

#include <cstddef>
#include <random>
#include <span>

namespace ultrahigh_ann {

UniformL2AnnIndex::UniformL2AnnIndex(const DenseMatrix& input,
                                     double sampling_mass,
                                     std::size_t repetitions,
                                     std::mt19937_64& random_engine)
    : index_(
          input,
          make_mass_matched_uniform_probabilities(input.cols(), sampling_mass),
          repetitions,
          random_engine)
{
}

std::size_t UniformL2AnnIndex::query(std::span<const float> query,
                                     CpuDenseL2QueryStrategy strategy) const
{
    return index_.query(query, strategy);
}

void UniformL2AnnIndex::query_batch(std::span<const float> queries,
                                    std::size_t query_count,
                                    std::span<std::size_t> output,
                                    CpuDenseL2QueryStrategy strategy) const
{
    index_.query_batch(queries, query_count, output, strategy);
}

IndexSpaceUsage UniformL2AnnIndex::space_usage() const noexcept
{
    return index_.space_usage();
}

}  // namespace ultrahigh_ann
