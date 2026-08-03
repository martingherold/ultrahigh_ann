#include "hnsvw25/l1/importance_sampling.hpp"
#include "hnsvw25/l1/importance_sampling_detail.hpp"
#include "core/finite_values.hpp"
#include "core/l1_distance.hpp"
#include "coordinate_sampling/coordinate_sampling.hpp"

#include <vector>
#include <cmath>


namespace ultrahigh_ann {

namespace detail {

std::vector<double> compute_l1_sampling_probabilities(
    const DenseMatrix& representatives)
{
    const std::size_t rows = representatives.rows();
    const std::size_t cols = representatives.cols();

    validate_finite_values(
        representatives.values(),
        "representatives");

    std::vector<double> probabilities(cols,0.0);
    
    for(std::size_t i=0;i<rows;++i){
        const auto row1 = representatives.row(i);
        for (std::size_t j=i+1;j<rows;++j){
            const auto row2 = representatives.row(j);
            
            double dist= detail::l1_dist(row1,row2);
            


            if(dist ==0){
                continue;
            }
            const double inverse_dist = 1.0 / dist;
            for(std::size_t d=0;d<cols;++d){
                //Check later if removing the static_cast improves performance.
                const double probability = std::abs(static_cast<double>(row1[d])-static_cast<double>(row2[d])) * inverse_dist;

                if(probability > probabilities[d]){
                    probabilities[d] = probability;
                }
            }
        }
    }
    return probabilities;
}

}  // namespace detail



std::vector<double> compute_l1_importance_probabilities(
    const DenseMatrix& representatives)
{
    return detail::compute_l1_sampling_probabilities(representatives);
}

CoordinateSample build_importance_sample(
    std::span<const double> probabilities,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    return build_coordinate_sample(
        probabilities,
        repetitions,
        random_engine);
}

CoordinateSample build_importance_sample(
    const DenseMatrix& representatives,
    std::size_t repetitions,
    std::mt19937_64& random_engine)
{
    const auto probabilities =
        compute_l1_importance_probabilities(representatives);
    return build_importance_sample(
        probabilities,
        repetitions,
        random_engine);
}

}  // namespace ultrahigh_ann
