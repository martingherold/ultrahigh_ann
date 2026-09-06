#include "compute_importance_probabilities.hpp"

int main(int argc, char** argv)
{
    return ultrahigh_ann::tools::run_compute_importance_probabilities(
        argc, argv, ultrahigh_ann::tools::ProbabilityDistance::l1);
}
