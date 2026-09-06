#pragma once

namespace ultrahigh_ann::tools {

enum class ProbabilityDistance {
    l1,
    l2,
};

int run_compute_importance_probabilities(int argc, char** argv,
                                         ProbabilityDistance distance);

}  // namespace ultrahigh_ann::tools
