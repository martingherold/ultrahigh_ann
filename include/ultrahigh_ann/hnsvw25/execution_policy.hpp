#pragma once

namespace ultrahigh_ann {

// Selects where reusable importance-sampling probabilities are built. GPU
// policies use binary32 arithmetic and return the common binary64 vector.
enum class ExecutionPolicy {
    sequential,
    cpu_parallel,
    gpu_fp32,
    gpu_cublas_fp32,
};

}  // namespace ultrahigh_ann
