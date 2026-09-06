#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include <stdexcept>

namespace ultrahigh_ann {

bool l2_gpu_backend_available() noexcept
{
    return false;
}

L2GpuProbabilityResult
compute_l2_importance_probabilities_gpu(const DenseMatrix& representatives,
                                        int device, std::size_t pair_chunks,
                                        L2GpuDistanceBackend distance_backend)
{
    static_cast<void>(representatives);
    static_cast<void>(device);
    static_cast<void>(pair_chunks);
    static_cast<void>(distance_backend);
    throw std::runtime_error("CUDA support is not enabled; reconfigure with "
                             "-DULTRAHIGH_ANN_ENABLE_CUDA=ON");
}

}  // namespace ultrahigh_ann
