#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace ultrahigh_ann::detail {

enum class CudaFloatProjectionStrategy {
    direct,
    gemm,
};

// Device-side multiplication of a compact binary32 query matrix by a
// transposed binary32 projection matrix. Results remain device-resident so a
// CUDA scan can consume them without a host round-trip.
class CudaFloatProjection {
  public:
    class Workspace {
      public:
        Workspace(const Workspace&) = delete;
        Workspace& operator=(const Workspace&) = delete;
        Workspace(Workspace&&) noexcept;
        Workspace& operator=(Workspace&&) noexcept;
        ~Workspace();

        [[nodiscard]] std::size_t maximum_batch_size() const noexcept;
        [[nodiscard]] std::size_t payload_bytes() const noexcept;

      private:
        friend class CudaFloatProjection;
        struct Impl;

        explicit Workspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    CudaFloatProjection(const DenseMatrix& projection_matrix, int device);
    CudaFloatProjection(const CudaFloatProjection&) = delete;
    CudaFloatProjection& operator=(const CudaFloatProjection&) = delete;
    CudaFloatProjection(CudaFloatProjection&&) noexcept;
    CudaFloatProjection& operator=(CudaFloatProjection&&) noexcept;
    ~CudaFloatProjection();

    [[nodiscard]] Workspace
    make_workspace(std::size_t maximum_batch_size) const;

    // sampled_queries is a host-resident row-major query_count by
    // sampled_dimension matrix. Returned pointers address row-major
    // query_count by projection_dimension matrices on device(). They remain
    // valid until the workspace is reused or destroyed.
    [[nodiscard]] const float*
    project_batch_float(std::span<const float> sampled_queries,
                        std::size_t query_count, Workspace& workspace,
                        CudaFloatProjectionStrategy strategy) const;

    [[nodiscard]] std::size_t sampled_dimension() const noexcept;
    [[nodiscard]] std::size_t projection_dimension() const noexcept;
    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace ultrahigh_ann::detail
