#pragma once

#include "ultrahigh_ann/core/dense_matrix.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace ultrahigh_ann::detail {

// Exhaustive median-of-absolute-differences scan over binary32 projected
// representatives. Queries must already reside on the same CUDA device.
class CudaMedianL1ScanIndex {
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
        friend class CudaMedianL1ScanIndex;
        struct Impl;

        explicit Workspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    explicit CudaMedianL1ScanIndex(const DenseMatrix& projected_representatives,
                                   int device);
    CudaMedianL1ScanIndex(const CudaMedianL1ScanIndex&) = delete;
    CudaMedianL1ScanIndex& operator=(const CudaMedianL1ScanIndex&) = delete;
    CudaMedianL1ScanIndex(CudaMedianL1ScanIndex&&) noexcept;
    CudaMedianL1ScanIndex& operator=(CudaMedianL1ScanIndex&&) noexcept;
    ~CudaMedianL1ScanIndex();

    [[nodiscard]] Workspace
    make_workspace(std::size_t maximum_batch_size) const;

    void query_device_batch(const float* projected_queries,
                            std::size_t query_count,
                            std::span<std::size_t> output,
                            Workspace& workspace) const;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t representative_count() const noexcept;
    [[nodiscard]] std::size_t projection_dimension() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

}  // namespace ultrahigh_ann::detail
