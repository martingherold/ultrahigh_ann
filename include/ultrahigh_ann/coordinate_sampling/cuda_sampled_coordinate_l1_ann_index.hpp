#pragma once

#include "ultrahigh_ann/coordinate_sampling/coordinate_sample.hpp"
#include "ultrahigh_ann/core/dense_matrix.hpp"
#include "ultrahigh_ann/core/index_space_usage.hpp"

#include <cstddef>
#include <memory>
#include <random>
#include <span>
#include <string>

namespace ultrahigh_ann {

// True when CUDA support is compiled in and a usable device is visible.
[[nodiscard]] bool cuda_sampled_coordinate_l1_available() noexcept;

// Weighted sampled-coordinate facade over CudaDenseL1ScanIndex. The sampled
// representatives and weights remain on one CUDA device, and only sampled
// query coordinates are packed and transferred.
class CudaSampledCoordinateL1AnnIndex {
public:
    class QueryWorkspace {
    public:
        QueryWorkspace(const QueryWorkspace&) = delete;
        QueryWorkspace& operator=(const QueryWorkspace&) = delete;
        QueryWorkspace(QueryWorkspace&&) noexcept;
        QueryWorkspace& operator=(QueryWorkspace&&) noexcept;
        ~QueryWorkspace();

        [[nodiscard]] std::size_t maximum_batch_size() const noexcept;
        [[nodiscard]] std::size_t payload_bytes() const noexcept;

    private:
        friend class CudaSampledCoordinateL1AnnIndex;
        struct Impl;

        explicit QueryWorkspace(std::unique_ptr<Impl> implementation);
        std::unique_ptr<Impl> implementation_;
    };

    CudaSampledCoordinateL1AnnIndex(const DenseMatrix& input,
                                    std::span<const double> probabilities,
                                    std::size_t repetitions,
                                    std::mt19937_64& random_engine,
                                    int device = 0);

    CudaSampledCoordinateL1AnnIndex(const DenseMatrix& input,
                                    CoordinateSample coordinate_sample,
                                    int device = 0);

    CudaSampledCoordinateL1AnnIndex(const CudaSampledCoordinateL1AnnIndex&) =
        delete;
    CudaSampledCoordinateL1AnnIndex&
    operator=(const CudaSampledCoordinateL1AnnIndex&) = delete;
    CudaSampledCoordinateL1AnnIndex(CudaSampledCoordinateL1AnnIndex&&) noexcept;
    CudaSampledCoordinateL1AnnIndex&
    operator=(CudaSampledCoordinateL1AnnIndex&&) noexcept;
    ~CudaSampledCoordinateL1AnnIndex();

    [[nodiscard]] QueryWorkspace
    make_query_workspace(std::size_t maximum_batch_size) const;

    [[nodiscard]] std::size_t query(std::span<const float> query,
                                    QueryWorkspace& workspace) const;

    // Queries are a contiguous row-major query_count by original-dimension
    // matrix. Packing transfers only the sampled coordinates to the device.
    void query_batch(std::span<const float> queries, std::size_t query_count,
                     std::span<std::size_t> output,
                     QueryWorkspace& workspace) const;

    [[nodiscard]] IndexSpaceUsage space_usage() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace ultrahigh_ann
