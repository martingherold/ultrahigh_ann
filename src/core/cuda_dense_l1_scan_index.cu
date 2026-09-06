#include "ultrahigh_ann/core/cuda_dense_l1_scan_index.hpp"

#include "core/finite_values.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ultrahigh_ann {
namespace {

constexpr unsigned int threads_per_block = 256;
constexpr unsigned int warp_size = 32;
constexpr unsigned int warps_per_block = threads_per_block / warp_size;

void check_cuda(cudaError_t status, std::string_view operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          std::string_view description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right,
                                      std::string_view description)
{
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left + right;
}

template<class T>
[[nodiscard]] std::size_t checked_bytes(std::size_t count,
                                        std::string_view description)
{
    return checked_product(count, sizeof(T), description);
}

class SelectedDevice {
public:
    explicit SelectedDevice(int requested_device) : device_(requested_device)
    {
        if (device_ < 0) {
            throw std::invalid_argument(
                "CUDA device index must be nonnegative");
        }
        int device_count{};
        check_cuda(cudaGetDeviceCount(&device_count),
                   "cannot enumerate CUDA devices");
        if (device_ >= device_count) {
            throw std::invalid_argument("CUDA device index is out of range");
        }
        check_cuda(cudaSetDevice(device_), "cannot select CUDA device");
        check_cuda(cudaGetDeviceProperties(&properties_, device_),
                   "cannot query CUDA device properties");
    }

    [[nodiscard]] int index() const noexcept
    {
        return device_;
    }

    [[nodiscard]] const cudaDeviceProp& properties() const noexcept
    {
        return properties_;
    }

private:
    int device_{};
    cudaDeviceProp properties_{};
};

template<class T> class DeviceBuffer {
public:
    DeviceBuffer(int device, std::size_t count)
        : device_(device),
          byte_count_(checked_bytes<T>(count, "CUDA device buffer"))
    {
        if (byte_count_ == 0) {
            return;
        }
        check_cuda(cudaSetDevice(device_), "cannot select CUDA device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()),
                   "cannot allocate CUDA device memory");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            static_cast<void>(cudaSetDevice(device_));
            static_cast<void>(cudaFree(data_));
        }
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] const T* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return byte_count_;
    }

private:
    int device_{};
    std::size_t byte_count_{};
    T* data_{};
};

__device__ __forceinline__ float warp_sum(float value)
{
    for (unsigned int offset = warp_size / 2U; offset != 0; offset /= 2U) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value)
{
    __shared__ float warp_sums[warps_per_block];
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    value = warp_sum(value);
    if (lane == 0U) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0U) {
        value = lane < warps_per_block ? warp_sums[lane] : 0.0F;
        value = warp_sum(value);
    }
    return value;
}

__device__ __forceinline__ bool
candidate_is_better(float candidate_distance,
                    unsigned long long candidate_index, float current_distance,
                    unsigned long long current_index)
{
    return candidate_distance < current_distance ||
           (candidate_distance == current_distance &&
            candidate_index < current_index);
}

__device__ __forceinline__ float positive_infinity()
{
    return __int_as_float(0x7f800000);
}

__device__ __forceinline__ void warp_argmin(float& best_distance,
                                            unsigned long long& best_index)
{
    for (unsigned int offset = warp_size / 2U; offset != 0; offset /= 2U) {
        const float candidate_distance =
            __shfl_down_sync(0xffffffffU, best_distance, offset);
        const unsigned long long candidate_index =
            __shfl_down_sync(0xffffffffU, best_index, offset);
        if (candidate_is_better(candidate_distance, candidate_index,
                                best_distance, best_index)) {
            best_distance = candidate_distance;
            best_index = candidate_index;
        }
    }
}

__device__ __forceinline__ void block_argmin(float& best_distance,
                                             unsigned long long& best_index,
                                             float* warp_distances,
                                             unsigned long long* warp_indices)
{
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const unsigned int warp = threadIdx.x / warp_size;
    warp_argmin(best_distance, best_index);
    if (lane == 0U) {
        warp_distances[warp] = best_distance;
        warp_indices[warp] = best_index;
    }
    __syncthreads();

    if (warp == 0U) {
        best_distance =
            lane < warps_per_block ? warp_distances[lane] : positive_infinity();
        best_index = lane < warps_per_block ? warp_indices[lane] : ULLONG_MAX;
        warp_argmin(best_distance, best_index);
    }
}

__global__ void compute_dense_l1_distances(
    const float* representatives, const float* queries,
    const float* coordinate_weights, std::size_t representative_count,
    std::size_t dimension, std::size_t query_count, float* distances)
{
    const std::size_t pair_index = blockIdx.x;
    const std::size_t total_pairs = representative_count * query_count;
    if (pair_index >= total_pairs) {
        return;
    }
    const std::size_t query_index = pair_index / representative_count;
    const std::size_t representative_index =
        pair_index - query_index * representative_count;
    const float* representative =
        representatives + representative_index * dimension;
    const float* query = queries + query_index * dimension;

    float distance = 0.0F;
    for (std::size_t column = threadIdx.x; column < dimension;
         column += blockDim.x) {
        const float difference = fabsf(representative[column] - query[column]);
        distance += coordinate_weights == nullptr
                        ? difference
                        : coordinate_weights[column] * difference;
    }
    distance = block_sum(distance);
    if (threadIdx.x == 0U) {
        distances[pair_index] = distance;
    }
}

__global__ void reduce_dense_l1_argmin(const float* distances,
                                       std::size_t representative_count,
                                       std::size_t query_count,
                                       std::size_t* results)
{
    const std::size_t query_index = blockIdx.x;
    if (query_index >= query_count) {
        return;
    }

    float best_distance = positive_infinity();
    unsigned long long best_index = ULLONG_MAX;
    const float* query_distances =
        distances + query_index * representative_count;
    for (std::size_t representative_index = threadIdx.x;
         representative_index < representative_count;
         representative_index += blockDim.x) {
        const float distance = query_distances[representative_index];
        if (candidate_is_better(
                distance, static_cast<unsigned long long>(representative_index),
                best_distance, best_index)) {
            best_distance = distance;
            best_index = representative_index;
        }
    }

    __shared__ float warp_distances[warps_per_block];
    __shared__ unsigned long long warp_indices[warps_per_block];
    block_argmin(best_distance, best_index, warp_distances, warp_indices);
    if (threadIdx.x == 0U) {
        results[query_index] = static_cast<std::size_t>(best_index);
    }
}

void check_kernel_launch(std::string_view description)
{
    check_cuda(cudaGetLastError(), description);
}

} // namespace

struct CudaDenseL1ScanIndex::Impl {
    Impl(const DenseMatrix& representatives,
         std::span<const float> coordinate_weights, int requested_device)
        : selected_device(requested_device),
          representative_count(representatives.rows()),
          dimension(representatives.cols()),
          representative_value_count(representatives.values().size()),
          device_representatives(selected_device.index(),
                                 representative_value_count),
          device_coordinate_weights(selected_device.index(),
                                    coordinate_weights.size()),
          payload_bytes(checked_add(device_representatives.bytes(),
                                    device_coordinate_weights.bytes(),
                                    "CUDA dense L1 index")),
          device_name(selected_device.properties().name)
    {
        if (representative_count == 0) {
            throw std::invalid_argument(
                "CudaDenseL1ScanIndex expects at least one representative");
        }
        if (!coordinate_weights.empty() &&
            coordinate_weights.size() != dimension) {
            throw std::invalid_argument(
                "CUDA dense L1 coordinate weights do not match dimension");
        }
        detail::validate_finite_values(representatives.values(),
                                       "representatives");
        for (const float weight : coordinate_weights) {
            if (!std::isfinite(weight) || weight <= 0.0F) {
                throw std::invalid_argument(
                    "CUDA dense L1 coordinate weights must be positive and "
                    "finite");
            }
        }
        if (device_representatives.bytes() != 0) {
            check_cuda(cudaMemcpy(device_representatives.data(),
                                  representatives.values().data(),
                                  device_representatives.bytes(),
                                  cudaMemcpyHostToDevice),
                       "cannot copy representatives to CUDA device");
        }
        if (device_coordinate_weights.bytes() != 0) {
            check_cuda(cudaMemcpy(device_coordinate_weights.data(),
                                  coordinate_weights.data(),
                                  device_coordinate_weights.bytes(),
                                  cudaMemcpyHostToDevice),
                       "cannot copy L1 coordinate weights to CUDA device");
        }
    }

    SelectedDevice selected_device;
    std::size_t representative_count{};
    std::size_t dimension{};
    std::size_t representative_value_count{};
    DeviceBuffer<float> device_representatives;
    DeviceBuffer<float> device_coordinate_weights;
    std::size_t payload_bytes{};
    std::string device_name;
};

struct CudaDenseL1ScanIndex::QueryWorkspace::Impl {
    Impl(const CudaDenseL1ScanIndex::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          query_value_count(checked_product(maximum_batch_size,
                                            owner->dimension,
                                            "CUDA dense L1 query workspace")),
          distance_count(checked_product(maximum_batch_size,
                                         owner->representative_count,
                                         "CUDA dense L1 distance workspace")),
          device_queries(owner->selected_device.index(), query_value_count),
          device_distances(owner->selected_device.index(), distance_count),
          device_results(owner->selected_device.index(), maximum_batch_size),
          payload_bytes(checked_add(
              checked_add(device_queries.bytes(), device_distances.bytes(),
                          "CUDA dense L1 workspace"),
              device_results.bytes(), "CUDA dense L1 workspace"))
    {
        if (maximum_batch_size == 0) {
            throw std::invalid_argument(
                "CUDA dense L1 workspace batch size must be positive");
        }
    }

    const CudaDenseL1ScanIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::size_t query_value_count{};
    std::size_t distance_count{};
    DeviceBuffer<float> device_queries;
    DeviceBuffer<float> device_distances;
    DeviceBuffer<std::size_t> device_results;
    std::size_t payload_bytes{};
};

bool cuda_dense_l1_scan_available() noexcept
{
    int device_count{};
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

CudaDenseL1ScanIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaDenseL1ScanIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaDenseL1ScanIndex::QueryWorkspace&
CudaDenseL1ScanIndex::QueryWorkspace::operator=(QueryWorkspace&&) noexcept =
    default;

CudaDenseL1ScanIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t
CudaDenseL1ScanIndex::QueryWorkspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t CudaDenseL1ScanIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(const DenseMatrix& representatives,
                                           int device)
    : implementation_(std::make_unique<Impl>(representatives,
                                             std::span<const float>{}, device))
{
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(
    const DenseMatrix& representatives,
    std::span<const float> coordinate_weights, int device)
    : implementation_(
          std::make_unique<Impl>(representatives, coordinate_weights, device))
{
}

CudaDenseL1ScanIndex::CudaDenseL1ScanIndex(CudaDenseL1ScanIndex&&) noexcept =
    default;

CudaDenseL1ScanIndex&
CudaDenseL1ScanIndex::operator=(CudaDenseL1ScanIndex&&) noexcept = default;

CudaDenseL1ScanIndex::~CudaDenseL1ScanIndex() = default;

CudaDenseL1ScanIndex::QueryWorkspace
CudaDenseL1ScanIndex::make_query_workspace(std::size_t maximum_batch_size) const
{
    return QueryWorkspace(std::make_unique<QueryWorkspace::Impl>(
        implementation_.get(), maximum_batch_size));
}

std::size_t CudaDenseL1ScanIndex::query(std::span<const float> query,
                                        QueryWorkspace& workspace) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, workspace);
    return output.front();
}

void CudaDenseL1ScanIndex::query_batch(std::span<const float> queries,
                                       std::size_t query_count,
                                       std::span<std::size_t> output,
                                       QueryWorkspace& workspace) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA dense L1 workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA dense L1 workspace capacity");
    }
    const std::size_t expected_query_values = checked_product(
        query_count, implementation_->dimension, "CUDA dense L1 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match index dimension");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA dense L1 output size does not match query count");
    }
    detail::validate_finite_values(queries, "queries");
    if (query_count == 0) {
        return;
    }
    if (implementation_->dimension == 0) {
        std::fill(output.begin(), output.end(), std::size_t{0});
        return;
    }

    check_cuda(cudaSetDevice(implementation_->selected_device.index()),
               "cannot select CUDA device");
    const std::size_t query_bytes = checked_bytes<float>(
        expected_query_values, "CUDA dense L1 query batch");
    if (query_bytes != 0) {
        check_cuda(cudaMemcpy(workspace.implementation_->device_queries.data(),
                              queries.data(), query_bytes,
                              cudaMemcpyHostToDevice),
                   "cannot copy dense L1 queries to CUDA device");
    }

    const std::size_t pair_count =
        checked_product(query_count, implementation_->representative_count,
                        "CUDA dense L1 query-representative grid");
    const auto maximum_grid = static_cast<std::size_t>(
        implementation_->selected_device.properties().maxGridSize[0]);
    if (pair_count > maximum_grid || query_count > maximum_grid) {
        throw std::length_error(
            "CUDA dense L1 query grid exceeds device limit");
    }

    compute_dense_l1_distances<<<static_cast<unsigned int>(pair_count),
                                 threads_per_block>>>(
        implementation_->device_representatives.data(),
        workspace.implementation_->device_queries.data(),
        implementation_->device_coordinate_weights.data(),
        implementation_->representative_count, implementation_->dimension,
        query_count, workspace.implementation_->device_distances.data());
    check_kernel_launch("cannot launch CUDA dense L1-distance kernel");

    reduce_dense_l1_argmin<<<static_cast<unsigned int>(query_count),
                             threads_per_block>>>(
        workspace.implementation_->device_distances.data(),
        implementation_->representative_count, query_count,
        workspace.implementation_->device_results.data());
    check_kernel_launch("cannot launch CUDA dense L1 argmin kernel");

    check_cuda(cudaMemcpy(output.data(),
                          workspace.implementation_->device_results.data(),
                          checked_bytes<std::size_t>(
                              query_count, "CUDA dense L1 query results"),
                          cudaMemcpyDeviceToHost),
               "cannot copy dense L1 query results from CUDA device");
}

IndexSpaceUsage CudaDenseL1ScanIndex::space_usage() const noexcept
{
    return IndexSpaceUsage{
        .index_payload_bytes = implementation_->payload_bytes,
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates = implementation_->dimension,
        .sampled_multiplicity = implementation_->dimension,
    };
}

int CudaDenseL1ScanIndex::device() const noexcept
{
    return implementation_->selected_device.index();
}

const std::string& CudaDenseL1ScanIndex::device_name() const noexcept
{
    return implementation_->device_name;
}

} // namespace ultrahigh_ann
