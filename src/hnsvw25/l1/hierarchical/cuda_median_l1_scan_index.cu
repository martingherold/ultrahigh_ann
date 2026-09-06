#include "hnsvw25/l1/hierarchical/cuda_median_l1_scan_index.hpp"

#include "core/finite_values.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

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

namespace ultrahigh_ann::detail {
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

template <class T>
[[nodiscard]] std::size_t checked_bytes(std::size_t count,
                                        std::string_view description)
{
    return checked_product(count, sizeof(T), description);
}

[[nodiscard]] std::size_t next_power_of_two(std::size_t value)
{
    std::size_t result = 1;
    while (result < value) {
        if (result > std::numeric_limits<std::size_t>::max() / 2) {
            throw std::length_error("CUDA median sort dimension overflows");
        }
        result *= 2;
    }
    return result;
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

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer(int device, std::size_t count)
        : device_(device),
          byte_count_(checked_bytes<T>(count, "CUDA median L1 buffer"))
    {
        if (byte_count_ == 0) {
            return;
        }
        check_cuda(cudaSetDevice(device_), "cannot select CUDA device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), byte_count_),
                   "cannot allocate CUDA median L1 buffer");
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

__device__ __forceinline__ float positive_infinity()
{
    return __int_as_float(0x7f800000);
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

__global__ void compute_median_absolute_differences(
    const float* representatives, const float* queries,
    std::size_t representative_count, std::size_t projection_dimension,
    std::size_t sorted_dimension, std::size_t query_count, float* distances,
    int* invalid_result)
{
    extern __shared__ float sorted_differences[];

    const std::size_t pair_index = blockIdx.x;
    const std::size_t pair_count = representative_count * query_count;
    if (pair_index >= pair_count) {
        return;
    }
    const std::size_t query_index = pair_index / representative_count;
    const std::size_t representative_index =
        pair_index - query_index * representative_count;
    const float* representative =
        representatives + representative_index * projection_dimension;
    const float* query = queries + query_index * projection_dimension;

    for (std::size_t index = threadIdx.x; index < sorted_dimension;
         index += blockDim.x) {
        float difference = positive_infinity();
        if (index < projection_dimension) {
            difference = fabsf(representative[index] - query[index]);
            if (!isfinite(difference)) {
                atomicExch(invalid_result, 1);
                difference = positive_infinity();
            }
        }
        sorted_differences[index] = difference;
    }
    __syncthreads();

    for (std::size_t width = 2; width <= sorted_dimension; width *= 2) {
        for (std::size_t stride = width / 2; stride != 0; stride /= 2) {
            for (std::size_t index = threadIdx.x; index < sorted_dimension;
                 index += blockDim.x) {
                const std::size_t partner = index ^ stride;
                if (partner > index) {
                    const bool ascending = (index & width) == 0;
                    const float first = sorted_differences[index];
                    const float second = sorted_differences[partner];
                    if ((first > second) == ascending) {
                        sorted_differences[index] = second;
                        sorted_differences[partner] = first;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0U) {
        const std::size_t middle = projection_dimension / 2;
        float median = sorted_differences[middle];
        if (projection_dimension % 2 == 0) {
            const float lower = sorted_differences[middle - 1];
            median = lower + (median - lower) * 0.5F;
        }
        if (!isfinite(median)) {
            atomicExch(invalid_result, 1);
        }
        distances[pair_index] = median;
    }
}

__global__ void reduce_median_l1_argmin(const float* distances,
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

}  // namespace

struct CudaMedianL1ScanIndex::Impl {
    Impl(const DenseMatrix& representatives, int requested_device)
        : selected_device(requested_device),
          representative_count(representatives.rows()),
          projection_dimension(representatives.cols()),
          sorted_dimension(next_power_of_two(projection_dimension)),
          device_representatives(selected_device.index(),
                                 representatives.values().size()),
          payload_bytes(device_representatives.bytes()),
          device_name(selected_device.properties().name)
    {
        if (representative_count == 0) {
            throw std::invalid_argument(
                "CUDA median L1 scan expects at least one representative");
        }
        if (projection_dimension == 0) {
            throw std::invalid_argument(
                "CUDA median L1 projection dimension must be positive");
        }
        const std::size_t shared_bytes = checked_bytes<float>(
            sorted_dimension, "CUDA median L1 shared workspace");
        if (shared_bytes >
            static_cast<std::size_t>(
                selected_device.properties().sharedMemPerBlock)) {
            throw std::invalid_argument(
                "CUDA median L1 projection dimension exceeds the device's "
                "per-block shared-memory capacity");
        }
        validate_finite_result(representatives.values(),
                               "projected representatives");
        check_cuda(cudaMemcpy(device_representatives.data(),
                              representatives.values().data(),
                              device_representatives.bytes(),
                              cudaMemcpyHostToDevice),
                   "cannot copy projected representatives to CUDA device");
    }

    SelectedDevice selected_device;
    std::size_t representative_count{};
    std::size_t projection_dimension{};
    std::size_t sorted_dimension{};
    DeviceBuffer<float> device_representatives;
    std::size_t payload_bytes{};
    std::string device_name;
};

struct CudaMedianL1ScanIndex::Workspace::Impl {
    Impl(const CudaMedianL1ScanIndex::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          distance_count(checked_product(maximum_batch_size,
                                         owner->representative_count,
                                         "CUDA median L1 distance workspace")),
          device_distances(owner->selected_device.index(), distance_count),
          device_results(owner->selected_device.index(), maximum_batch_size),
          device_invalid_result(owner->selected_device.index(), 1),
          payload_bytes(checked_add(
              checked_add(device_distances.bytes(), device_results.bytes(),
                          "CUDA median L1 workspace"),
              device_invalid_result.bytes(), "CUDA median L1 workspace"))
    {
        if (maximum_batch_size == 0) {
            throw std::invalid_argument(
                "CUDA median L1 workspace batch size must be positive");
        }
    }

    const CudaMedianL1ScanIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::size_t distance_count{};
    DeviceBuffer<float> device_distances;
    DeviceBuffer<std::size_t> device_results;
    DeviceBuffer<int> device_invalid_result;
    std::size_t payload_bytes{};
};

CudaMedianL1ScanIndex::Workspace::Workspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{}

CudaMedianL1ScanIndex::Workspace::Workspace(Workspace&&) noexcept = default;

CudaMedianL1ScanIndex::Workspace&
CudaMedianL1ScanIndex::Workspace::operator=(Workspace&&) noexcept = default;

CudaMedianL1ScanIndex::Workspace::~Workspace() = default;

std::size_t
CudaMedianL1ScanIndex::Workspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t CudaMedianL1ScanIndex::Workspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaMedianL1ScanIndex::CudaMedianL1ScanIndex(
    const DenseMatrix& projected_representatives, int device)
    : implementation_(std::make_unique<Impl>(projected_representatives, device))
{}

CudaMedianL1ScanIndex::CudaMedianL1ScanIndex(CudaMedianL1ScanIndex&&) noexcept =
    default;

CudaMedianL1ScanIndex&
CudaMedianL1ScanIndex::operator=(CudaMedianL1ScanIndex&&) noexcept = default;

CudaMedianL1ScanIndex::~CudaMedianL1ScanIndex() = default;

CudaMedianL1ScanIndex::Workspace
CudaMedianL1ScanIndex::make_workspace(std::size_t maximum_batch_size) const
{
    return Workspace(std::make_unique<Workspace::Impl>(implementation_.get(),
                                                       maximum_batch_size));
}

void CudaMedianL1ScanIndex::query_device_batch(const float* projected_queries,
                                               std::size_t query_count,
                                               std::span<std::size_t> output,
                                               Workspace& workspace) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA median L1 workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA median L1 workspace capacity");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA median L1 output size does not match query count");
    }
    if (query_count == 0) {
        return;
    }
    if (projected_queries == nullptr) {
        throw std::invalid_argument(
            "CUDA median L1 projected-query pointer must not be null");
    }

    check_cuda(cudaSetDevice(implementation_->selected_device.index()),
               "cannot select CUDA device");
    const std::size_t pair_count =
        checked_product(query_count, implementation_->representative_count,
                        "CUDA median L1 query-representative grid");
    const auto maximum_grid = static_cast<std::size_t>(
        implementation_->selected_device.properties().maxGridSize[0]);
    if (pair_count > maximum_grid || query_count > maximum_grid) {
        throw std::length_error(
            "CUDA median L1 query grid exceeds device limit");
    }

    check_cuda(
        cudaMemset(workspace.implementation_->device_invalid_result.data(), 0,
                   sizeof(int)),
        "cannot initialize CUDA median L1 validation");
    const std::size_t shared_bytes = checked_bytes<float>(
        implementation_->sorted_dimension, "CUDA median L1 shared workspace");
    compute_median_absolute_differences<<<static_cast<unsigned int>(pair_count),
                                          threads_per_block, shared_bytes>>>(
        implementation_->device_representatives.data(), projected_queries,
        implementation_->representative_count,
        implementation_->projection_dimension,
        implementation_->sorted_dimension, query_count,
        workspace.implementation_->device_distances.data(),
        workspace.implementation_->device_invalid_result.data());
    check_kernel_launch("cannot launch CUDA median absolute-difference kernel");

    reduce_median_l1_argmin<<<static_cast<unsigned int>(query_count),
                              threads_per_block>>>(
        workspace.implementation_->device_distances.data(),
        implementation_->representative_count, query_count,
        workspace.implementation_->device_results.data());
    check_kernel_launch("cannot launch CUDA median L1 argmin kernel");

    check_cuda(cudaMemcpy(output.data(),
                          workspace.implementation_->device_results.data(),
                          checked_bytes<std::size_t>(
                              query_count, "CUDA median L1 query results"),
                          cudaMemcpyDeviceToHost),
               "cannot copy CUDA median L1 query results from device");
    int invalid_result{};
    check_cuda(
        cudaMemcpy(&invalid_result,
                   workspace.implementation_->device_invalid_result.data(),
                   sizeof(invalid_result), cudaMemcpyDeviceToHost),
        "cannot validate CUDA median L1 distances");
    if (invalid_result != 0) {
        throw std::overflow_error("CUDA median L1 distance is not finite");
    }
}

std::size_t CudaMedianL1ScanIndex::payload_bytes() const noexcept
{
    return implementation_->payload_bytes;
}

std::size_t CudaMedianL1ScanIndex::representative_count() const noexcept
{
    return implementation_->representative_count;
}

std::size_t CudaMedianL1ScanIndex::projection_dimension() const noexcept
{
    return implementation_->projection_dimension;
}

int CudaMedianL1ScanIndex::device() const noexcept
{
    return implementation_->selected_device.index();
}

const std::string& CudaMedianL1ScanIndex::device_name() const noexcept
{
    return implementation_->device_name;
}

}  // namespace ultrahigh_ann::detail
