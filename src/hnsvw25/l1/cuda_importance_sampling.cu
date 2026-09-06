#include "ultrahigh_ann/hnsvw25/l1/importance_sampling.hpp"

#include "core/finite_values.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ultrahigh_ann {
namespace {

using Clock = std::chrono::steady_clock;

constexpr unsigned int threads_per_block = 256;
constexpr unsigned int warp_size = 32;
constexpr unsigned int warps_per_block = threads_per_block / warp_size;
constexpr std::size_t maximum_packed_row_count = 65'536;

[[nodiscard]] double elapsed_ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

void check_cuda(cudaError_t status, std::string_view operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

class CudaEvent {
  public:
    CudaEvent()
    {
        check_cuda(cudaEventCreate(&event_), "cannot create CUDA event");
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    ~CudaEvent()
    {
        if (event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(event_));
        }
    }

    [[nodiscard]] cudaEvent_t get() const noexcept
    {
        return event_;
    }

  private:
    cudaEvent_t event_{};
};

template <class Function> [[nodiscard]] double time_cuda(Function&& function)
{
    CudaEvent start;
    CudaEvent stop;
    check_cuda(cudaEventRecord(start.get()), "cannot record CUDA start event");
    std::forward<Function>(function)();
    check_cuda(cudaEventRecord(stop.get()), "cannot record CUDA stop event");
    check_cuda(cudaEventSynchronize(stop.get()),
               "CUDA operation did not complete");
    float milliseconds{};
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cannot measure CUDA elapsed time");
    return static_cast<double>(milliseconds);
}

template <class T> class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count)
    {
        if (count_ == 0) {
            return;
        }
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::length_error("CUDA allocation size overflows");
        }
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cannot allocate CUDA device memory");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
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
        return count_ * sizeof(T);
    }

  private:
    T* data_{};
    std::size_t count_{};
};

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
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return count * sizeof(T);
}

[[nodiscard]] std::size_t pair_count(std::size_t row_count)
{
    if (row_count < 2) {
        return 0;
    }
    return row_count * (row_count - 1) / 2;
}

[[nodiscard]] std::vector<std::uint32_t>
make_packed_pairs(std::size_t row_count)
{
    std::vector<std::uint32_t> pairs(pair_count(row_count));
    std::size_t pair_index = 0;
    for (std::size_t first = 0; first + 1 < row_count; ++first) {
        for (std::size_t second = first + 1; second < row_count; ++second) {
            pairs[pair_index++] = (static_cast<std::uint32_t>(first) << 16U) |
                                  static_cast<std::uint32_t>(second);
        }
    }
    return pairs;
}

struct PairRange {
    std::size_t begin{};
    std::size_t end{};
};

[[nodiscard]] PairRange pair_range(std::size_t index, std::size_t total,
                                   std::size_t chunk_count)
{
    const std::size_t base = total / chunk_count;
    const std::size_t remainder = total % chunk_count;
    const std::size_t begin = index * base + std::min(index, remainder);
    return PairRange{
        .begin = begin,
        .end = begin + base + static_cast<std::size_t>(index < remainder),
    };
}

__device__ __forceinline__ void
unpack_pair(std::uint32_t packed_pair, std::size_t& first, std::size_t& second)
{
    first = static_cast<std::size_t>(packed_pair >> 16U);
    second = static_cast<std::size_t>(packed_pair & 0xffffU);
}

__device__ __forceinline__ float minimum_probability(float probability)
{
    return fminf(1.0F, probability);
}

__device__ __forceinline__ float maximum_probability(float left, float right)
{
    return fmaxf(left, right);
}

__global__ void compute_inverse_l1_distances(const float* representatives,
                                             const std::uint32_t* pairs,
                                             std::size_t dimension,
                                             std::size_t pair_begin,
                                             std::size_t pair_end,
                                             float* inverse_distances)
{
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t global_warp = global_thread / warp_size;
    const std::size_t warp_count =
        (static_cast<std::size_t>(gridDim.x) * blockDim.x) / warp_size;

    for (std::size_t pair_index = pair_begin + global_warp;
         pair_index < pair_end; pair_index += warp_count) {
        std::size_t first{};
        std::size_t second{};
        unpack_pair(pairs[pair_index], first, second);
        const float* first_row = representatives + first * dimension;
        const float* second_row = representatives + second * dimension;

        float distance{};
        for (std::size_t column = lane; column < dimension;
             column += warp_size) {
            distance += fabsf(first_row[column] - second_row[column]);
        }
        for (unsigned int offset = warp_size / 2U; offset != 0; offset /= 2U) {
            distance += __shfl_down_sync(0xffffffffU, distance, offset);
        }
        if (lane == 0U) {
            inverse_distances[pair_index] =
                distance == 0.0F ? 0.0F : 1.0F / distance;
        }
    }
}

__global__ void compute_partial_coordinate_maxima(
    const float* representatives, const std::uint32_t* pairs,
    const float* inverse_distances, std::size_t dimension,
    std::size_t pair_begin, std::size_t pair_end, float* partial_probabilities)
{
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= dimension) {
        return;
    }

    float maximum_value{};
    for (std::size_t pair_index = pair_begin; pair_index < pair_end;
         ++pair_index) {
        const float inverse_distance = inverse_distances[pair_index];
        if (inverse_distance == 0.0F) {
            continue;
        }
        std::size_t first{};
        std::size_t second{};
        unpack_pair(pairs[pair_index], first, second);
        const float difference =
            fabsf(representatives[first * dimension + column] -
                  representatives[second * dimension + column]);
        const float probability =
            minimum_probability(difference * inverse_distance);
        maximum_value = maximum_probability(maximum_value, probability);
    }
    partial_probabilities[column] = maximum_value;
}

__global__ void reduce_coordinate_maxima(const float* partial_probabilities,
                                         std::size_t dimension,
                                         std::size_t chunk_count,
                                         float* probabilities)
{
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= dimension) {
        return;
    }
    float maximum_value{};
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        maximum_value = maximum_probability(
            maximum_value, partial_probabilities[chunk * dimension + column]);
    }
    probabilities[column] = maximum_value;
}

void check_kernel_launch(std::string_view description)
{
    check_cuda(cudaGetLastError(), description);
}

[[nodiscard]] unsigned int
inverse_distance_block_count(std::size_t chunk_pair_count,
                             const cudaDeviceProp& properties)
{
    const std::size_t blocks_for_pairs =
        (chunk_pair_count + warps_per_block - 1) / warps_per_block;
    const std::size_t occupancy_blocks =
        static_cast<std::size_t>(properties.multiProcessorCount) * 32;
    return static_cast<unsigned int>(
        std::max<std::size_t>(1, std::min(blocks_for_pairs, occupancy_blocks)));
}

[[nodiscard]] unsigned int
coordinate_block_count(std::size_t dimension, const cudaDeviceProp& properties)
{
    const std::size_t blocks =
        (dimension + threads_per_block - 1) / threads_per_block;
    if (blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error("coordinate dimension exceeds CUDA grid limit");
    }
    return static_cast<unsigned int>(blocks);
}

}  // namespace

bool l1_gpu_backend_available() noexcept
{
    int device_count{};
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

L1GpuProbabilityResult compute_l1_importance_probabilities_gpu(
    const DenseMatrix& representatives, int device,
    std::size_t requested_pair_chunks, L1GpuDistanceBackend distance_backend)
{
    const auto total_start = Clock::now();
    if (distance_backend != L1GpuDistanceBackend::direct) {
        throw std::invalid_argument("unknown L1 GPU distance backend");
    }
    if (requested_pair_chunks == 0) {
        throw std::invalid_argument("GPU pair chunk count must be positive");
    }
    if (device < 0) {
        throw std::invalid_argument("CUDA device index must be nonnegative");
    }
    if (representatives.rows() > maximum_packed_row_count) {
        throw std::invalid_argument(
            "CUDA L1 probability construction supports at most 65536 rows");
    }
    detail::validate_finite_values(representatives.values(), "representatives");

    int device_count{};
    check_cuda(cudaGetDeviceCount(&device_count),
               "cannot enumerate CUDA devices");
    if (device >= device_count) {
        throw std::invalid_argument("CUDA device index is out of range");
    }
    check_cuda(cudaSetDevice(device), "cannot select CUDA device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
               "cannot query CUDA device properties");

    const std::size_t row_count = representatives.rows();
    const std::size_t dimension = representatives.cols();
    const std::size_t total_pair_count = pair_count(row_count);
    L1GpuProbabilityResult result{
        .probabilities = std::vector<double>(dimension, 0.0),
        .device_name = properties.name,
        .device = device,
        .distance_backend = distance_backend,
        .pair_count = total_pair_count,
        .pair_chunks = 0,
        .device_working_set_bytes = 0,
        .timings = {},
    };
    if (total_pair_count == 0 || dimension == 0) {
        result.pair_chunks = total_pair_count == 0 ? 0 : 1;
        result.timings.total_ms = elapsed_ms(total_start);
        return result;
    }

    const std::size_t chunk_count =
        std::min(requested_pair_chunks, total_pair_count);
    result.pair_chunks = chunk_count;
    std::vector<std::uint32_t> pairs = make_packed_pairs(row_count);

    if (chunk_count > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::length_error("partial-probability size overflows");
    }
    const std::size_t matrix_value_count = representatives.values().size();
    const std::size_t partial_value_count = chunk_count * dimension;

    std::size_t required_device_bytes =
        checked_bytes<float>(matrix_value_count, "representative matrix");
    required_device_bytes = checked_add(
        required_device_bytes,
        checked_bytes<std::uint32_t>(pairs.size(), "packed pair list"),
        "CUDA working set");
    required_device_bytes =
        checked_add(required_device_bytes,
                    checked_bytes<float>(total_pair_count, "inverse distances"),
                    "CUDA working set");
    required_device_bytes = checked_add(
        required_device_bytes,
        checked_bytes<float>(partial_value_count, "partial probabilities"),
        "CUDA working set");
    required_device_bytes = checked_add(
        required_device_bytes, checked_bytes<float>(dimension, "probabilities"),
        "CUDA working set");
    result.device_working_set_bytes = required_device_bytes;

    std::size_t free_device_bytes{};
    std::size_t total_device_bytes{};
    check_cuda(cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
               "cannot query CUDA memory");
    static_cast<void>(total_device_bytes);
    if (required_device_bytes > free_device_bytes) {
        throw std::runtime_error(
            "CUDA device has insufficient free memory: requires " +
            std::to_string(required_device_bytes) + " bytes, has " +
            std::to_string(free_device_bytes) + " bytes");
    }

    DeviceBuffer<float> device_representatives(matrix_value_count);
    DeviceBuffer<std::uint32_t> device_pairs(pairs.size());
    DeviceBuffer<float> device_inverse_distances(total_pair_count);
    DeviceBuffer<float> device_partial_probabilities(partial_value_count);
    DeviceBuffer<float> device_probabilities(dimension);

    result.timings.host_to_device_ms = time_cuda([&] {
        check_cuda(cudaMemcpy(device_representatives.data(),
                              representatives.values().data(),
                              device_representatives.bytes(),
                              cudaMemcpyHostToDevice),
                   "cannot copy representatives to CUDA device");
        check_cuda(cudaMemcpy(device_pairs.data(), pairs.data(),
                              device_pairs.bytes(), cudaMemcpyHostToDevice),
                   "cannot copy pair list to CUDA device");
    });

    result.timings.inverse_distance_ms = time_cuda([&] {
        for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const PairRange range =
                pair_range(chunk, total_pair_count, chunk_count);
            const unsigned int blocks = inverse_distance_block_count(
                range.end - range.begin, properties);
            compute_inverse_l1_distances<<<blocks, threads_per_block>>>(
                device_representatives.data(), device_pairs.data(), dimension,
                range.begin, range.end, device_inverse_distances.data());
            check_kernel_launch("cannot launch CUDA inverse-distance kernel");
        }
    });

    const unsigned int coordinate_blocks =
        coordinate_block_count(dimension, properties);
    result.timings.coordinate_maximum_ms = time_cuda([&] {
        for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const PairRange range =
                pair_range(chunk, total_pair_count, chunk_count);
            compute_partial_coordinate_maxima<<<coordinate_blocks,
                                                threads_per_block>>>(
                device_representatives.data(), device_pairs.data(),
                device_inverse_distances.data(), dimension, range.begin,
                range.end,
                device_partial_probabilities.data() + chunk * dimension);
            check_kernel_launch("cannot launch CUDA coordinate-maximum kernel");
        }
        reduce_coordinate_maxima<<<coordinate_blocks, threads_per_block>>>(
            device_partial_probabilities.data(), dimension, chunk_count,
            device_probabilities.data());
        check_kernel_launch("cannot launch CUDA maximum-reduction kernel");
    });

    std::vector<float> fp32_probabilities(dimension);
    result.timings.device_to_host_ms = time_cuda([&] {
        check_cuda(
            cudaMemcpy(fp32_probabilities.data(), device_probabilities.data(),
                       device_probabilities.bytes(), cudaMemcpyDeviceToHost),
            "cannot copy probabilities from CUDA device");
    });
    std::transform(fp32_probabilities.begin(), fp32_probabilities.end(),
                   result.probabilities.begin(), [](float probability) {
                       return static_cast<double>(probability);
                   });

    for (const double probability : result.probabilities) {
        if (!std::isfinite(probability) || probability < 0.0 ||
            probability > 1.0) {
            throw std::runtime_error(
                "CUDA probability computation produced an invalid value");
        }
    }
    result.timings.total_ms = elapsed_ms(total_start);
    return result;
}

}  // namespace ultrahigh_ann
