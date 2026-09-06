#include "ultrahigh_ann/hnsvw25/l2/importance_sampling.hpp"

#include "core/finite_values.hpp"

#include <cublas_v2.h>
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

[[nodiscard]] std::string_view cublas_status_name(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "license error";
    }
    return "unknown error";
}

void check_cublas(cublasStatus_t status, std::string_view operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::string(cublas_status_name(status)));
    }
}

class CublasHandle {
  public:
    CublasHandle()
    {
        check_cublas(cublasCreate(&handle_), "cannot create cuBLAS handle");
        try {
            check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                         "cannot select cuBLAS FP32 math mode");
        } catch (...) {
            static_cast<void>(cublasDestroy(handle_));
            handle_ = nullptr;
            throw;
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    ~CublasHandle()
    {
        if (handle_ != nullptr) {
            static_cast<void>(cublasDestroy(handle_));
        }
    }

    [[nodiscard]] cublasHandle_t get() const noexcept
    {
        return handle_;
    }

  private:
    cublasHandle_t handle_{};
};

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

[[nodiscard]] std::size_t checked_product(std::size_t left, std::size_t right,
                                          std::string_view description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::length_error(std::string(description) + " size overflows");
    }
    return left * right;
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

__device__ __forceinline__ float fused_square_add(float difference,
                                                  float accumulator)
{
    return fmaf(difference, difference, accumulator);
}

__device__ __forceinline__ float minimum_probability(float probability)
{
    return fminf(1.0F, probability);
}

__device__ __forceinline__ float maximum_probability(float left, float right)
{
    return fmaxf(left, right);
}

template <bool refine_marked_only = false>
__global__ void compute_inverse_squared_distances(
    const float* representatives, const std::uint32_t* pairs,
    std::size_t dimension, std::size_t pair_begin, std::size_t pair_end,
    float* inverse_squared_distances)
{
    const unsigned int lane = threadIdx.x & (warp_size - 1U);
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t global_warp = global_thread / warp_size;
    const std::size_t warp_count =
        (static_cast<std::size_t>(gridDim.x) * blockDim.x) / warp_size;

    for (std::size_t pair_index = pair_begin + global_warp;
         pair_index < pair_end; pair_index += warp_count) {
        if constexpr (refine_marked_only) {
            if (inverse_squared_distances[pair_index] >= 0.0F) {
                continue;
            }
        }
        std::size_t first{};
        std::size_t second{};
        unpack_pair(pairs[pair_index], first, second);
        const float* first_row = representatives + first * dimension;
        const float* second_row = representatives + second * dimension;

        float squared_distance{};
        for (std::size_t column = lane; column < dimension;
             column += warp_size) {
            const float difference = first_row[column] - second_row[column];
            squared_distance = fused_square_add(difference, squared_distance);
        }
        for (unsigned int offset = warp_size / 2U; offset != 0; offset /= 2U) {
            squared_distance +=
                __shfl_down_sync(0xffffffffU, squared_distance, offset);
        }
        if (lane == 0U) {
            inverse_squared_distances[pair_index] =
                squared_distance == 0.0F ? 0.0F : 1.0F / squared_distance;
        }
    }
}

__global__ void initialize_inverse_squared_distances_from_gram(
    const float* gram_matrix, const std::uint32_t* pairs, std::size_t row_count,
    std::size_t pair_count, float refinement_relative_threshold,
    float* inverse_squared_distances)
{
    const std::size_t global_thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t thread_count =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t pair_index = global_thread; pair_index < pair_count;
         pair_index += thread_count) {
        std::size_t first{};
        std::size_t second{};
        unpack_pair(pairs[pair_index], first, second);
        const float first_norm = gram_matrix[first + first * row_count];
        const float second_norm = gram_matrix[second + second * row_count];
        const float dot_product = gram_matrix[first + second * row_count];
        const float squared_distance =
            fmaf(-2.0F, dot_product, first_norm + second_norm);
        const float cancellation_scale =
            fabsf(first_norm) + fabsf(second_norm) + 2.0F * fabsf(dot_product);
        const float refinement_threshold =
            refinement_relative_threshold * cancellation_scale;

        // A negative marker asks the direct warp kernel to recompute pairs for
        // which the norm/dot-product identity loses too much significance.
        inverse_squared_distances[pair_index] =
            isfinite(squared_distance) &&
                    squared_distance > refinement_threshold
                ? 1.0F / squared_distance
                : -1.0F;
    }
}

__global__ void compute_partial_coordinate_maxima(
    const float* representatives, const std::uint32_t* pairs,
    const float* inverse_squared_distances, std::size_t dimension,
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
        const float inverse_distance = inverse_squared_distances[pair_index];
        if (inverse_distance == 0.0F) {
            continue;
        }
        std::size_t first{};
        std::size_t second{};
        unpack_pair(pairs[pair_index], first, second);
        const float difference = representatives[first * dimension + column] -
                                 representatives[second * dimension + column];
        const float probability =
            minimum_probability(difference * difference * inverse_distance);
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

bool l2_gpu_backend_available() noexcept
{
    int device_count{};
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

namespace {

L2GpuProbabilityResult compute_l2_importance_probabilities_gpu_impl(
    const DenseMatrix& representatives, int device,
    std::size_t requested_pair_chunks, L2GpuDistanceBackend distance_backend)
{
    const auto total_start = Clock::now();
    if (requested_pair_chunks == 0) {
        throw std::invalid_argument("GPU pair chunk count must be positive");
    }
    if (device < 0) {
        throw std::invalid_argument("CUDA device index must be nonnegative");
    }
    if (representatives.rows() > maximum_packed_row_count) {
        throw std::invalid_argument(
            "CUDA L2 probability construction supports at most 65536 rows");
    }
    if (distance_backend == L2GpuDistanceBackend::cublas &&
        (representatives.rows() >
             static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
         representatives.cols() >
             static_cast<std::size_t>(std::numeric_limits<int>::max()))) {
        throw std::invalid_argument(
            "cuBLAS L2 probability dimensions exceed the integer API limit");
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
    L2GpuProbabilityResult result{
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

    const std::size_t matrix_value_count = representatives.values().size();
    if (dimension != 0 &&
        chunk_count > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::length_error("partial-probability size overflows");
    }
    const std::size_t partial_value_count = chunk_count * dimension;
    const std::size_t gram_value_count =
        distance_backend == L2GpuDistanceBackend::cublas
            ? checked_product(row_count, row_count, "cuBLAS Gram matrix")
            : 0;

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
    required_device_bytes = checked_add(
        required_device_bytes,
        checked_bytes<float>(gram_value_count, "cuBLAS Gram matrix"),
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
    DeviceBuffer<float> device_gram_matrix(gram_value_count);

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

    if (distance_backend == L2GpuDistanceBackend::direct) {
        result.timings.inverse_distance_ms = time_cuda([&] {
            for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
                const PairRange range =
                    pair_range(chunk, total_pair_count, chunk_count);
                const unsigned int blocks = inverse_distance_block_count(
                    range.end - range.begin, properties);
                compute_inverse_squared_distances<>
                    <<<blocks, threads_per_block>>>(
                        device_representatives.data(), device_pairs.data(),
                        dimension, range.begin, range.end,
                        device_inverse_distances.data());
                check_kernel_launch(
                    "cannot launch CUDA inverse-distance kernel");
            }
        });
    } else {
        CublasHandle cublas_handle;
        result.timings.inverse_distance_ms = time_cuda([&] {
            constexpr float alpha = 1.0F;
            constexpr float beta = 0.0F;
            check_cublas(
                cublasSgemm(
                    cublas_handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(row_count), static_cast<int>(row_count),
                    static_cast<int>(dimension), &alpha,
                    device_representatives.data(), static_cast<int>(dimension),
                    device_representatives.data(), static_cast<int>(dimension),
                    &beta, device_gram_matrix.data(),
                    static_cast<int>(row_count)),
                "cannot compute cuBLAS representative Gram matrix");

            const unsigned int blocks =
                inverse_distance_block_count(total_pair_count, properties);
            constexpr float refinement_relative_threshold =
                256.0F * std::numeric_limits<float>::epsilon();
            initialize_inverse_squared_distances_from_gram<<<
                blocks, threads_per_block>>>(
                device_gram_matrix.data(), device_pairs.data(), row_count,
                total_pair_count, refinement_relative_threshold,
                device_inverse_distances.data());
            check_kernel_launch(
                "cannot launch cuBLAS inverse-distance conversion kernel");
            compute_inverse_squared_distances<true>
                <<<blocks, threads_per_block>>>(
                    device_representatives.data(), device_pairs.data(),
                    dimension, 0, total_pair_count,
                    device_inverse_distances.data());
            check_kernel_launch(
                "cannot launch cuBLAS distance-refinement kernel");
        });
    }

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

}  // namespace

L2GpuProbabilityResult compute_l2_importance_probabilities_gpu(
    const DenseMatrix& representatives, int device,
    std::size_t requested_pair_chunks, L2GpuDistanceBackend distance_backend)
{
    return compute_l2_importance_probabilities_gpu_impl(
        representatives, device, requested_pair_chunks, distance_backend);
}

}  // namespace ultrahigh_ann
