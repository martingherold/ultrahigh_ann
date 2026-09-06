#include "core/cuda_float_projection.hpp"

#include "core/finite_values.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

class CublasHandle {
  public:
    explicit CublasHandle(int device) : device_(device)
    {
        check_cuda(cudaSetDevice(device_), "cannot select CUDA device");
        check_cublas(cublasCreate(&handle_), "cannot create cuBLAS handle");
        try {
            check_cublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH),
                         "cannot select cuBLAS math mode");
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
            static_cast<void>(cudaSetDevice(device_));
            static_cast<void>(cublasDestroy(handle_));
        }
    }

    [[nodiscard]] cublasHandle_t get() const noexcept
    {
        return handle_;
    }

  private:
    int device_{};
    cublasHandle_t handle_{};
};

template <class T> class DeviceBuffer {
  public:
    DeviceBuffer(int device, std::size_t count)
        : device_(device),
          byte_count_(checked_bytes<T>(count, "CUDA projection buffer"))
    {
        if (byte_count_ == 0) {
            return;
        }
        check_cuda(cudaSetDevice(device_), "cannot select CUDA device");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), byte_count_),
                   "cannot allocate CUDA projection buffer");
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

__global__ void project_queries_direct(
    const float* sampled_queries, const float* projection_matrix,
    std::size_t sampled_dimension, std::size_t projection_dimension,
    std::size_t query_count, float* projected_queries, int* invalid_result)
{
    const std::size_t output_index = blockIdx.x;
    const std::size_t output_count = query_count * projection_dimension;
    if (output_index >= output_count) {
        return;
    }
    const std::size_t query_index = output_index / projection_dimension;
    const std::size_t projection_index =
        output_index - query_index * projection_dimension;
    const float* query = sampled_queries + query_index * sampled_dimension;
    const float* projection =
        projection_matrix + projection_index * sampled_dimension;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < sampled_dimension;
         column += blockDim.x) {
        sum = fmaf(query[column], projection[column], sum);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0U) {
        if (!isfinite(sum)) {
            atomicExch(invalid_result, 1);
        }
        projected_queries[output_index] = sum;
    }
}

__global__ void validate_float_results(const float* input, std::size_t count,
                                       int* invalid_result)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && !isfinite(input[index])) {
        atomicExch(invalid_result, 1);
    }
}

void check_kernel_launch(std::string_view description)
{
    check_cuda(cudaGetLastError(), description);
}

[[nodiscard]] unsigned int
one_dimensional_grid(std::size_t value_count, const cudaDeviceProp& properties,
                     std::string_view description)
{
    const std::size_t block_count =
        (value_count + threads_per_block - 1U) / threads_per_block;
    if (block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::length_error(std::string(description) +
                                " grid exceeds device limit");
    }
    return static_cast<unsigned int>(block_count);
}

}  // namespace

struct CudaFloatProjection::Impl {
    Impl(const DenseMatrix& projection_matrix, int requested_device)
        : selected_device(requested_device),
          cublas_handle(selected_device.index()),
          sampled_dimension(projection_matrix.cols()),
          projection_dimension(projection_matrix.rows()),
          device_projection_matrix(selected_device.index(),
                                   projection_matrix.values().size()),
          payload_bytes(device_projection_matrix.bytes())
    {
        if (projection_dimension == 0) {
            throw std::invalid_argument(
                "CUDA projection dimension must be positive");
        }
        if (sampled_dimension > static_cast<std::size_t>(INT_MAX) ||
            projection_dimension > static_cast<std::size_t>(INT_MAX)) {
            throw std::length_error(
                "CUDA projection matrix dimensions exceed cuBLAS integer "
                "limits");
        }
        validate_finite_result(projection_matrix.values(), "projection matrix");
        if (device_projection_matrix.bytes() != 0) {
            check_cuda(cudaMemcpy(device_projection_matrix.data(),
                                  projection_matrix.values().data(),
                                  device_projection_matrix.bytes(),
                                  cudaMemcpyHostToDevice),
                       "cannot copy projection matrix to CUDA device");
        }
    }

    SelectedDevice selected_device;
    CublasHandle cublas_handle;
    std::size_t sampled_dimension{};
    std::size_t projection_dimension{};
    DeviceBuffer<float> device_projection_matrix;
    std::size_t payload_bytes{};
};

struct CudaFloatProjection::Workspace::Impl {
    Impl(const CudaFloatProjection::Impl* owner_value,
         std::size_t maximum_batch_size_value)
        : owner(owner_value), maximum_batch_size(maximum_batch_size_value),
          sampled_value_count(
              checked_product(maximum_batch_size, owner->sampled_dimension,
                              "CUDA sampled-query projection workspace")),
          projected_value_count(
              checked_product(maximum_batch_size, owner->projection_dimension,
                              "CUDA projected-query workspace")),
          device_sampled_values(owner->selected_device.index(),
                                sampled_value_count),
          device_projected_values(owner->selected_device.index(),
                                  projected_value_count),
          device_invalid_result(owner->selected_device.index(), 1),
          payload_bytes(checked_add(checked_add(device_sampled_values.bytes(),
                                                device_projected_values.bytes(),
                                                "CUDA projection workspace"),
                                    device_invalid_result.bytes(),
                                    "CUDA projection workspace"))
    {
        if (maximum_batch_size == 0) {
            throw std::invalid_argument(
                "CUDA projection workspace batch size must be positive");
        }
        if (maximum_batch_size > static_cast<std::size_t>(INT_MAX)) {
            throw std::length_error(
                "CUDA projection batch size exceeds cuBLAS integer limits");
        }
    }

    const CudaFloatProjection::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::size_t sampled_value_count{};
    std::size_t projected_value_count{};
    DeviceBuffer<float> device_sampled_values;
    DeviceBuffer<float> device_projected_values;
    DeviceBuffer<int> device_invalid_result;
    std::size_t payload_bytes{};
};

CudaFloatProjection::Workspace::Workspace(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{}

CudaFloatProjection::Workspace::Workspace(Workspace&&) noexcept = default;

CudaFloatProjection::Workspace&
CudaFloatProjection::Workspace::operator=(Workspace&&) noexcept = default;

CudaFloatProjection::Workspace::~Workspace() = default;

std::size_t CudaFloatProjection::Workspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t CudaFloatProjection::Workspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaFloatProjection::CudaFloatProjection(const DenseMatrix& projection_matrix,
                                         int device)
    : implementation_(std::make_unique<Impl>(projection_matrix, device))
{}

CudaFloatProjection::CudaFloatProjection(CudaFloatProjection&&) noexcept =
    default;

CudaFloatProjection&
CudaFloatProjection::operator=(CudaFloatProjection&&) noexcept = default;

CudaFloatProjection::~CudaFloatProjection() = default;

CudaFloatProjection::Workspace
CudaFloatProjection::make_workspace(std::size_t maximum_batch_size) const
{
    return Workspace(std::make_unique<Workspace::Impl>(implementation_.get(),
                                                       maximum_batch_size));
}

const float* CudaFloatProjection::project_batch_float(
    std::span<const float> sampled_queries, std::size_t query_count,
    Workspace& workspace, CudaFloatProjectionStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA projection workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA projection workspace capacity");
    }
    const std::size_t sampled_value_count =
        checked_product(query_count, implementation_->sampled_dimension,
                        "CUDA sampled-query projection batch");
    if (sampled_queries.size() != sampled_value_count) {
        throw std::invalid_argument(
            "sampled query batch dimensions do not match projection matrix");
    }
    validate_finite_values(sampled_queries, "sampled queries");
    if (query_count == 0) {
        return workspace.implementation_->device_projected_values.data();
    }

    check_cuda(cudaSetDevice(implementation_->selected_device.index()),
               "cannot select CUDA device");
    const std::size_t projected_value_count =
        checked_product(query_count, implementation_->projection_dimension,
                        "CUDA projected-query batch");
    check_cuda(
        cudaMemset(workspace.implementation_->device_invalid_result.data(), 0,
                   sizeof(int)),
        "cannot initialize CUDA projection validation");
    if (implementation_->sampled_dimension == 0) {
        check_cuda(
            cudaMemset(
                workspace.implementation_->device_projected_values.data(), 0,
                checked_bytes<float>(projected_value_count,
                                     "CUDA projected-query batch")),
            "cannot initialize CUDA projected queries");
        return workspace.implementation_->device_projected_values.data();
    }

    check_cuda(
        cudaMemcpy(workspace.implementation_->device_sampled_values.data(),
                   sampled_queries.data(),
                   checked_bytes<float>(sampled_value_count,
                                        "CUDA sampled-query batch"),
                   cudaMemcpyHostToDevice),
        "cannot copy sampled queries to CUDA device");
    if (strategy == CudaFloatProjectionStrategy::direct) {
        const auto maximum_grid = static_cast<std::size_t>(
            implementation_->selected_device.properties().maxGridSize[0]);
        if (projected_value_count > maximum_grid) {
            throw std::length_error(
                "CUDA direct-projection grid exceeds device limit");
        }
        project_queries_direct<<<static_cast<unsigned int>(
                                     projected_value_count),
                                 threads_per_block>>>(
            workspace.implementation_->device_sampled_values.data(),
            implementation_->device_projection_matrix.data(),
            implementation_->sampled_dimension,
            implementation_->projection_dimension, query_count,
            workspace.implementation_->device_projected_values.data(),
            workspace.implementation_->device_invalid_result.data());
        check_kernel_launch("cannot launch CUDA direct matrix projection");
    } else if (strategy == CudaFloatProjectionStrategy::gemm) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(
            cublasSgemm(
                implementation_->cublas_handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(implementation_->projection_dimension),
                static_cast<int>(query_count),
                static_cast<int>(implementation_->sampled_dimension), &alpha,
                implementation_->device_projection_matrix.data(),
                static_cast<int>(implementation_->sampled_dimension),
                workspace.implementation_->device_sampled_values.data(),
                static_cast<int>(implementation_->sampled_dimension), &beta,
                workspace.implementation_->device_projected_values.data(),
                static_cast<int>(implementation_->projection_dimension)),
            "cannot compute CUDA FP32 GEMM matrix projection");

        validate_float_results<<<
            one_dimensional_grid(projected_value_count,
                                 implementation_->selected_device.properties(),
                                 "CUDA projected-query validation"),
            threads_per_block>>>(
            workspace.implementation_->device_projected_values.data(),
            projected_value_count,
            workspace.implementation_->device_invalid_result.data());
        check_kernel_launch("cannot launch CUDA projection validation");
    } else {
        throw std::invalid_argument("unknown CUDA matrix projection strategy");
    }

    int invalid_result{};
    check_cuda(
        cudaMemcpy(&invalid_result,
                   workspace.implementation_->device_invalid_result.data(),
                   sizeof(invalid_result), cudaMemcpyDeviceToHost),
        "cannot validate CUDA projected queries");
    if (invalid_result != 0) {
        throw std::overflow_error("CUDA projected query is not finite");
    }
    return workspace.implementation_->device_projected_values.data();
}

std::size_t CudaFloatProjection::sampled_dimension() const noexcept
{
    return implementation_->sampled_dimension;
}

std::size_t CudaFloatProjection::projection_dimension() const noexcept
{
    return implementation_->projection_dimension;
}

std::size_t CudaFloatProjection::payload_bytes() const noexcept
{
    return implementation_->payload_bytes;
}

int CudaFloatProjection::device() const noexcept
{
    return implementation_->selected_device.index();
}

}  // namespace ultrahigh_ann::detail
