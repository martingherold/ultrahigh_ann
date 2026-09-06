#include "ultrahigh_ann/core/cuda_dense_l2_scan_index.hpp"

#include "core/finite_values.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <climits>
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

    [[nodiscard]] int index() const noexcept { return device_; }

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
            static_cast<void>(cudaSetDevice(device_));
            static_cast<void>(cublasDestroy(handle_));
        }
    }

    [[nodiscard]] cublasHandle_t get() const noexcept { return handle_; }

  private:
    int device_{};
    cublasHandle_t handle_{};
};

template <class T> class DeviceBuffer {
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

    [[nodiscard]] T* data() noexcept { return data_; }

    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t bytes() const noexcept { return byte_count_; }

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

__global__ void compute_row_squared_norms(const float* values,
                                          std::size_t row_count,
                                          std::size_t dimension,
                                          float* squared_norms)
{
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count) {
        return;
    }
    const float* row = values + row_index * dimension;
    float squared_norm = 0.0F;
    for (std::size_t column = threadIdx.x; column < dimension;
         column += blockDim.x) {
        squared_norm = fmaf(row[column], row[column], squared_norm);
    }
    squared_norm = block_sum(squared_norm);
    if (threadIdx.x == 0U) {
        squared_norms[row_index] = squared_norm;
    }
}

__global__ void compute_dense_l2_distances(const float* representatives,
                                           const float* queries,
                                           std::size_t representative_count,
                                           std::size_t dimension,
                                           std::size_t query_count,
                                           float* distances)
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

    float squared_distance = 0.0F;
    for (std::size_t column = threadIdx.x; column < dimension;
         column += blockDim.x) {
        const float difference = representative[column] - query[column];
        squared_distance = fmaf(difference, difference, squared_distance);
    }
    squared_distance = block_sum(squared_distance);
    if (threadIdx.x == 0U) {
        distances[pair_index] = squared_distance;
    }
}

__global__ void reduce_dense_l2_argmin(const float* distances,
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

__global__ void reduce_gemm_l2_argmin(const float* dot_products,
                                      const float* representative_squared_norms,
                                      const float* query_squared_norms,
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
    const float* query_dot_products =
        dot_products + query_index * representative_count;
    const float query_squared_norm = query_squared_norms[query_index];
    for (std::size_t representative_index = threadIdx.x;
         representative_index < representative_count;
         representative_index += blockDim.x) {
        const float distance =
            representative_squared_norms[representative_index] +
            query_squared_norm -
            2.0F * query_dot_products[representative_index];
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

struct CudaDenseL2ScanIndex::Impl {
    Impl(const DenseMatrix& representatives, int requested_device)
        : selected_device(requested_device),
          cublas_handle(selected_device.index()),
          representative_count(representatives.rows()),
          dimension(representatives.cols()),
          representative_value_count(representatives.values().size()),
          device_representatives(selected_device.index(),
                                 representative_value_count),
          device_representative_squared_norms(selected_device.index(),
                                              representative_count),
          payload_bytes(checked_add(device_representatives.bytes(),
                                    device_representative_squared_norms.bytes(),
                                    "CUDA dense L2 index")),
          device_name(selected_device.properties().name)
    {
        if (representative_count == 0) {
            throw std::invalid_argument(
                "CudaDenseL2ScanIndex expects at least one representative");
        }
        if (representative_count > static_cast<std::size_t>(INT_MAX) ||
            dimension > static_cast<std::size_t>(INT_MAX)) {
            throw std::length_error(
                "CUDA dense L2 matrix dimensions exceed cuBLAS integer limits");
        }
        detail::validate_finite_values(representatives.values(),
                                       "representatives");
        if (device_representatives.bytes() != 0) {
            check_cuda(cudaMemcpy(device_representatives.data(),
                                  representatives.values().data(),
                                  device_representatives.bytes(),
                                  cudaMemcpyHostToDevice),
                       "cannot copy representatives to CUDA device");
        }
        const auto maximum_grid = static_cast<std::size_t>(
            selected_device.properties().maxGridSize[0]);
        if (representative_count > maximum_grid) {
            throw std::length_error(
                "CUDA dense L2 representative grid exceeds device limit");
        }
        if (dimension == 0) {
            check_cuda(cudaMemset(device_representative_squared_norms.data(), 0,
                                  device_representative_squared_norms.bytes()),
                       "cannot initialize CUDA representative norms");
        } else {
            compute_row_squared_norms<<<static_cast<unsigned int>(
                                            representative_count),
                                        threads_per_block>>>(
                device_representatives.data(), representative_count, dimension,
                device_representative_squared_norms.data());
            check_kernel_launch(
                "cannot launch CUDA representative-norm kernel");
            check_cuda(cudaDeviceSynchronize(),
                       "CUDA representative-norm kernel failed");
        }
    }

    SelectedDevice selected_device;
    CublasHandle cublas_handle;
    std::size_t representative_count{};
    std::size_t dimension{};
    std::size_t representative_value_count{};
    DeviceBuffer<float> device_representatives;
    DeviceBuffer<float> device_representative_squared_norms;
    std::size_t payload_bytes{};
    std::string device_name;
};

struct CudaDenseL2ScanIndex::QueryWorkspace::Impl {
    Impl(const CudaDenseL2ScanIndex::Impl* owner,
         std::size_t maximum_batch_size)
        : owner(owner), maximum_batch_size(maximum_batch_size),
          query_value_count(checked_product(maximum_batch_size,
                                            owner->dimension,
                                            "CUDA dense L2 query workspace")),
          distance_count(checked_product(maximum_batch_size,
                                         owner->representative_count,
                                         "CUDA dense L2 distance workspace")),
          device_queries(owner->selected_device.index(), query_value_count),
          device_distances(owner->selected_device.index(), distance_count),
          device_query_squared_norms(owner->selected_device.index(),
                                     maximum_batch_size),
          device_results(owner->selected_device.index(), maximum_batch_size),
          payload_bytes(checked_add(
              checked_add(
                  checked_bytes<float>(query_value_count,
                                       "CUDA dense L2 query workspace"),
                  checked_bytes<float>(distance_count,
                                       "CUDA dense L2 distance workspace"),
                  "CUDA dense L2 workspace"),
              checked_add(
                  checked_bytes<float>(maximum_batch_size,
                                       "CUDA dense L2 query-norm workspace"),
                  checked_bytes<std::size_t>(maximum_batch_size,
                                             "CUDA dense L2 result workspace"),
                  "CUDA dense L2 workspace"),
              "CUDA dense L2 workspace"))
    {
        if (maximum_batch_size == 0) {
            throw std::invalid_argument(
                "CUDA dense L2 workspace batch size must be positive");
        }
        if (maximum_batch_size > static_cast<std::size_t>(INT_MAX)) {
            throw std::length_error(
                "CUDA dense L2 batch size exceeds cuBLAS integer limits");
        }
    }

    const CudaDenseL2ScanIndex::Impl* owner{};
    std::size_t maximum_batch_size{};
    std::size_t query_value_count{};
    std::size_t distance_count{};
    DeviceBuffer<float> device_queries;
    DeviceBuffer<float> device_distances;
    DeviceBuffer<float> device_query_squared_norms;
    DeviceBuffer<std::size_t> device_results;
    std::size_t payload_bytes{};
};

bool cuda_dense_l2_scan_available() noexcept
{
    int device_count{};
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

CudaDenseL2ScanIndex::QueryWorkspace::QueryWorkspace(
    std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation))
{
}

CudaDenseL2ScanIndex::QueryWorkspace::QueryWorkspace(
    QueryWorkspace&&) noexcept = default;

CudaDenseL2ScanIndex::QueryWorkspace&
CudaDenseL2ScanIndex::QueryWorkspace::operator=(QueryWorkspace&&) noexcept =
    default;

CudaDenseL2ScanIndex::QueryWorkspace::~QueryWorkspace() = default;

std::size_t
CudaDenseL2ScanIndex::QueryWorkspace::maximum_batch_size() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->maximum_batch_size;
}

std::size_t CudaDenseL2ScanIndex::QueryWorkspace::payload_bytes() const noexcept
{
    return implementation_ == nullptr ? 0 : implementation_->payload_bytes;
}

CudaDenseL2ScanIndex::CudaDenseL2ScanIndex(const DenseMatrix& representatives,
                                           int device)
    : implementation_(std::make_unique<Impl>(representatives, device))
{
}

CudaDenseL2ScanIndex::CudaDenseL2ScanIndex(CudaDenseL2ScanIndex&&) noexcept =
    default;

CudaDenseL2ScanIndex&
CudaDenseL2ScanIndex::operator=(CudaDenseL2ScanIndex&&) noexcept = default;

CudaDenseL2ScanIndex::~CudaDenseL2ScanIndex() = default;

CudaDenseL2ScanIndex::QueryWorkspace
CudaDenseL2ScanIndex::make_query_workspace(std::size_t maximum_batch_size) const
{
    return QueryWorkspace(std::make_unique<QueryWorkspace::Impl>(
        implementation_.get(), maximum_batch_size));
}

std::size_t CudaDenseL2ScanIndex::query(std::span<const float> query,
                                        QueryWorkspace& workspace) const
{
    std::array<std::size_t, 1> output{};
    query_batch(query, 1, output, workspace);
    return output.front();
}

void CudaDenseL2ScanIndex::query_batch(std::span<const float> queries,
                                       std::size_t query_count,
                                       std::span<std::size_t> output,
                                       QueryWorkspace& workspace,
                                       CudaDenseL2QueryStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA dense L2 workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA dense L2 workspace capacity");
    }
    const std::size_t expected_query_values = checked_product(
        query_count, implementation_->dimension, "CUDA dense L2 query batch");
    if (queries.size() != expected_query_values) {
        throw std::invalid_argument(
            "query batch dimensions do not match index dimension");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA dense L2 output size does not match query count");
    }
    detail::validate_finite_values(queries, "queries");
    if (query_count == 0) {
        return;
    }

    check_cuda(cudaSetDevice(implementation_->selected_device.index()),
               "cannot select CUDA device");
    const std::size_t query_bytes = checked_bytes<float>(
        expected_query_values, "CUDA dense L2 query batch");
    if (query_bytes != 0) {
        check_cuda(cudaMemcpy(workspace.implementation_->device_queries.data(),
                              queries.data(), query_bytes,
                              cudaMemcpyHostToDevice),
                   "cannot copy dense L2 queries to CUDA device");
    }

    query_device_batch(workspace.implementation_->device_queries.data(),
                       query_count, output, workspace, strategy);
}

void CudaDenseL2ScanIndex::query_device_batch(
    const float* device_queries,
    std::size_t query_count,
    std::span<std::size_t> output,
    QueryWorkspace& workspace,
    CudaDenseL2QueryStrategy strategy) const
{
    if (workspace.implementation_ == nullptr ||
        workspace.implementation_->owner != implementation_.get()) {
        throw std::invalid_argument(
            "CUDA dense L2 workspace belongs to a different index");
    }
    if (query_count > workspace.implementation_->maximum_batch_size) {
        throw std::invalid_argument(
            "query count exceeds CUDA dense L2 workspace capacity");
    }
    if (output.size() != query_count) {
        throw std::invalid_argument(
            "CUDA dense L2 output size does not match query count");
    }
    if (query_count == 0) {
        return;
    }
    if (implementation_->dimension != 0 && device_queries == nullptr) {
        throw std::invalid_argument(
            "CUDA dense L2 device query pointer must not be null");
    }

    check_cuda(cudaSetDevice(implementation_->selected_device.index()),
               "cannot select CUDA device");

    const std::size_t pair_count =
        checked_product(query_count, implementation_->representative_count,
                        "CUDA dense L2 query-representative grid");
    const auto maximum_grid = static_cast<std::size_t>(
        implementation_->selected_device.properties().maxGridSize[0]);
    if (pair_count > maximum_grid || query_count > maximum_grid) {
        throw std::length_error(
            "CUDA dense L2 query grid exceeds device limit");
    }
    if (strategy == CudaDenseL2QueryStrategy::direct) {
        compute_dense_l2_distances<<<static_cast<unsigned int>(pair_count),
                                     threads_per_block>>>(
            implementation_->device_representatives.data(),
            device_queries,
            implementation_->representative_count, implementation_->dimension,
            query_count, workspace.implementation_->device_distances.data());
        check_kernel_launch("cannot launch CUDA dense L2-distance kernel");

        reduce_dense_l2_argmin<<<static_cast<unsigned int>(query_count),
                                 threads_per_block>>>(
            workspace.implementation_->device_distances.data(),
            implementation_->representative_count, query_count,
            workspace.implementation_->device_results.data());
        check_kernel_launch("cannot launch CUDA dense L2 argmin kernel");
    } else if (strategy == CudaDenseL2QueryStrategy::gemm) {
        if (implementation_->dimension == 0) {
            check_cuda(
                cudaMemset(workspace.implementation_->device_distances.data(),
                           0,
                           checked_bytes<float>(pair_count,
                                                "CUDA dense L2 GEMM output")),
                "cannot initialize CUDA dense L2 GEMM output");
            check_cuda(
                cudaMemset(workspace.implementation_->device_query_squared_norms
                               .data(),
                           0,
                           checked_bytes<float>(query_count,
                                                "CUDA dense L2 query norms")),
                "cannot initialize CUDA dense L2 query norms");
        } else {
            compute_row_squared_norms<<<static_cast<unsigned int>(query_count),
                                        threads_per_block>>>(
                device_queries, query_count,
                implementation_->dimension,
                workspace.implementation_->device_query_squared_norms.data());
            check_kernel_launch("cannot launch CUDA query-norm kernel");

            constexpr float alpha = 1.0F;
            constexpr float beta = 0.0F;
            check_cublas(
                cublasSgemm(
                    implementation_->cublas_handle.get(), CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    static_cast<int>(implementation_->representative_count),
                    static_cast<int>(query_count),
                    static_cast<int>(implementation_->dimension), &alpha,
                    implementation_->device_representatives.data(),
                    static_cast<int>(implementation_->dimension),
                    device_queries,
                    static_cast<int>(implementation_->dimension), &beta,
                    workspace.implementation_->device_distances.data(),
                    static_cast<int>(implementation_->representative_count)),
                "cannot compute CUDA dense L2 query dot products");
        }

        reduce_gemm_l2_argmin<<<static_cast<unsigned int>(query_count),
                                threads_per_block>>>(
            workspace.implementation_->device_distances.data(),
            implementation_->device_representative_squared_norms.data(),
            workspace.implementation_->device_query_squared_norms.data(),
            implementation_->representative_count, query_count,
            workspace.implementation_->device_results.data());
        check_kernel_launch("cannot launch CUDA dense L2 GEMM argmin kernel");
    } else {
        throw std::invalid_argument("unknown CUDA dense L2 query strategy");
    }

    check_cuda(cudaMemcpy(output.data(),
                          workspace.implementation_->device_results.data(),
                          checked_bytes<std::size_t>(
                              query_count, "CUDA dense L2 query results"),
                          cudaMemcpyDeviceToHost),
               "cannot copy dense L2 query results from CUDA device");
}

IndexSpaceUsage CudaDenseL2ScanIndex::space_usage() const noexcept
{
    return IndexSpaceUsage{
        .index_payload_bytes = implementation_->payload_bytes,
        .query_workspace_payload_bytes = 0,
        .unique_query_coordinates = implementation_->dimension,
        .sampled_multiplicity = implementation_->dimension,
    };
}

int CudaDenseL2ScanIndex::device() const noexcept
{
    return implementation_->selected_device.index();
}

const std::string& CudaDenseL2ScanIndex::device_name() const noexcept
{
    return implementation_->device_name;
}

} // namespace ultrahigh_ann
