/**
 * @file ie_kernels_cuda.cu
 * @brief CUDA math kernels used by inference engine (int8/int4 quantized ops, etc.).
 *
 * Note: The CUDA error-reporting function is now implemented in
 * engine/src/devices/ie_device_cuda.cu.
 */

#include "ie_kernels_cuda.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

extern "C" const char *ie_cuda_last_error_string(void);  // defined in ie_device_cuda.cu

// -----------------------------------------------------------------------------
// Example placeholder kernels
// -----------------------------------------------------------------------------

__global__ void ie_cuda_vector_add(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void ie_cuda_vector_mul(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

// -----------------------------------------------------------------------------
// Host-callable wrappers
// -----------------------------------------------------------------------------

extern "C" int ie_cuda_vector_add_launch(const float *a, const float *b, float *out, size_t n) {
  if (!a || !b || !out || n == 0) return -1;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  ie_cuda_vector_add<<<grid, block>>>(a, b, out, n);
  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    fprintf(stderr, "[ie_cuda_vector_add_launch] %s\n", cudaGetErrorString(st));
    return -2;
  }
  return 0;
}

extern "C" int ie_cuda_vector_mul_launch(const float *a, const float *b, float *out, size_t n) {
  if (!a || !b || !out || n == 0) return -1;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  ie_cuda_vector_mul<<<grid, block>>>(a, b, out, n);
  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    fprintf(stderr, "[ie_cuda_vector_mul_launch] %s\n", cudaGetErrorString(st));
    return -2;
  }
  return 0;
}
