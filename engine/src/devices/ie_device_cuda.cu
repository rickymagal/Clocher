/**
 * @file ie_device_cuda.cu
 * @brief CUDA backend implementation: device availability + FP32 GEMV kernel.
 *
 * This TU compiles with NVCC when IE_WITH_CUDA=1 and provides a
 * minimal, dependency-free GEMV path. It favors clarity and
 * correctness; later we can add persistent buffers and streams.
 *
 * ## Kernel strategy
 * One thread-block computes one output row (y[r] = dot(W[r,*], x)).
 * Each thread accumulates a partial sum over K with grid-stride loops,
 * then we reduce within the block to a single float and write y[r].
 *
 * ## Notes
 * - No cuBLAS is used (first-party requirement).
 * - Error handling wraps CUDA runtime calls and returns negative codes.
 */

#include "ie_device_cuda.h"

#if defined(IE_WITH_CUDA) && (IE_WITH_CUDA+0)==1

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ------------------------------ Utilities -------------------------------- */

/** @brief Map CUDA error to our unified return code. */
static inline int rc_from_cuda(cudaError_t st) {
  return (st == cudaSuccess) ? IE_CUDA_OK : IE_CUDA_ERR_RUNTIME;
}

/** @brief Simple CUDA error logging helper (compiled in only when needed). */
static inline void log_cuda_error(const char* where, cudaError_t st) {
  fprintf(stderr, "cuda-error at %s: %s (%d)\n", where, cudaGetErrorString(st), (int)st);
}

/* ------------------------------ Kernel ----------------------------------- */

/**
 * @brief Compute one row of y = W * x per block.
 *
 * @param W    Row-major matrix (rows x cols).
 * @param x    Vector (cols).
 * @param y    Output vector (rows).
 * @param rows Number of rows.
 * @param cols Number of cols.
 * @param ldw  Leading dimension (>= cols).
 * @param row_stride_e Row stride in elements (>= ldw).
 */
__global__ void gemv_rowwise_f32_kernel(const float* __restrict__ W,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int rows,
                                        int cols,
                                        int ldw,
                                        size_t row_stride_e)
{
  const int r = blockIdx.x;              // row handled by this block
  if (r >= rows) return;

  const float* row = W + r * row_stride_e;

  // Each thread accumulates partial over K with grid-stride loop on K.
  float sum = 0.0f;
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    sum += row[k] * x[k];
  }

  // Block-wide reduction in shared memory
  __shared__ float smem[256];            // assumes blockDim.x <= 256
  smem[threadIdx.x] = sum;
  __syncthreads();

  // Reduce to lane 0
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    y[r] = smem[0];
  }
}

/* --------------------------- Public API ---------------------------------- */

int ie_cuda_is_available(void) {
  int n = 0;
  cudaError_t st = cudaGetDeviceCount(&n);
  if (st != cudaSuccess) {
    // If the driver/runtime is missing or not initialized, treat as unavailable.
    return IE_CUDA_UNAVAILABLE;
  }
  return (n > 0) ? IE_CUDA_OK : IE_CUDA_UNAVAILABLE;
}

int ie_cuda_gemv_f32_strided(const float *W,
                             const float *x,
                             float *y,
                             int rows,
                             int cols,
                             int ldw_e,
                             size_t row_stride_bytes)
{
  if (!W || !x || !y) return IE_CUDA_EINVAL;
  if (rows <= 0 || cols <= 0 || ldw_e < cols) return IE_CUDA_EINVAL;

  // Check availability once; if no device, report unavailable.
  {
    int rc = ie_cuda_is_available();
    if (rc != IE_CUDA_OK) return rc;
  }

  const size_t row_stride_e = (row_stride_bytes == 0)
                            ? (size_t)ldw_e
                            : (row_stride_bytes / sizeof(float));

  // Allocate device buffers
  float *dW = nullptr, *dx = nullptr, *dy = nullptr;
  const size_t bytesW = (size_t)rows * row_stride_e * sizeof(float);
  const size_t bytesX = (size_t)cols * sizeof(float);
  const size_t bytesY = (size_t)rows * sizeof(float);

  cudaError_t st;

  st = cudaMalloc((void**)&dW, bytesW);
  if (st != cudaSuccess) { log_cuda_error("cudaMalloc(dW)", st); return IE_CUDA_ERR_RUNTIME; }
  st = cudaMalloc((void**)&dx, bytesX);
  if (st != cudaSuccess) { log_cuda_error("cudaMalloc(dx)", st); cudaFree(dW); return IE_CUDA_ERR_RUNTIME; }
  st = cudaMalloc((void**)&dy, bytesY);
  if (st != cudaSuccess) { log_cuda_error("cudaMalloc(dy)", st); cudaFree(dx); cudaFree(dW); return IE_CUDA_ERR_RUNTIME; }

  // Copy inputs H2D
  st = cudaMemcpy(dW, W, bytesW, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) { log_cuda_error("cudaMemcpy(dW)", st); cudaFree(dy); cudaFree(dx); cudaFree(dW); return IE_CUDA_ERR_RUNTIME; }

  st = cudaMemcpy(dx, x, bytesX, cudaMemcpyHostToDevice);
  if (st != cudaSuccess) { log_cuda_error("cudaMemcpy(dx)", st); cudaFree(dy); cudaFree(dx); cudaFree(dW); return IE_CUDA_ERR_RUNTIME; }

  // Launch kernel: one block per row; 256 threads per block (safe upper shared size).
  dim3 grid(rows);
  dim3 block(256);
  gemv_rowwise_f32_kernel<<<grid, block>>>(dW, dx, dy, rows, cols, ldw_e, row_stride_e);
  st = cudaGetLastError();
  if (st != cudaSuccess) {
    log_cuda_error("kernel launch", st);
    cudaFree(dy); cudaFree(dx); cudaFree(dW);
    return IE_CUDA_ERR_RUNTIME;
  }

  // Copy result D2H
  st = cudaMemcpy(y, dy, bytesY, cudaMemcpyDeviceToHost);
  if (st != cudaSuccess) { log_cuda_error("cudaMemcpy(y)", st); cudaFree(dy); cudaFree(dx); cudaFree(dW); return IE_CUDA_ERR_RUNTIME; }

  // Cleanup
  cudaFree(dy);
  cudaFree(dx);
  cudaFree(dW);

  return IE_CUDA_OK;
}

int ie_cuda_gemv_f32(const float *W,
                     const float *x,
                     float *y,
                     int rows,
                     int cols,
                     int ldw)
{
  return ie_cuda_gemv_f32_strided(W, x, y, rows, cols, ldw, 0 /* row_stride_bytes */);
}

#else /* IE_WITH_CUDA != 1 */

int ie_cuda_is_available(void) {
  return IE_CUDA_UNAVAILABLE;
}

int ie_cuda_gemv_f32_strided(const float *W,
                             const float *x,
                             float *y,
                             int rows,
                             int cols,
                             int ldw_e,
                             size_t row_stride_bytes)
{
  (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)ldw_e; (void)row_stride_bytes;
  return IE_CUDA_UNAVAILABLE;
}

int ie_cuda_gemv_f32(const float *W,
                     const float *x,
                     float *y,
                     int rows,
                     int cols,
                     int ldw)
{
  (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)ldw;
  return IE_CUDA_UNAVAILABLE;
}

#endif /* IE_WITH_CUDA */

