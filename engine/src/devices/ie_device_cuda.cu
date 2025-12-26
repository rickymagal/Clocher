/**
 * @file ie_device_cuda.cu
 * @brief CUDA runtime implementation for the engine's C ABI CUDA wrapper layer.
 *
 * This translation unit is compiled with NVCC and provides a strict C ABI so
 * that C code can invoke CUDA functionality without including CUDA headers.
 */

#include "ie_device_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

/* =============================================================================
 * Error handling
 * ========================================================================== */

static char g_last_err[256] = "OK";

/**
 * @brief Store the last error message into a static buffer.
 *
 * @param what Context string.
 * @param st CUDA status.
 */
static void ie_cuda_set_last_error(const char *what, cudaError_t st) {
  const char *msg = cudaGetErrorString(st);
  std::snprintf(g_last_err, sizeof(g_last_err), "%s: %s", what ? what : "CUDA", msg ? msg : "Unknown");
}

extern "C" const char *ie_cuda_last_error_string(void) {
  return g_last_err;
}

/* =============================================================================
 * Minimal kernels
 * ========================================================================== */

/**
 * @brief Row-wise GEMV kernel: each thread computes one output row.
 *
 * This is a correctness-first kernel intended to validate real GPU execution.
 * Performance work (tiling, vectorization, shared memory, etc.) can follow.
 *
 * @param W Row-major matrix (rows x cols).
 * @param x Input vector (cols).
 * @param y Output vector (rows).
 * @param bias Optional bias (rows) or NULL.
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
__global__ void ie_gemv_rowwise_f32_kernel(const float *W,
                                          const float *x,
                                          float *y,
                                          const float *bias,
                                          size_t rows,
                                          size_t cols) {
  size_t r = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (r >= rows) return;

  const float *Wr = W + r * cols;
  float acc = 0.0f;

  for (size_t c = 0; c < cols; ++c) {
    acc += Wr[c] * x[c];
  }

  if (bias) acc += bias[r];
  y[r] = acc;
}

/* =============================================================================
 * Public C ABI
 * ========================================================================== */

extern "C" int ie_cuda_is_available(void) {
  int n = 0;
  cudaError_t st = cudaGetDeviceCount(&n);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaGetDeviceCount", st);
    return 0;
  }
  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
  return (n > 0) ? 1 : 0;
}

extern "C" int ie_cuda_init(int device_ordinal,
                            char *out_name, size_t name_cap,
                            char *out_driver, size_t driver_cap) {
  cudaError_t st;

  st = cudaSetDevice(device_ordinal);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaSetDevice", st);
    return -1;
  }

  /* Force runtime/driver initialization (observable driver activity). */
  st = cudaFree(0);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaFree(0)", st);
    return -2;
  }

  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  st = cudaGetDeviceProperties(&prop, device_ordinal);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaGetDeviceProperties", st);
    return -3;
  }

  if (out_name && name_cap) {
    std::snprintf(out_name, name_cap, "%s", prop.name);
  }

  int drv = 0, rt = 0;
  cudaDriverGetVersion(&drv);
  cudaRuntimeGetVersion(&rt);

  if (out_driver && driver_cap) {
    std::snprintf(out_driver, driver_cap, "driver=%d runtime=%d", drv, rt);
  }

  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
  return 0;
}

extern "C" void *ie_cuda_malloc(size_t nbytes) {
  void *p = nullptr;
  cudaError_t st = cudaMalloc(&p, nbytes);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaMalloc", st);
    return nullptr;
  }
  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
  return p;
}

extern "C" void ie_cuda_free(void *p) {
  if (!p) return;
  cudaError_t st = cudaFree(p);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaFree", st);
    return;
  }
  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
}

extern "C" int ie_cuda_memcpy(void *dst, const void *src, size_t nbytes, ie_cuda_copy_kind_t kind) {
  cudaMemcpyKind ck = cudaMemcpyDefault;
  switch (kind) {
    case IE_CUDA_COPY_H2D: ck = cudaMemcpyHostToDevice; break;
    case IE_CUDA_COPY_D2H: ck = cudaMemcpyDeviceToHost; break;
    case IE_CUDA_COPY_D2D: ck = cudaMemcpyDeviceToDevice; break;
    case IE_CUDA_COPY_DEFAULT: default: ck = cudaMemcpyDefault; break;
  }

  cudaError_t st = cudaMemcpy(dst, src, nbytes, ck);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaMemcpy", st);
    return -1;
  }

  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
  return 0;
}

extern "C" int ie_cuda_gemv_f32(const float *dW,
                               const float *dx,
                               float *dy,
                               size_t rows,
                               size_t cols,
                               const float *dbias) {
  if (!dW || !dx || !dy || rows == 0 || cols == 0) {
    std::snprintf(g_last_err, sizeof(g_last_err), "ie_cuda_gemv_f32: invalid args");
    return -1;
  }

  const int block = 256;
  const int grid = (int)((rows + (size_t)block - 1) / (size_t)block);

  ie_gemv_rowwise_f32_kernel<<<grid, block>>>(dW, dx, dy, dbias, rows, cols);

  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("kernel launch", st);
    return -2;
  }

  /* Correctness-first: sync. You can replace with stream semantics later. */
  st = cudaDeviceSynchronize();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error("cudaDeviceSynchronize", st);
    return -3;
  }

  std::snprintf(g_last_err, sizeof(g_last_err), "OK");
  return 0;
}
