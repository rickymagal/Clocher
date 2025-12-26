/**
 * @file ie_device_cuda.cu
 * @brief CUDA runtime implementation for the engine's C ABI CUDA wrapper layer.
 *
 * This translation unit is compiled with NVCC and provides a strict C ABI so
 * that C code can invoke CUDA functionality without including CUDA headers.
 *
 * Error handling:
 *  - A single process-wide static buffer stores the last CUDA error string.
 *  - ie_cuda_last_error_string() exposes it via C ABI.
 *  - ie_cuda_set_last_error_string() and ie_cuda_clear_last_error() are also
 *    exported so other CUDA translation units can report errors without
 *    defining duplicate symbols.
 */

#include "ie_device_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

/* =============================================================================
 * Error handling (single TU ownership)
 * ========================================================================== */

/** @brief Global last-error string buffer (process-wide, owned by this TU). */
static char g_last_err[256] = "OK";

/**
 * @brief Store the last error message into the global static buffer.
 *
 * @param what Context string (may be NULL).
 * @param st CUDA status code.
 */
static void ie_cuda_set_last_error_cuda(const char *what, cudaError_t st) {
  const char *msg = cudaGetErrorString(st);
  std::snprintf(g_last_err, sizeof(g_last_err), "%s: %s",
                what ? what : "CUDA",
                msg ? msg : "Unknown");
}

/**
 * @brief Set the last error string directly (C ABI).
 *
 * This is intended for other CUDA translation units to report errors without
 * defining their own ie_cuda_last_error_string symbol.
 *
 * @param s Error string (may be NULL).
 */
extern "C" void ie_cuda_set_last_error_string(const char *s) {
  if (!s || !*s) {
    std::snprintf(g_last_err, sizeof(g_last_err), "%s", "unknown CUDA error");
    return;
  }
  std::snprintf(g_last_err, sizeof(g_last_err), "%s", s);
}

/**
 * @brief Clear the last error string (C ABI).
 */
extern "C" void ie_cuda_clear_last_error(void) {
  std::snprintf(g_last_err, sizeof(g_last_err), "%s", "OK");
}

/**
 * @brief Retrieve the last CUDA error string (C ABI).
 *
 * @return Pointer to a NUL-terminated static buffer (never NULL).
 */
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
  const size_t r = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
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

/**
 * @brief Check if CUDA is available by querying device count.
 *
 * @return 1 if at least one CUDA device is visible, 0 otherwise.
 */
extern "C" int ie_cuda_is_available(void) {
  int n = 0;
  const cudaError_t st = cudaGetDeviceCount(&n);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaGetDeviceCount", st);
    return 0;
  }
  ie_cuda_clear_last_error();
  return (n > 0) ? 1 : 0;
}

/**
 * @brief Initialize CUDA on a given device and return device/driver info.
 *
 * This forces runtime/driver initialization via cudaFree(0) so preflight
 * checks can observe real driver activity.
 *
 * @param device_ordinal CUDA device ordinal.
 * @param out_name Output buffer for device name (may be NULL).
 * @param name_cap Capacity of @p out_name in bytes.
 * @param out_driver Output buffer for driver/runtime versions (may be NULL).
 * @param driver_cap Capacity of @p out_driver in bytes.
 * @return 0 on success, negative on failure.
 */
extern "C" int ie_cuda_init(int device_ordinal,
                            char *out_name, size_t name_cap,
                            char *out_driver, size_t driver_cap) {
  cudaError_t st = cudaSetDevice(device_ordinal);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaSetDevice", st);
    return -1;
  }

  st = cudaFree(0);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaFree(0)", st);
    return -2;
  }

  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  st = cudaGetDeviceProperties(&prop, device_ordinal);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaGetDeviceProperties", st);
    return -3;
  }

  if (out_name && name_cap) {
    std::snprintf(out_name, name_cap, "%s", prop.name);
  }

  int drv = 0, rt = 0;
  (void)cudaDriverGetVersion(&drv);
  (void)cudaRuntimeGetVersion(&rt);

  if (out_driver && driver_cap) {
    std::snprintf(out_driver, driver_cap, "driver=%d runtime=%d", drv, rt);
  }

  ie_cuda_clear_last_error();
  return 0;
}

/**
 * @brief Allocate GPU memory.
 *
 * @param nbytes Number of bytes.
 * @return Device pointer on success, NULL on failure.
 */
extern "C" void *ie_cuda_malloc(size_t nbytes) {
  void *p = nullptr;
  const cudaError_t st = cudaMalloc(&p, nbytes);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaMalloc", st);
    return nullptr;
  }
  ie_cuda_clear_last_error();
  return p;
}

/**
 * @brief Free GPU memory (no-op for NULL).
 *
 * @param p Device pointer (may be NULL).
 */
extern "C" void ie_cuda_free(void *p) {
  if (!p) return;
  const cudaError_t st = cudaFree(p);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaFree", st);
    return;
  }
  ie_cuda_clear_last_error();
}

/**
 * @brief Copy memory between host/device with an explicit kind.
 *
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nbytes Number of bytes.
 * @param kind Copy direction.
 * @return 0 on success, negative on failure.
 */
extern "C" int ie_cuda_memcpy(void *dst, const void *src, size_t nbytes, ie_cuda_copy_kind_t kind) {
  cudaMemcpyKind ck = cudaMemcpyDefault;
  switch (kind) {
    case IE_CUDA_COPY_H2D: ck = cudaMemcpyHostToDevice; break;
    case IE_CUDA_COPY_D2H: ck = cudaMemcpyDeviceToHost; break;
    case IE_CUDA_COPY_D2D: ck = cudaMemcpyDeviceToDevice; break;
    case IE_CUDA_COPY_DEFAULT: default: ck = cudaMemcpyDefault; break;
  }

  const cudaError_t st = cudaMemcpy(dst, src, nbytes, ck);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaMemcpy", st);
    return -1;
  }

  ie_cuda_clear_last_error();
  return 0;
}

/**
 * @brief Launch a simple row-wise GEMV in FP32 and synchronize.
 *
 * @param dW Device pointer to row-major weights (rows x cols).
 * @param dx Device pointer to input vector (cols).
 * @param dy Device pointer to output vector (rows).
 * @param rows Rows.
 * @param cols Cols.
 * @param dbias Optional bias vector (rows) or NULL.
 * @return 0 on success, negative on failure.
 */
extern "C" int ie_cuda_gemv_f32(const float *dW,
                               const float *dx,
                               float *dy,
                               size_t rows,
                               size_t cols,
                               const float *dbias) {
  if (!dW || !dx || !dy || rows == 0 || cols == 0) {
    std::snprintf(g_last_err, sizeof(g_last_err), "%s", "ie_cuda_gemv_f32: invalid args");
    return -1;
  }

  const int block = 256;
  const int grid = (int)((rows + (size_t)block - 1) / (size_t)block);

  ie_gemv_rowwise_f32_kernel<<<grid, block>>>(dW, dx, dy, dbias, rows, cols);

  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("kernel launch", st);
    return -2;
  }

  st = cudaDeviceSynchronize();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaDeviceSynchronize", st);
    return -3;
  }

  ie_cuda_clear_last_error();
  return 0;
}
