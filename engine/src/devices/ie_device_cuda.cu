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
#include <cstdlib>
#include <cstdint>
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

/**
 * @brief Query free/total device memory.
 */
extern "C" int ie_cuda_mem_get_info(size_t *out_free, size_t *out_total) {
  if (!out_free || !out_total) return -1;
  size_t free_b = 0;
  size_t total_b = 0;
  const cudaError_t st = cudaMemGetInfo(&free_b, &total_b);
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("cudaMemGetInfo", st);
    return -2;
  }
  *out_free = free_b;
  *out_total = total_b;
  ie_cuda_clear_last_error();
  return 0;
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

/**
 * @brief Decode BF16 -> FP32 (device).
 */
__device__ __forceinline__ float ie_bf16_to_f32_u16_dev(uint16_t v) {
  const uint32_t u = ((uint32_t)v) << 16;
  return __uint_as_float(u);
}

/**
 * @brief Decode log2(u8, q3) scale encoding to FP32 (device).
 */
__device__ __forceinline__ float ie_log2_u8_q3_to_f32_dev(uint8_t v) {
  const float step = 0.125f;
  const int bias = 128;
  const float exp = ((float)((int)v - bias)) * step;
  return exp2f(exp);
}

/**
 * @brief Decode FP8 E4M3 -> FP32 (device).
 *
 * Matches ie_unpack_fp8_e4m3() in engine/src/quant/act_fp8.c.
 */
__device__ __forceinline__ float ie_fp8_e4m3_to_f32_dev(uint8_t v) {
  if (v == 0u) return 0.0f;
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 3) & 0xF;
  const uint8_t man = (v & 0x7);

  if (exp == 0) {
    return sign ? -0.0f : 0.0f;
  }
  const int bias = 7;
  const int e = ((int)exp) - bias;
  const float frac = (float)man / 8.0f;
  const float val = (1.0f + frac) * ldexpf(1.0f, e);
  return sign ? -val : val;
}

/**
 * @brief Decode a signed 4-bit integer stored in a nibble (device).
 */
__device__ __forceinline__ int8_t ie_s4_from_u4_dev(uint8_t u) {
  u &= 0x0F;
  return (u >= 8) ? (int8_t)((int)u - 16) : (int8_t)u;
}

/**
 * @brief Row-per-block GEMV kernel for Q4_0 weights and FP32 activations.
 *
 * Each block computes one output row. Threads within the block split the
 * block-level work across Q4 blocks and then reduce.
 */
__global__ void ie_gemv_q4_0_f32_kernel_rowblock(const uint8_t *W_blocks,
                                                const uint8_t *W_scales,
                                                size_t scale_bytes,
                                                const float *x,
                                                float *y,
                                                size_t rows,
                                                size_t cols,
                                                const uint16_t *bias_bf16) {
  const size_t r = (size_t)blockIdx.x;
  if (r >= rows) return;
  if (scale_bytes != 1u && scale_bytes != 2u) return;

  const size_t blocks_per_row = (cols + 31u) / 32u;
  const size_t row_w_bytes = blocks_per_row * 16u;
  const size_t row_s_bytes = blocks_per_row * (size_t)scale_bytes;

  const uint8_t *wrow = W_blocks + r * row_w_bytes;
  const uint8_t *srow = W_scales + r * row_s_bytes;

  float acc = 0.0f;
  for (size_t b = (size_t)threadIdx.x; b < blocks_per_row; b += (size_t)blockDim.x) {
    float s = 0.0f;
    if (scale_bytes == 2u) {
      const uint8_t *sp = srow + b * 2u;
      const uint16_t s16 = (uint16_t)sp[0] | ((uint16_t)sp[1] << 8);
      s = ie_bf16_to_f32_u16_dev(s16);
    } else {
      s = ie_fp8_e4m3_to_f32_dev(srow[b]);
    }

    const uint8_t *blk = wrow + b * 16u;
    const size_t base_c = b * 32u;
    const size_t limit_c = (base_c + 32u <= cols) ? (base_c + 32u) : cols;

    size_t c = base_c;
    for (size_t i = 0; i < 16u && c < limit_c; ++i) {
      const uint8_t byte = blk[i];
      const int8_t w0 = ie_s4_from_u4_dev((uint8_t)(byte & 0x0F));
      acc += (float)w0 * s * x[c];
      ++c;
      if (c >= limit_c) break;
      const int8_t w1 = ie_s4_from_u4_dev((uint8_t)(byte >> 4));
      acc += (float)w1 * s * x[c];
      ++c;
    }
  }

  __shared__ float ssum[128];
  ssum[threadIdx.x] = acc;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = ssum[0];
    if (bias_bf16) out += ie_bf16_to_f32_u16_dev(bias_bf16[r]);
    y[r] = out;
  }
}

__global__ void ie_gemv_bf16_f32_kernel_rowblock(const uint16_t *W_bf16,
                                                 const float *x,
                                                 float *y,
                                                 size_t rows,
                                                 size_t cols,
                                                 const uint16_t *bias_bf16) {
  const size_t r = (size_t)blockIdx.x;
  if (r >= rows) return;

  const uint16_t *wrow = W_bf16 + r * cols;

  float acc = 0.0f;
  for (size_t c = (size_t)threadIdx.x; c < cols; c += (size_t)blockDim.x) {
    const float w = ie_bf16_to_f32_u16_dev(wrow[c]);
    acc += w * x[c];
  }

  __shared__ float ssum[128];
  ssum[threadIdx.x] = acc;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ssum[threadIdx.x] += ssum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = ssum[0];
    if (bias_bf16) out += ie_bf16_to_f32_u16_dev(bias_bf16[r]);
    y[r] = out;
  }
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

/**
 * @brief Launch a Q4_0 GEMV kernel and synchronize.
 */
extern "C" int ie_cuda_gemv_q4_0_f32(const uint8_t *dW_q4,
                                    const uint8_t *dW_scales,
                                    size_t scale_bytes,
                                    const float *dx,
                                    float *dy,
                                    size_t rows,
                                    size_t cols,
                                    const uint16_t *dbias_bf16) {
  static int logged = 0;
  if (!dW_q4 || !dW_scales || !dx || !dy || rows == 0 || cols == 0) {
    std::snprintf(g_last_err, sizeof(g_last_err), "%s", "ie_cuda_gemv_q4_0_f32: invalid args");
    return -1;
  }

  const int block = 128;
  const int grid = (int)rows;

  if (!logged) {
    const char *log = std::getenv("IE_CUDA_LOG_KERNEL");
    if (log && log[0] != '\0' && log[0] != '0') {
      std::fprintf(stderr,
                   "info: cuda: q4 gemv kernel launch (rows=%zu cols=%zu scale_bytes=%zu)\n",
                   rows, cols, scale_bytes);
      std::fflush(stderr);
    }
    logged = 1;
  }

  ie_gemv_q4_0_f32_kernel_rowblock<<<grid, block>>>(dW_q4, dW_scales, scale_bytes,
                                                   dx, dy, rows, cols, dbias_bf16);

  cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("q4 kernel launch", st);
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

extern "C" int ie_cuda_gemv_bf16_f32(const uint16_t *dW_bf16,
                                     const float *dx,
                                     float *dy,
                                     size_t rows,
                                     size_t cols,
                                     const uint16_t *dbias_bf16) {
  if (!dW_bf16 || !dx || !dy || rows == 0 || cols == 0) return -1;

  const dim3 grid((unsigned)rows);
  const dim3 block(128);
  ie_gemv_bf16_f32_kernel_rowblock<<<grid, block>>>(dW_bf16, dx, dy, rows, cols, dbias_bf16);

  const cudaError_t st = cudaGetLastError();
  if (st != cudaSuccess) {
    ie_cuda_set_last_error_cuda("ie_gemv_bf16_f32_kernel_rowblock", st);
    return -2;
  }
  return 0;
}
