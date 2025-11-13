// File: engine/src/kernels/ie_kernels_cuda.cu
// -----------------------------------------------------------------------------
/// @file ie_kernels_cuda.cu
/// @brief CUDA implementations of GEMV/activation/packing kernels + C-ABI launchers.
///
/// This is the **only** translation unit that includes CUDA headers. The public
/// header `ie_kernels_cuda.h` keeps the rest of the codebase CUDA-agnostic.
///
/// ## Kernel policy (baseline, safe defaults)
/// - **GEMV row-wise:** one block per output row, 256 threads per block, strided
///   K loop with shared-memory reduction.
/// - **Fused GEMV+bias+activation:** same grid policy with a short epilogue.
/// - **Vector tanh:** grid-stride loop with 256-thread blocks, large grid.
/// - **Packing (Blocked-K):** 2D grid over (cols, rows) with 32x8 threads.
///
/// Additions in this file:
/// - Fused INT8 activation GEMV (per-tensor parameters).
/// - Fused FP8 activation GEMV (E4M3/E5M2) with on-device decoders.
///
/// These settings favor portability and predictability. They are easy to tune
/// per architecture once you profile on target GPUs.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stddef.h>
#include <stdint.h>

#define IE_CUDA_OK 0

/* Expose the same stream type name as in the public header. */
#define IE_CUDA_STREAM_T_DEFINED
typedef cudaStream_t ie_cuda_stream_t;

#include "ie_kernels_cuda.h"
#include "ie_quant_act.h" /* for ie_fp8_format, ie_act_kind_t */

static __thread char g_ie_cuda_err[256] = {0};

/**
 * @brief Set or clear the per-thread CUDA error string buffer.
 * @param msg NULL to clear; otherwise a short literal/diagnostic string.
 */
static inline void ie_cuda_set_err(const char *msg) {
  if (!msg) {
    g_ie_cuda_err[0] = '\0';
    return;
  }
  size_t i = 0;
  for (; i + 1 < sizeof(g_ie_cuda_err) && msg[i]; ++i) g_ie_cuda_err[i] = msg[i];
  g_ie_cuda_err[i] = '\0';
}

/**
 * @brief Retrieve the last CUDA error string set by a launcher in this TU.
 * @return Pointer to a thread-local, NUL-terminated message (may be "").
 */
const char *ie_cuda_last_error_string(void) { return g_ie_cuda_err; }

/**
 * @brief CUDA error guard macro that stores the error string and returns.
 *
 * Use this after launching kernels to convert CUDA errors into negative
 * integer codes and a readable message retrievable via
 * ::ie_cuda_last_error_string().
 */
#define CUDA_GUARD(call)                                                     \
  do {                                                                       \
    cudaError_t _st = (call);                                                \
    if (_st != cudaSuccess) {                                                \
      ie_cuda_set_err(cudaGetErrorString(_st));                              \
      return -(int)_st;                                                      \
    }                                                                        \
  } while (0)

/* ========================================================================== */
/*                               Device helpers                               */
/* ========================================================================== */

/**
 * @brief Apply a simple activation kind to a scalar.
 * @param x   Input value.
 * @param act Activation selector (::IE_ACT_NONE, ::IE_ACT_RELU, ::IE_ACT_TANH).
 * @return Activated output.
 */
__device__ inline float ie_apply_activation(float x, ie_act_kind_t act) {
  if (act == IE_ACT_RELU) {
    return x > 0.f ? x : 0.f;
  } else if (act == IE_ACT_TANH) {
    return tanhf(x);
  }
  return x; /* IE_ACT_NONE */
}

/* ---- FP8 decoders (device) ---- */

/**
 * @brief Decode one E4M3 FP8 byte on device (subnormals flushed to zero).
 * @param v Encoded E4M3 byte.
 * @return Decoded 32-bit float.
 */
__device__ inline float ie_decode_fp8_e4m3_u8(uint8_t v) {
  if (v == 0u) return 0.0f;
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 3) & 0xF;
  const uint8_t man = (v & 0x7);
  if (exp == 0) return sign ? -0.0f : 0.0f;
  const int bias = 7;
  const int e = (int)exp - bias;
  const float frac = (float)man / 8.0f;
  const float val = (1.0f + frac) * __int2float_rn(1 << e);
  return sign ? -val : val;
}

/**
 * @brief Decode one E5M2 FP8 byte on device (IEEE-like special cases).
 * @param v Encoded E5M2 byte.
 * @return Decoded 32-bit float (NaN/Inf propagated).
 */
__device__ inline float ie_decode_fp8_e5m2_u8(uint8_t v) {
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 2) & 0x1F;
  const uint8_t man = (v & 0x3);
  if (exp == 0) return sign ? -0.0f : 0.0f;
  if (exp == 0x1F) {
    if (man == 0) return sign ? -CUDART_INF_F : CUDART_INF_F;
    return CUDART_NAN_F;
  }
  const int bias = 15;
  const int e = (int)exp - bias;
  const float frac = (float)man / 4.0f;
  const float val = (1.0f + frac) * __int2float_rn(1 << e);
  return sign ? -val : val;
}

/* ========================================================================== */
/*                                  Kernels                                   */
/* ========================================================================== */

/**
 * @brief Row-wise GEMV kernel: y = alpha * W * x + beta * y.
 *
 * **Mapping:** 1 block per row. Threads accumulate a strided reduction over K
 * and reduce via shared memory. Assumes @p ldW >= @p cols.
 *
 * @param W     Row-major matrix on device, size rows x ldW.
 * @param x     Input vector on device, size cols.
 * @param y     Output vector on device, size rows.
 * @param rows  Number of rows / outputs.
 * @param cols  Number of columns / inputs.
 * @param ldW   Leading dimension (>= cols).
 * @param alpha Scale for W*x.
 * @param beta  Scale for existing y.
 */
__global__ void k_gemv_rowwise_f32(const float *__restrict__ W,
                                   const float *__restrict__ x,
                                   float *__restrict__ y, int rows, int cols,
                                   int ldW, float alpha, float beta) {
  int r = blockIdx.x;
  if (r >= rows) return;

  float acc = 0.f;
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    acc += W[(size_t)r * (size_t)ldW + (size_t)k] * x[k];
  }

  __shared__ float buf[256];
  buf[threadIdx.x] = acc;
  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = alpha * buf[0];
    if (beta != 0.f) out += beta * y[r];
    y[r] = out;
  }
}

/**
 * @brief Fused GEMV + bias + activation kernel (float activations).
 *
 * Same mapping as ::k_gemv_rowwise_f32 with an epilogue that adds bias and
 * applies a simple activation (none/ReLU/tanh).
 */
__global__ void k_gemv_bias_act_f32(const float *__restrict__ W,
                                    const float *__restrict__ x,
                                    const float *__restrict__ bias,
                                    float *__restrict__ y, int rows, int cols,
                                    int ldW, float alpha, float beta,
                                    ie_act_kind_t act) {
  int r = blockIdx.x;
  if (r >= rows) return;

  float acc = 0.f;
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    acc += W[(size_t)r * (size_t)ldW + (size_t)k] * x[k];
  }

  __shared__ float buf[256];
  buf[threadIdx.x] = acc;
  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float b = bias ? bias[r] : 0.f;
    float out = alpha * buf[0] + b;
    if (beta != 0.f) out += beta * y[r];
    y[r] = ie_apply_activation(out, act);
  }
}

/**
 * @brief Row-wise GEMV with INT8 activations (per-tensor), fused dequantization.
 *
 * Dequantization model: real = scale * (q - zero_point).
 *
 * @param W      Float weights on device (row-major), size rows x ldW.
 * @param xq     INT8 activations on device, length cols.
 * @param y      Output on device, length rows.
 * @param rows   Rows.
 * @param cols   Cols.
 * @param ldW    Leading dimension (>= cols).
 * @param scale  Per-tensor scale.
 * @param zp     Per-tensor zero-point (int).
 * @param alpha  Scale for W*x.
 * @param beta   Scale for existing y.
 */
__global__ void k_gemv_rowwise_qi8_f32(const float *__restrict__ W,
                                       const int8_t *__restrict__ xq,
                                       float *__restrict__ y, int rows,
                                       int cols, int ldW, float scale, int zp,
                                       float alpha, float beta) {
  int r = blockIdx.x;
  if (r >= rows) return;

  float acc = 0.f;
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    const float xv = scale * ((int)xq[k] - zp);
    acc += W[(size_t)r * (size_t)ldW + (size_t)k] * xv;
  }

  __shared__ float buf[256];
  buf[threadIdx.x] = acc;
  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = alpha * buf[0];
    if (beta != 0.f) out += beta * y[r];
    y[r] = out;
  }
}

/**
 * @brief Row-wise GEMV with FP8 activations (E4M3/E5M2), fused byte decode.
 *
 * @param W     Float weights on device (row-major), size rows x ldW.
 * @param x8    FP8 activations on device (bytes), length cols.
 * @param y     Output on device, length rows.
 * @param rows  Rows.
 * @param cols  Cols.
 * @param ldW   Leading dimension (>= cols).
 * @param fmt   FP8 format (::IE_FP8_E4M3 or ::IE_FP8_E5M2).
 * @param alpha Scale for W*x.
 * @param beta  Scale for existing y.
 */
__global__ void k_gemv_rowwise_qfp8_f32(const float *__restrict__ W,
                                        const uint8_t *__restrict__ x8,
                                        float *__restrict__ y, int rows,
                                        int cols, int ldW, ie_fp8_format fmt,
                                        float alpha, float beta) {
  int r = blockIdx.x;
  if (r >= rows) return;

  const bool e4m3 = (fmt == IE_FP8_E4M3);

  float acc = 0.f;
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    const uint8_t b = x8[k];
    const float xv =
        e4m3 ? ie_decode_fp8_e4m3_u8(b) : ie_decode_fp8_e5m2_u8(b);
    acc += W[(size_t)r * (size_t)ldW + (size_t)k] * xv;
  }

  __shared__ float buf[256];
  buf[threadIdx.x] = acc;
  __syncthreads();

  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float out = alpha * buf[0];
    if (beta != 0.f) out += beta * y[r];
    y[r] = out;
  }
}

/**
 * @brief Elementwise hyperbolic tangent on a vector.
 * @param y Output vector on device (length n).
 * @param x Input vector on device (length n).
 * @param n Number of elements.
 */
__global__ void k_vec_tanh_f32(float *__restrict__ y,
                               const float *__restrict__ x, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = tanhf(x[i]);
  }
}

/**
 * @brief Pack row-major W into Blocked-K layout on device (see header docs).
 *
 * Layout: for each (row r, column k) compute block index kb = k / block_k and
 * offset ko = k % block_k, and store at:
 *   dst = ((kb * rows + r) * block_k + ko).
 *
 * @param Wp      Destination (blocked) on device.
 * @param W       Source row-major on device (rows x ldW).
 * @param rows    Number of rows.
 * @param cols    Number of columns.
 * @param ldW     Leading dimension (>= cols).
 * @param block_k Block size in K (>= 1).
 */
__global__ void k_pack_w_blockedk_f32(float *__restrict__ Wp,
                                      const float *__restrict__ W, int rows,
                                      int cols, int ldW, int block_k) {
  int r0 = blockIdx.y * blockDim.y + threadIdx.y;
  int k0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int r = r0; r < rows; r += blockDim.y * gridDim.y) {
    for (int k = k0; k < cols; k += blockDim.x * gridDim.x) {
      int kb = k / block_k;
      int ko = k % block_k;
      size_t dst = ((size_t)kb * (size_t)rows + (size_t)r) * (size_t)block_k +
                   (size_t)ko;
      Wp[dst] = W[(size_t)r * (size_t)ldW + (size_t)k];
    }
  }
}

/* ========================================================================== */
/*                                  Launchers                                 */
/* ========================================================================== */

/**
 * @brief Launch row-wise GEMV (float path).
 * @copydetails k_gemv_rowwise_f32
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_gemv_rowwise_f32(const float *W, const float *x, float *y,
                                    int rows, int cols, int ldW, float alpha,
                                    float beta, ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_rowwise_f32<<<grid, block, 0, stream>>>(W, x, y, rows, cols, ldW,
                                                  alpha, beta);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @brief Launch fused GEMV + bias + activation (float).
 * @copydetails k_gemv_bias_act_f32
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_gemv_bias_act_f32(const float *W, const float *x,
                                     const float *bias, float *y, int rows,
                                     int cols, int ldW, float alpha, float beta,
                                     ie_act_kind_t act,
                                     ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_bias_act_f32<<<grid, block, 0, stream>>>(W, x, bias, y, rows, cols,
                                                  ldW, alpha, beta, act);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @brief Launch GEMV with INT8 activations (per-tensor), fused dequantization.
 * @copydetails k_gemv_rowwise_qi8_f32
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_gemv_rowwise_qi8_f32(const float *W, const int8_t *xq,
                                        float *y, int rows, int cols, int ldW,
                                        float scale, int zp, float alpha,
                                        float beta, ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !xq || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_rowwise_qi8_f32<<<grid, block, 0, stream>>>(W, xq, y, rows, cols,
                                                     ldW, scale, zp, alpha,
                                                     beta);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @brief Launch GEMV with FP8 activations (E4M3/E5M2), fused byte decode.
 * @copydetails k_gemv_rowwise_qfp8_f32
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_gemv_rowwise_qfp8_f32(const float *W, const uint8_t *x8,
                                         float *y, int rows, int cols, int ldW,
                                         ie_fp8_format fmt, float alpha,
                                         float beta, ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !x8 || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_rowwise_qfp8_f32<<<grid, block, 0, stream>>>(W, x8, y, rows, cols,
                                                      ldW, fmt, alpha, beta);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @brief Launch elementwise tanh on device.
 * @param y Output pointer (device).
 * @param x Input pointer (device).
 * @param n Number of elements.
 * @param stream CUDA stream (may be 0).
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_vec_tanh_f32(float *y, const float *x, int n,
                                ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!x || !y || n <= 0) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  const int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  k_vec_tanh_f32<<<grid, block, 0, stream>>>(y, x, n);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @brief Launch packing of row-major W into Blocked-K layout on device.
 * @copydetails k_pack_w_blockedk_f32
 * @return 0 on success or negative CUDA error code on failure.
 */
int ie_cuda_launch_pack_w_blockedk_f32(float *Wp, const float *W, int rows,
                                       int cols, int ldW, int block_k,
                                       ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!Wp || !W || rows <= 0 || cols <= 0 || ldW < cols || block_k <= 0) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(32, 8, 1); /* 256 threads per block */
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);
  k_pack_w_blockedk_f32<<<grid, block, 0, stream>>>(Wp, W, rows, cols, ldW,
                                                    block_k);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}