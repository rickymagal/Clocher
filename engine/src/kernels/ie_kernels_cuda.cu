/**
 * @file ie_kernels_cuda.cu
 * @brief CUDA implementations of GEMV/activation/packing kernels + C-ABI launchers.
 *
 * This is the **only** TU that includes CUDA headers. The public header
 * `ie_kernels_cuda.h` keeps the rest of the codebase CUDA-agnostic.
 *
 * ## Kernel policy (baseline, safe defaults)
 * - **GEMV row-wise:** one block per output row, 256 threads per block, strided
 *   K loop with shared-memory reduction.
 * - **Fused GEMV+bias+activation:** same grid policy with a short epilogue.
 * - **Vector tanh:** grid-stride loop with 256-thread blocks, large grid.
 * - **Packing (Blocked-K):** 2D grid over `(cols, rows)` with 32x8 threads.
 *
 * These settings favor portability and predictability. They are easy to tune
 * per architecture once you profile on target GPUs.
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <stddef.h>

#define IE_CUDA_OK 0

/* Expose the same stream type name as in the public header. */
#define IE_CUDA_STREAM_T_DEFINED
typedef cudaStream_t ie_cuda_stream_t;

#include "ie_kernels_cuda.h"

/* ========================================================================== */
/*                                  Errors                                    */
/* ========================================================================== */

/**
 * @brief Thread-local buffer holding the most recent CUDA error message.
 *
 * The C-ABI launchers store any CUDA error textual description here so callers
 * can retrieve it via ::ie_cuda_last_error_string().
 */
static __thread char g_ie_cuda_err[256] = {0};

/**
 * @brief Store an error message into the thread-local buffer (truncates).
 *
 * @param msg NUL-terminated message to copy; if `NULL`, clears the buffer.
 */
static inline void ie_cuda_set_err(const char* msg) {
  if (!msg) { g_ie_cuda_err[0] = '\0'; return; }
  size_t i = 0;
  for (; i + 1 < sizeof(g_ie_cuda_err) && msg[i]; ++i) g_ie_cuda_err[i] = msg[i];
  g_ie_cuda_err[i] = '\0';
}

/**
 * @copydoc ie_cuda_last_error_string
 */
const char* ie_cuda_last_error_string(void) {
  return g_ie_cuda_err;
}

/**
 * @brief Guard macro for CUDA runtime calls inside launchers.
 *
 * On failure: records the CUDA error string and returns the negative error code.
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
 * @brief Apply activation in an epilogue.
 *
 * @param x    Input value.
 * @param act  Activation kind (see ::ie_act_kind_t).
 * @return Activated value.
 */
__device__ inline float ie_apply_activation(float x, ie_act_kind_t act) {
  if (act == IE_ACT_RELU) {
    return x > 0.f ? x : 0.f;
  } else if (act == IE_ACT_TANH) {
    return tanhf(x);
  }
  return x; /* IE_ACT_NONE */
}

/* ========================================================================== */
/*                                  Kernels                                   */
/* ========================================================================== */

/**
 * @brief Kernel: row-wise GEMV (computes `y = alpha * W * x + beta * y`).
 *
 * **Mapping:** one block per output row; threads perform a strided reduction
 * along the K (column) dimension; partial sums are reduced in shared memory.
 *
 * @param W      Row-major matrix, `rows x ldW` (device).
 * @param x      Input vector of length `cols` (device).
 * @param y      Output vector of length `rows` (device).
 * @param rows   Number of rows in `W` and elements in `y`.
 * @param cols   Number of columns in `W` and elements in `x`.
 * @param ldW    Leading dimension (>= cols).
 * @param alpha  Scale for `W*x`.
 * @param beta   Scale for existing `y`.
 */
__global__ void k_gemv_rowwise_f32(const float* __restrict__ W,
                                   const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int rows, int cols, int ldW,
                                   float alpha, float beta) {
  int r = blockIdx.x;
  if (r >= rows) return;

  float acc = 0.f;
  /* Parallel dot-product across K with a strided loop. */
  for (int k = threadIdx.x; k < cols; k += blockDim.x) {
    acc += W[(size_t)r * (size_t)ldW + (size_t)k] * x[k];
  }

  /* Shared-memory reduction within the block. */
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
 * @brief Kernel: fused GEMV + bias + activation.
 *
 * Computes `y = act(alpha * W*x + bias + beta*y)`.
 *
 * @param W      Row-major matrix, `rows x ldW` (device).
 * @param x      Input vector of length `cols` (device).
 * @param bias   Per-row bias of length `rows` (device, may be NULL).
 * @param y      Output vector of length `rows` (device).
 * @param rows   Number of rows / elements in `y`.
 * @param cols   Number of columns / elements in `x`.
 * @param ldW    Leading dimension (>= cols).
 * @param alpha  Scale for `W*x`.
 * @param beta   Scale for existing `y`.
 * @param act    Activation kind.
 */
__global__ void k_gemv_bias_act_f32(const float* __restrict__ W,
                                    const float* __restrict__ x,
                                    const float* __restrict__ bias,
                                    float* __restrict__ y,
                                    int rows, int cols, int ldW,
                                    float alpha, float beta,
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
 * @brief Kernel: elementwise `tanh` on a vector.
 *
 * @param y   Output vector (device).
 * @param x   Input vector (device).
 * @param n   Number of elements.
 */
__global__ void k_vec_tanh_f32(float* __restrict__ y,
                               const float* __restrict__ x,
                               int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    y[i] = tanhf(x[i]);
  }
}

/**
 * @brief Kernel: pack row-major `W` into Blocked-K layout (see header docs).
 *
 * @param Wp       Destination (device).
 * @param W        Source row-major (device).
 * @param rows     Rows.
 * @param cols     Cols.
 * @param ldW      Leading dimension of `W`.
 * @param block_k  Tile size along K.
 */
__global__ void k_pack_w_blockedk_f32(float* __restrict__ Wp,
                                      const float* __restrict__ W,
                                      int rows, int cols, int ldW,
                                      int block_k) {
  int r0 = blockIdx.y * blockDim.y + threadIdx.y;
  int k0 = blockIdx.x * blockDim.x + threadIdx.x;

  for (int r = r0; r < rows; r += blockDim.y * gridDim.y) {
    for (int k = k0; k < cols; k += blockDim.x * gridDim.x) {
      int kb = k / block_k;
      int ko = k % block_k;
      size_t dst = ((size_t)kb * (size_t)rows + (size_t)r) * (size_t)block_k + (size_t)ko;
      Wp[dst] = W[(size_t)r * (size_t)ldW + (size_t)k];
    }
  }
}

/* ========================================================================== */
/*                                  Launchers                                 */
/* ========================================================================== */

/**
 * @copydoc ie_cuda_launch_gemv_rowwise_f32
 */
int ie_cuda_launch_gemv_rowwise_f32(const float* W,
                                    const float* x,
                                    float*       y,
                                    int          rows,
                                    int          cols,
                                    int          ldW,
                                    float        alpha,
                                    float        beta,
                                    ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_rowwise_f32<<<grid, block, 0, stream>>>(W, x, y, rows, cols, ldW, alpha, beta);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @copydoc ie_cuda_launch_gemv_bias_act_f32
 */
int ie_cuda_launch_gemv_bias_act_f32(const float* W,
                                     const float* x,
                                     const float* bias,
                                     float*       y,
                                     int          rows,
                                     int          cols,
                                     int          ldW,
                                     float        alpha,
                                     float        beta,
                                     ie_act_kind_t act,
                                     ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(256, 1, 1);
  dim3 grid(rows, 1, 1);
  k_gemv_bias_act_f32<<<grid, block, 0, stream>>>(
      W, x, bias, y, rows, cols, ldW, alpha, beta, act);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @copydoc ie_cuda_launch_vec_tanh_f32
 */
int ie_cuda_launch_vec_tanh_f32(float* y,
                                const float* x,
                                int n,
                                ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!x || !y || n <= 0) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  const int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535; /* SM limit for 1D grids on many archs */
  k_vec_tanh_f32<<<grid, block, 0, stream>>>(y, x, n);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}

/**
 * @copydoc ie_cuda_launch_pack_w_blockedk_f32
 */
int ie_cuda_launch_pack_w_blockedk_f32(float*       Wp,
                                       const float* W,
                                       int          rows,
                                       int          cols,
                                       int          ldW,
                                       int          block_k,
                                       ie_cuda_stream_t stream) {
  ie_cuda_set_err(NULL);
  if (!Wp || !W || rows <= 0 || cols <= 0 || ldW < cols || block_k <= 0) {
    ie_cuda_set_err("invalid arguments");
    return -1;
  }
  dim3 block(32, 8, 1);  /* 256 threads per block */
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y, 1);
  k_pack_w_blockedk_f32<<<grid, block, 0, stream>>>(Wp, W, rows, cols, ldW, block_k);
  CUDA_GUARD(cudaGetLastError());
  return IE_CUDA_OK;
}
