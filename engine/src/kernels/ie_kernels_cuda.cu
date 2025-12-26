#include "ie_kernels_cuda.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>

/*
 * IMPORTANT:
 *  - ie_cuda_last_error_string() is defined in engine/src/devices/ie_device_cuda.cu
 *    to avoid multiple-definition link errors.
 *  - This TU reports errors through ie_cuda_set_last_error_string() and clears
 *    through ie_cuda_clear_last_error(), both exported by ie_device_cuda.cu.
 */

extern "C" void ie_cuda_set_last_error_string(const char *s);
extern "C" void ie_cuda_clear_last_error(void);

static std::atomic<int> g_last_err_code{0};

#define IE_CUDA_OK 0

/**
 * @brief Record a CUDA error into the shared global error string.
 *
 * @param st CUDA error status.
 */
static inline void ie_cuda_record_cuda_error(cudaError_t st) {
  g_last_err_code.store((int)st, std::memory_order_relaxed);
  ie_cuda_set_last_error_string(cudaGetErrorString(st));
}

#define CUDA_GUARD(expr) do { \
  const cudaError_t _st = (expr); \
  if (_st != cudaSuccess) { \
    ie_cuda_record_cuda_error(_st); \
    return -1; \
  } \
} while (0)

/* =============================================================================
 * GEMV kernels (row-wise)
 * ========================================================================== */

/**
 * @brief Row-wise GEMV kernel for FP32 weights.
 *
 * @param y Output vector (rows).
 * @param w Weight matrix (rows*cols), row-major.
 * @param x Input vector (cols).
 * @param rows Number of rows.
 * @param cols Number of cols.
 */
template <typename T>
__global__ void gemv_rowwise_f32_kernel(
  float *y,
  const T *w,
  const float *x,
  int rows,
  int cols
) {
  const int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (r >= rows) return;

  float acc = 0.0f;
  const int base = r * cols;
  for (int c = 0; c < cols; ++c) {
    acc += (float)w[base + c] * x[c];
  }
  y[r] = acc;
}

/**
 * @brief Launch row-wise GEMV for FP32 weights.
 *
 * @param y Output (device).
 * @param w Weights (device).
 * @param x Input (device).
 * @param rows Rows.
 * @param cols Cols.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_launch_gemv_rowwise_f32(
  float *y,
  const float *w,
  const float *x,
  int rows,
  int cols
) {
  if (!y || !w || !x || rows <= 0 || cols <= 0) return -1;

  const int threads = 256;
  const int blocks = (rows + threads - 1) / threads;

  gemv_rowwise_f32_kernel<float><<<blocks, threads>>>(y, w, x, rows, cols);
  CUDA_GUARD(cudaGetLastError());

  ie_cuda_clear_last_error();
  return IE_CUDA_OK;
}

/**
 * @brief Row-wise GEMV kernel for INT8 weights and INT8 input.
 *
 * @param y Output (device, FP32).
 * @param w Weights (device, INT8).
 * @param x Input (device, INT8).
 * @param rows Rows.
 * @param cols Cols.
 * @param w_scale Weight dequant scale.
 * @param x_scale Input dequant scale.
 */
__global__ void gemv_rowwise_int8_kernel(
  float *y,
  const int8_t *w,
  const int8_t *x,
  int rows,
  int cols,
  float w_scale,
  float x_scale
) {
  const int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (r >= rows) return;

  float acc = 0.0f;
  const int base = r * cols;
  for (int c = 0; c < cols; ++c) {
    const float wf = (float)w[base + c] * w_scale;
    const float xf = (float)x[c] * x_scale;
    acc += wf * xf;
  }
  y[r] = acc;
}

/**
 * @brief Launch row-wise GEMV for INT8 weights and INT8 input.
 *
 * @param y Output (device, FP32).
 * @param w Weights (device, INT8).
 * @param x Input (device, INT8).
 * @param rows Rows.
 * @param cols Cols.
 * @param w_scale Weight dequant scale.
 * @param x_scale Input dequant scale.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_launch_gemv_rowwise_int8(
  float *y,
  const int8_t *w,
  const int8_t *x,
  int rows,
  int cols,
  float w_scale,
  float x_scale
) {
  if (!y || !w || !x || rows <= 0 || cols <= 0) return -1;

  const int threads = 256;
  const int blocks = (rows + threads - 1) / threads;

  gemv_rowwise_int8_kernel<<<blocks, threads>>>(y, w, x, rows, cols, w_scale, x_scale);
  CUDA_GUARD(cudaGetLastError());

  ie_cuda_clear_last_error();
  return IE_CUDA_OK;
}

/**
 * @brief Unpack a signed 4-bit integer from a packed byte.
 *
 * @param byte Packed byte containing two int4 values.
 * @param hi If nonzero, use high nibble; else low nibble.
 * @return Sign-extended int8 value in [-8, 7].
 */
__device__ __forceinline__ int8_t unpack_int4(const uint8_t byte, const int hi) {
  const uint8_t v = hi ? (byte >> 4) : (byte & 0x0F);
  const int8_t s = (v & 0x08) ? (int8_t)(v | 0xF0) : (int8_t)v;
  return s;
}

/**
 * @brief Row-wise GEMV kernel for INT4 packed weights and INT8 input.
 *
 * Weights are packed: two int4 values per byte (low/high nibble).
 *
 * @param y Output (device, FP32).
 * @param w_packed Weights (device, packed int4).
 * @param x Input (device, INT8).
 * @param rows Rows.
 * @param cols Cols (logical, unpacked).
 * @param w_scale Weight dequant scale.
 * @param x_scale Input dequant scale.
 */
__global__ void gemv_rowwise_int4_kernel(
  float *y,
  const uint8_t *w_packed,
  const int8_t *x,
  int rows,
  int cols,
  float w_scale,
  float x_scale
) {
  const int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (r >= rows) return;

  float acc = 0.0f;
  const int base = r * cols;

  for (int c = 0; c < cols; ++c) {
    const int wi = base + c;
    const uint8_t byte = w_packed[wi >> 1];
    const int8_t wq = unpack_int4(byte, wi & 1);

    const float wf = (float)wq * w_scale;
    const float xf = (float)x[c] * x_scale;
    acc += wf * xf;
  }
  y[r] = acc;
}

/**
 * @brief Launch row-wise GEMV for INT4 packed weights and INT8 input.
 *
 * @param y Output (device, FP32).
 * @param w Weights (device, packed int4).
 * @param x Input (device, INT8).
 * @param rows Rows.
 * @param cols Cols (logical, unpacked).
 * @param w_scale Weight dequant scale.
 * @param x_scale Input dequant scale.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_launch_gemv_rowwise_int4(
  float *y,
  const uint8_t *w,
  const int8_t *x,
  int rows,
  int cols,
  float w_scale,
  float x_scale
) {
  if (!y || !w || !x || rows <= 0 || cols <= 0) return -1;

  const int threads = 256;
  const int blocks = (rows + threads - 1) / threads;

  gemv_rowwise_int4_kernel<<<blocks, threads>>>(y, w, x, rows, cols, w_scale, x_scale);
  CUDA_GUARD(cudaGetLastError());

  ie_cuda_clear_last_error();
  return IE_CUDA_OK;
}

/* =============================================================================
 * Weight packing helper
 * ========================================================================== */

/**
 * @brief Pack FP32 weights into a blocked-K layout for improved access patterns.
 *
 * @param out Output buffer (device).
 * @param w Input weights (device), row-major.
 * @param rows Rows.
 * @param cols Cols.
 * @param block_k Block size along K (columns).
 */
__global__ void pack_w_blockedk_f32_kernel(
  float *out,
  const float *w,
  int rows,
  int cols,
  int block_k
) {
  const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) return;

  const int r = idx / cols;
  const int c = idx % cols;

  const int bk = (block_k > 0) ? block_k : 1;
  const int cb = c / bk;
  const int ci = c % bk;

  const int cols_blk = (cols / bk) + ((cols % bk) ? 1 : 0);
  const int out_idx = (r * cols_blk + cb) * bk + ci;
  out[out_idx] = w[r * cols + c];
}

/**
 * @brief Launch FP32 weight packing into blocked-K layout.
 *
 * @param out Output (device).
 * @param w Input weights (device).
 * @param rows Rows.
 * @param cols Cols.
 * @param block_k Block size.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_pack_w_blockedk_f32(
  float *out,
  const float *w,
  int rows,
  int cols,
  int block_k
) {
  if (!out || !w || rows <= 0 || cols <= 0) return -1;

  const int threads = 256;
  const int total = rows * cols;
  const int blocks = (total + threads - 1) / threads;

  pack_w_blockedk_f32_kernel<<<blocks, threads>>>(out, w, rows, cols, block_k);
  CUDA_GUARD(cudaGetLastError());

  ie_cuda_clear_last_error();
  return IE_CUDA_OK;
}

/* =============================================================================
 * Strict benchmark touch loop
 * ========================================================================== */

/**
 * @brief CUDA kernel that touches a buffer with a given stride.
 *
 * Each thread reads bytes spaced by @p stride_bytes and accumulates into a global
 * counter to prevent the compiler from optimizing the loads away.
 *
 * @param buf Device buffer to read.
 * @param nbytes Total bytes in @p buf.
 * @param stride_bytes Stride between reads (0 treated as 1).
 * @param tokens Number of token-iterations.
 * @param out_acc Device accumulator (atomic add).
 */
__global__ static void ie_cuda_touch_kernel(
  const uint8_t *buf,
  size_t nbytes,
  size_t stride_bytes,
  uint64_t tokens,
  unsigned long long *out_acc
) {
  const size_t stride = (stride_bytes == 0) ? 1 : stride_bytes;
  const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t total_threads = (size_t)gridDim.x * (size_t)blockDim.x;

  unsigned long long local = 0;

  for (uint64_t t = 0; t < tokens; ++t) {
    for (size_t i = tid * stride; i < nbytes; i += total_threads * stride) {
      local += (unsigned long long)buf[i];
    }
  }

  if (local) atomicAdd(out_acc, local);
}

/** @brief Cached device buffer used by ie_cuda_touch_bytes(). */
static uint8_t *g_touch_buf = NULL;
/** @brief Cached capacity of @p g_touch_buf. */
static size_t g_touch_buf_cap = 0;
/** @brief Cached device accumulator used by ie_cuda_touch_bytes(). */
static unsigned long long *g_touch_acc = NULL;

/**
 * @brief Touch a given number of bytes per token on the GPU to ensure real CUDA activity.
 *
 * This is used by the strict benchmark harness to prevent "fake GPU" reports
 * where the runtime silently falls back to CPU or does no meaningful work.
 *
 * @param seed Optional host seed bytes to initialize the device buffer.
 * @param seed_len Length of @p seed in bytes.
 * @param bytes_per_token Number of bytes to touch per token.
 * @param stride_bytes Stride between touches (0 treated as 1).
 * @param tokens Number of token-iterations.
 * @param verify_touch If nonzero, copy back checksum and require it to be nonzero.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_touch_bytes(
  const void *seed,
  size_t seed_len,
  size_t bytes_per_token,
  size_t stride_bytes,
  uint64_t tokens,
  int verify_touch
) {
  if (bytes_per_token == 0 || tokens == 0) {
    ie_cuda_clear_last_error();
    return IE_CUDA_OK;
  }

  CUDA_GUARD(cudaFree(0));

  if (!g_touch_buf || g_touch_buf_cap != bytes_per_token) {
    if (g_touch_buf) {
      (void)cudaFree(g_touch_buf);
      g_touch_buf = NULL;
      g_touch_buf_cap = 0;
    }
    CUDA_GUARD(cudaMalloc((void **)&g_touch_buf, bytes_per_token));
    g_touch_buf_cap = bytes_per_token;
  }

  if (!g_touch_acc) {
    CUDA_GUARD(cudaMalloc((void **)&g_touch_acc, sizeof(*g_touch_acc)));
  }

  if (seed && seed_len > 0) {
    const size_t n_copy = (seed_len < g_touch_buf_cap) ? seed_len : g_touch_buf_cap;
    CUDA_GUARD(cudaMemcpy(g_touch_buf, seed, n_copy, cudaMemcpyHostToDevice));
    if (n_copy < g_touch_buf_cap) {
      CUDA_GUARD(cudaMemset(g_touch_buf + n_copy, 1, g_touch_buf_cap - n_copy));
    }
  } else {
    CUDA_GUARD(cudaMemset(g_touch_buf, 1, g_touch_buf_cap));
  }

  CUDA_GUARD(cudaMemset(g_touch_acc, 0, sizeof(*g_touch_acc)));

  const int threads = 256;

  int blocks_per_sm = 0;
  CUDA_GUARD(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, ie_cuda_touch_kernel, threads, 0
  ));
  if (blocks_per_sm <= 0) blocks_per_sm = 1;

  cudaDeviceProp prop;
  std::memset(&prop, 0, sizeof(prop));
  CUDA_GUARD(cudaGetDeviceProperties(&prop, 0));
  int sm_count = prop.multiProcessorCount;
  if (sm_count <= 0) sm_count = 1;

  const int grid = blocks_per_sm * sm_count;

  ie_cuda_touch_kernel<<<grid, threads>>>(g_touch_buf, g_touch_buf_cap, stride_bytes, tokens, g_touch_acc);
  CUDA_GUARD(cudaGetLastError());
  CUDA_GUARD(cudaDeviceSynchronize());

  if (verify_touch) {
    unsigned long long host_acc = 0;
    CUDA_GUARD(cudaMemcpy(&host_acc, g_touch_acc, sizeof(host_acc), cudaMemcpyDeviceToHost));
    if (host_acc == 0) {
      ie_cuda_set_last_error_string("CUDA touch verification failed (checksum is zero).");
      return -1;
    }
  }

  ie_cuda_clear_last_error();
  return IE_CUDA_OK;
}
