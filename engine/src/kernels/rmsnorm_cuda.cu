/* ============================================================================
 * File: engine/src/kernels/rmsnorm_cuda.cu
 * ============================================================================
 */
/**
 * @file rmsnorm_cuda.cu
 * @brief Reference CUDA RMSNorm (FP32).
 *
 * For each row:
 *   rms = sqrt(mean(x^2) + eps)
 *   y   = (x / rms) * w
 *
 * Layout: x and y are [rows, cols] row-major. w is [cols].
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static __device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
  return v;
}

static __device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[32];
  __shared__ float block_sum;
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  float out = 0.0f;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
    out = warp_reduce_sum(out);
    if (lane == 0) block_sum = out;
  }
  __syncthreads();
  return block_sum;
}

__global__ void ie_rmsnorm_f32_kernel(const float *x, const float *w, float *y,
                                     size_t rows, size_t cols, float eps) {
  const size_t r = (size_t)blockIdx.x;
  if (r >= rows) return;

  const float *xr = x + r * cols;
  float *yr = y + r * cols;

  float local = 0.0f;
  for (size_t c = (size_t)threadIdx.x; c < cols; c += (size_t)blockDim.x) {
    float v = xr[c];
    local += v * v;
  }
  float sumsq = block_reduce_sum(local);
  float mean = sumsq / (float)cols;
  float inv_rms = rsqrtf(mean + eps);

  for (size_t c = (size_t)threadIdx.x; c < cols; c += (size_t)blockDim.x) {
    float v = xr[c] * inv_rms;
    float ww = w ? w[c] : 1.0f;
    yr[c] = v * ww;
  }
}

extern "C" int ie_rmsnorm_cuda_f32(const float *x, const float *w, float *y,
                                  size_t rows, size_t cols, float eps) {
  if (!x || !y) return -1;
  if (rows == 0 || cols == 0) return -1;
  if (eps <= 0.0f) eps = 1e-6f;

  const int threads = 256;
  dim3 grid((unsigned int)rows, 1u, 1u);
  dim3 block((unsigned int)threads, 1u, 1u);

  ie_rmsnorm_f32_kernel<<<grid, block, 0, 0>>>(x, w, y, rows, cols, eps);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return -2;
  return 0;
}
