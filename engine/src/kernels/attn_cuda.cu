/* ============================================================================
 * File: engine/src/kernels/attn_cuda.cu
 * ============================================================================
 */
/**
 * @file attn_cuda.cu
 * @brief Reference CUDA attention kernel (FP32) for causal self-attention.
 *
 * Computes, per head:
 *   scores[t] = dot(Q[h,:], K[t,h,:]) * inv_sqrt_d
 *   out[h,:]  = sum_t softmax(scores)[t] * V[t,h,:]
 *
 * Layout for Q/out: [heads, head_dim]
 * Layout for K/V cache: [seq_len, heads, head_dim] flattened with head_dim fastest:
 *   index(t,h,d) = ((t * heads + h) * head_dim + d)
 *
 * This is a correctness-first implementation for a single-step decode attention.
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static __device__ __forceinline__ size_t kv_index(size_t t, size_t h, size_t d, size_t H, size_t D) {
  return ((t * H) + h) * D + d;
}

static __device__ __forceinline__ float warp_reduce_max(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float o = __shfl_down_sync(0xFFFFFFFFu, v, offset);
    v = fmaxf(v, o);
  }
  return v;
}

static __device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
  }
  return v;
}

static __device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warp_reduce_max(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  float out = -INFINITY;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane] : -INFINITY;
    out = warp_reduce_max(out);
  }
  return __shfl_sync(0xFFFFFFFFu, out, 0);
}

static __device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) smem[wid] = v;
  __syncthreads();

  float out = 0.0f;
  if (wid == 0) {
    out = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
    out = warp_reduce_sum(out);
  }
  return __shfl_sync(0xFFFFFFFFu, out, 0);
}

/**
 * One block per head.
 * Threads cooperate over tokens t in [0, seq_len).
 * Three passes:
 *  1) max score
 *  2) sum exp(score-max)
 *  3) accumulate weighted V
 */
__global__ void ie_attn_causal_f32_kernel(const float *Q,
                                         const float *K,
                                         const float *V,
                                         size_t seq_len,
                                         size_t heads,
                                         size_t head_dim,
                                         float inv_sqrt_d,
                                         float *out) {
  const size_t h = (size_t)blockIdx.x;
  if (h >= heads) return;

  const float *Qh = Q + h * head_dim;

  float local_max = -INFINITY;
  for (size_t t = (size_t)threadIdx.x; t < seq_len; t += (size_t)blockDim.x) {
    const float *Kh = K + kv_index(t, h, 0, heads, head_dim);
    float acc = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) acc += Qh[d] * Kh[d];
    acc *= inv_sqrt_d;
    local_max = fmaxf(local_max, acc);
  }
  const float mx = block_reduce_max(local_max);

  float local_sum = 0.0f;
  for (size_t t = (size_t)threadIdx.x; t < seq_len; t += (size_t)blockDim.x) {
    const float *Kh = K + kv_index(t, h, 0, heads, head_dim);
    float acc = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) acc += Qh[d] * Kh[d];
    acc *= inv_sqrt_d;
    local_sum += expf(acc - mx);
  }
  const float denom = block_reduce_sum(local_sum);

  for (size_t d = (size_t)threadIdx.x; d < head_dim; d += (size_t)blockDim.x) {
    float acc_out = 0.0f;
    for (size_t t = 0; t < seq_len; ++t) {
      const float *Kh = K + kv_index(t, h, 0, heads, head_dim);
      float s = 0.0f;
      for (size_t j = 0; j < head_dim; ++j) s += Qh[j] * Kh[j];
      s *= inv_sqrt_d;
      const float w = (denom > 0.0f) ? (expf(s - mx) / denom) : (1.0f / (float)seq_len);
      const float *Vh = V + kv_index(t, h, 0, heads, head_dim);
      acc_out += w * Vh[d];
    }
    out[h * head_dim + d] = acc_out;
  }
}

extern "C" int ie_attn_cuda_causal_f32(const float *Q,
                                      const float *K,
                                      const float *V,
                                      size_t seq_len,
                                      size_t heads,
                                      size_t head_dim,
                                      float inv_sqrt_d,
                                      float *out) {
  if (!Q || !K || !V || !out) return -1;
  if (seq_len == 0 || heads == 0 || head_dim == 0) return -1;
  if (inv_sqrt_d == 0.0f) inv_sqrt_d = 1.0f;

  const int threads = 256;
  dim3 grid((unsigned int)heads, 1u, 1u);
  dim3 block((unsigned int)threads, 1u, 1u);

  ie_attn_causal_f32_kernel<<<grid, block, 0, 0>>>(Q, K, V, seq_len, heads, head_dim, inv_sqrt_d, out);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return -2;
  return 0;
}
