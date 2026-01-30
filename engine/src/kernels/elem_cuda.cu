/* ============================================================================
 * File: engine/src/kernels/elem_cuda.cu
 * ============================================================================
 */
/**
 * @file elem_cuda.cu
 * @brief Simple CUDA elementwise kernels (FP32).
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static __device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void ie_zero_f32_kernel(float *y, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < n) y[i] = 0.0f;
}

__global__ void ie_add_inplace_f32_kernel(float *y, const float *x, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < n) y[i] += x[i];
}

__global__ void ie_add_scaled_inplace_f32_kernel(float *y, const float *x, float scale, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < n) y[i] += scale * x[i];
}

__global__ void ie_silu_mul_f32_kernel(const float *gate, const float *up, float *out, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < n) out[i] = silu_f32(gate[i]) * up[i];
}

__global__ void ie_fix_nonfinite_f32_kernel(float *y, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i < n) {
    const float v = y[i];
    if (!isfinite(v)) y[i] = 0.0f;
  }
}

__global__ void ie_rope_f32_kernel(float *q, float *k,
                                  size_t heads, size_t head_dim,
                                  float pos_f, float theta) {
  size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  const size_t pairs = head_dim / 2u;
  const size_t total = heads * pairs;
  if (idx >= total) return;

  const size_t h = idx / pairs;
  const size_t i = idx - h * pairs;
  const size_t j = 2u * i;

  const float di = (float)i;
  const float dh = (float)head_dim;
  const float exponent = (-2.0f * di) / dh;
  const float invf = expf(exponent * logf(theta));
  const float ang = pos_f * invf;
  const float c = cosf(ang);
  const float s = sinf(ang);

  if (q) {
    float *qv = q + h * head_dim;
    const float x0 = qv[j + 0u];
    const float x1 = qv[j + 1u];
    qv[j + 0u] = x0 * c - x1 * s;
    qv[j + 1u] = x0 * s + x1 * c;
  }
  if (k) {
    float *kv = k + h * head_dim;
    const float x0 = kv[j + 0u];
    const float x1 = kv[j + 1u];
    kv[j + 0u] = x0 * c - x1 * s;
    kv[j + 1u] = x0 * s + x1 * c;
  }
}

extern "C" int ie_cuda_zero_f32(float *y, size_t n) {
  if (!y || n == 0u) return -1;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((n + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_zero_f32_kernel<<<grid, block, 0, 0>>>(y, n);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

extern "C" int ie_cuda_add_inplace_f32(float *y, const float *x, size_t n) {
  if (!y || !x || n == 0u) return -1;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((n + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_add_inplace_f32_kernel<<<grid, block, 0, 0>>>(y, x, n);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

extern "C" int ie_cuda_add_scaled_inplace_f32(float *y, const float *x, float scale, size_t n) {
  if (!y || !x || n == 0u) return -1;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((n + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_add_scaled_inplace_f32_kernel<<<grid, block, 0, 0>>>(y, x, scale, n);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

extern "C" int ie_cuda_silu_mul_f32(const float *gate, const float *up, float *out, size_t n) {
  if (!gate || !up || !out || n == 0u) return -1;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((n + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_silu_mul_f32_kernel<<<grid, block, 0, 0>>>(gate, up, out, n);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

extern "C" int ie_cuda_fix_nonfinite_f32(float *y, size_t n) {
  if (!y || n == 0u) return -1;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((n + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_fix_nonfinite_f32_kernel<<<grid, block, 0, 0>>>(y, n);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -2;
}

extern "C" int ie_cuda_rope_f32(float *q, float *k,
                               size_t heads, size_t head_dim,
                               uint32_t pos, float theta, float pos_mul) {
  if ((!q && !k) || heads == 0u) return -1;
  if ((head_dim & 1u) != 0u || head_dim < 2u) return -2;
  if (!(theta > 0.0f)) return -3;
  const float pos_f = (float)pos * pos_mul;
  const size_t pairs = head_dim / 2u;
  const size_t total = heads * pairs;
  const int threads = 256;
  const dim3 block((unsigned int)threads, 1u, 1u);
  const dim3 grid((unsigned int)((total + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
  ie_rope_f32_kernel<<<grid, block, 0, 0>>>(q, k, heads, head_dim, pos_f, theta);
  cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? 0 : -4;
}
