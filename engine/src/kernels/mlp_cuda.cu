/* ============================================================================
 * File: engine/src/kernels/mlp_cuda.cu
 * ============================================================================
 */
/**
 * @file mlp_cuda.cu
 * @brief Reference CUDA MLP kernels (FP32).
 *
 * Implements a common "SwiGLU" MLP:
 *   gate = W_gate * x + b_gate
 *   up   = W_up   * x + b_up
 *   act  = silu(gate) * up
 *   out  = W_down * act + b_down
 *
 * All matrices are row-major:
 *   W rows are contiguous, W[r, c] at W[r*cols + c].
 *
 * This is correctness-first; caller supplies device buffers tmp_gate/tmp_up.
 */

#include "ie_device_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static __device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void ie_matvec_f32_kernel(const float *W, const float *x, const float *bias,
                                    float *y, size_t rows, size_t cols) {
  size_t r = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (r >= rows) return;

  const float *Wr = W + r * cols;
  float acc = 0.0f;
  for (size_t c = 0; c < cols; ++c) acc += Wr[c] * x[c];
  if (bias) acc += bias[r];
  y[r] = acc;
}

__global__ void ie_swiglu_fuse_kernel(const float *gate, const float *up, float *act, size_t n) {
  size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (i >= n) return;
  act[i] = silu_f32(gate[i]) * up[i];
}

extern "C" int ie_mlp_cuda_swiglu_f32(const float *W_gate, const float *W_up, const float *W_down,
                                     const float *x,
                                     size_t in_dim, size_t hidden_dim, size_t out_dim,
                                     const float *b_gate, const float *b_up, const float *b_down,
                                     float *out,
                                     float *tmp_gate, float *tmp_up) {
  if (!W_gate || !W_up || !W_down || !x || !out || !tmp_gate || !tmp_up) return -1;
  if (in_dim == 0 || hidden_dim == 0 || out_dim == 0) return -1;

  const int threads = 256;
  cudaStream_t s = (cudaStream_t)ie_cuda_get_stream();

  {
    dim3 grid((unsigned int)((hidden_dim + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
    dim3 block((unsigned int)threads, 1u, 1u);
    ie_matvec_f32_kernel<<<grid, block, 0, s ? s : 0>>>(W_gate, x, b_gate, tmp_gate, hidden_dim, in_dim);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -2;
  }

  {
    dim3 grid((unsigned int)((hidden_dim + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
    dim3 block((unsigned int)threads, 1u, 1u);
    ie_matvec_f32_kernel<<<grid, block, 0, s ? s : 0>>>(W_up, x, b_up, tmp_up, hidden_dim, in_dim);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -3;
  }

  {
    dim3 grid((unsigned int)((hidden_dim + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
    dim3 block((unsigned int)threads, 1u, 1u);
    ie_swiglu_fuse_kernel<<<grid, block, 0, s ? s : 0>>>(tmp_gate, tmp_up, tmp_gate, hidden_dim);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -4;
  }

  {
    dim3 grid((unsigned int)((out_dim + (size_t)threads - 1u) / (size_t)threads), 1u, 1u);
    dim3 block((unsigned int)threads, 1u, 1u);
    ie_matvec_f32_kernel<<<grid, block, 0, s ? s : 0>>>(W_down, tmp_gate, b_down, out, out_dim, hidden_dim);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return -5;
  }

  return 0;
}
