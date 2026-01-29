/**
 * @file ie_elem_cuda.h
 * @brief C ABI wrappers for simple CUDA elementwise kernels (FP32).
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief y[i] = 0 for i in [0, n).
 */
int ie_cuda_zero_f32(float *y, size_t n);

/**
 * @brief y[i] += x[i] for i in [0, n).
 */
int ie_cuda_add_inplace_f32(float *y, const float *x, size_t n);

/**
 * @brief y[i] += scale * x[i] for i in [0, n).
 */
int ie_cuda_add_scaled_inplace_f32(float *y, const float *x, float scale, size_t n);

/**
 * @brief out[i] = silu(gate[i]) * up[i] for i in [0, n).
 */
int ie_cuda_silu_mul_f32(const float *gate, const float *up, float *out, size_t n);

/**
 * @brief Apply RoPE to q/k on device (in-place).
 *
 * @param q        Device pointer to Q (may be NULL).
 * @param k        Device pointer to K (may be NULL).
 * @param heads    Number of heads.
 * @param head_dim Per-head dimension (even).
 * @param pos      Token position.
 * @param theta    RoPE base theta.
 * @param pos_mul  Position multiplier (matches ie_rope_pos_mul()).
 */
int ie_cuda_rope_f32(float *q, float *k,
                     size_t heads, size_t head_dim,
                     uint32_t pos, float theta, float pos_mul);

#ifdef __cplusplus
} /* extern "C" */
#endif
