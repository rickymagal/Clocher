/* ============================================================================
 * File: engine/src/kernels/attn_cpu.c
 * ============================================================================
 */
/**
 * @file attn_cpu.c
 * @brief Reference CPU attention kernel (FP32) for causal self-attention.
 *
 * Computes, per head:
 *   scores[t] = dot(Q[h,:], K[t,h,:]) * inv_sqrt_d
 *   w = softmax(scores[0..seq_len-1])
 *   out[h,:] = sum_t w[t] * V[t,h,:]
 *
 * Layout for Q/out: [heads, head_dim]
 * Layout for K/V cache: [seq_len, heads, head_dim] flattened with head_dim fastest:
 *   index(t,h,d) = ((t * heads + h) * head_dim + d)
 *
 * This implementation is allocation-free: caller supplies a scratch buffer of
 * length seq_len floats, reused per head.
 */

#include "ie_kernels.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static inline size_t kv_index(size_t t, size_t h, size_t d, size_t H, size_t D) {
  return ((t * H) + h) * D + d;
}

static float dot_f32(const float *a, const float *b, size_t n) {
  float acc = 0.0f;
  for (size_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

static void softmax_inplace_f32(float *x, size_t n) {
  if (!x || n == 0) return;

  float mx = x[0];
  for (size_t i = 1; i < n; ++i) {
    if (x[i] > mx) mx = x[i];
  }

  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float v = expf(x[i] - mx);
    x[i] = v;
    sum += v;
  }

  if (sum > 0.0f) {
    const float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
  } else {
    const float inv = 1.0f / (float)n;
    for (size_t i = 0; i < n; ++i) x[i] = inv;
  }
}

/**
 * @brief Causal attention forward (FP32).
 *
 * @param Q           Pointer to Q slice [heads*head_dim].
 * @param K           Pointer to K cache [seq_len*heads*head_dim].
 * @param V           Pointer to V cache [seq_len*heads*head_dim].
 * @param seq_len     Number of cached tokens to attend over (>= 1).
 * @param heads       Number of heads (>= 1).
 * @param head_dim    Dimension per head (>= 1).
 * @param inv_sqrt_d  Scale factor (typically 1/sqrt(head_dim)).
 * @param out         Output buffer [heads*head_dim].
 * @param scratch     Scratch buffer [seq_len] floats (reused per head).
 * @return 0 on success, non-zero on invalid args.
 */
int ie_attn_cpu_causal_f32(const float *Q, const float *K, const float *V,
                           size_t seq_len, size_t heads, size_t head_dim,
                           float inv_sqrt_d, float *out, float *scratch) {
  if (!Q || !K || !V || !out || !scratch) return -1;
  if (seq_len == 0 || heads == 0 || head_dim == 0) return -1;

  if (inv_sqrt_d == 0.0f) inv_sqrt_d = 1.0f;

  memset(out, 0, heads * head_dim * sizeof(float));

  for (size_t h = 0; h < heads; ++h) {
    const float *Qh = Q + h * head_dim;

    for (size_t t = 0; t < seq_len; ++t) {
      const float *Kh = K + kv_index(t, h, 0, heads, head_dim);
      scratch[t] = dot_f32(Qh, Kh, head_dim) * inv_sqrt_d;
    }

    softmax_inplace_f32(scratch, seq_len);

    float *Oh = out + h * head_dim;
    for (size_t t = 0; t < seq_len; ++t) {
      const float w = scratch[t];
      const float *Vh = V + kv_index(t, h, 0, heads, head_dim);
      for (size_t d = 0; d < head_dim; ++d) {
        Oh[d] += w * Vh[d];
      }
    }
  }

  return 0;
}
