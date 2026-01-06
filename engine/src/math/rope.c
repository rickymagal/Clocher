/* ============================================================================
 * File: engine/src/math/rope.c
 * ============================================================================
 */
/**
 * @file rope.c
 * @brief Rotary Position Embedding (RoPE) fp32 implementation.
 *
 * @details
 * This file implements standard RoPE rotation for interleaved pairs of channels.
 * It is intended as a correctness-first implementation for the initial CPU
 * forward pass.
 *
 * The rotation is applied per pair (2*i, 2*i+1) using:
 *
 *   inv_freq(i) = theta^(-2i / head_dim)
 *   angle       = pos * inv_freq(i)
 *   [y0] = [ cos -sin ] [x0]
 *   [y1]   [ sin  cos ] [x1]
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_rope.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Compute theta^(-2i/head_dim) as a float.
 *
 * @details
 * This helper uses exp/log for stability across a wide range of head sizes.
 *
 * @param theta    RoPE base theta.
 * @param i        Pair index (0-based).
 * @param head_dim Per-head dimension.
 * @return Inverse frequency for the given pair index.
 */
static float ie_rope_inv_freq(float theta, size_t i, size_t head_dim) {
  if (!(theta > 0.0f) || head_dim == 0u) return 0.0f;

  const float di = (float)i;
  const float dh = (float)head_dim;
  const float exponent = (-2.0f * di) / dh;

  return expf(exponent * logf(theta));
}

int ie_rope_apply_one_f32(float *x, size_t head_dim, uint32_t pos, float theta) {
  if (!x) return -1;
  if (head_dim < 2u) return -2;
  if ((head_dim & 1u) != 0u) return -3;
  if (!(theta > 0.0f)) return -4;

  const size_t pairs = head_dim / 2u;
  const float p = (float)pos;

  for (size_t i = 0; i < pairs; ++i) {
    const size_t j = 2u * i;

    const float invf = ie_rope_inv_freq(theta, i, head_dim);
    const float ang = p * invf;

    const float c = cosf(ang);
    const float s = sinf(ang);

    const float x0 = x[j + 0u];
    const float x1 = x[j + 1u];

    x[j + 0u] = x0 * c - x1 * s;
    x[j + 1u] = x0 * s + x1 * c;
  }

  return 0;
}

int ie_rope_apply_f32(float *q, float *k, size_t heads, size_t head_dim, uint32_t pos,
                      float theta) {
  if (!q && !k) return 0;
  if (heads == 0u) return -1;

  if ((head_dim & 1u) != 0u) return -2;
  if (head_dim < 2u) return -3;
  if (!(theta > 0.0f)) return -4;

  for (size_t h = 0; h < heads; ++h) {
    if (q) {
      int rc = ie_rope_apply_one_f32(q + h * head_dim, head_dim, pos, theta);
      if (rc != 0) return rc;
    }
    if (k) {
      int rc = ie_rope_apply_one_f32(k + h * head_dim, head_dim, pos, theta);
      if (rc != 0) return rc;
    }
  }

  return 0;
}
