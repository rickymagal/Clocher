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
 *
 * RoPE scaling:
 * - This module supports optional position scaling inside the RoPE helper.
 * - Set exactly one of the environment variables below:
 *     - IE_ROPE_LINEAR_SCALE:  a positive float factor f. Effective pos = pos / f.
 *     - IE_ROPE_POS_SCALE:     a positive float multiplier m. Effective pos = pos * m.
 * - If neither is set, scaling defaults to 1.0 (vanilla RoPE).
 *
 * Notes:
 * - Scaling is applied by modifying the effective position used to compute angles,
 *   keeping all callers unchanged.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_rope.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* ------------------------------------------------------------------------- */
/* Internal configuration                                                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Cached position multiplier for RoPE angle computation.
 *
 * @details
 * If set to 1.0, RoPE is vanilla.
 * If set to m != 1.0, RoPE uses effective_pos = pos * m.
 *
 * This is initialized lazily from environment variables to keep callers simple.
 */
static float ie_rope_pos_mul_cached = -1.0f;

/**
 * @brief Parse a positive float from an environment variable.
 *
 * @param name Environment variable name.
 * @param out  Output value on success.
 * @return 1 if parsed successfully and value is positive, 0 otherwise.
 */
static int ie_rope_env_pos_float(const char *name, float *out) {
  if (!name || !out) return 0;

  const char *s = getenv(name);
  if (!s || !*s) return 0;

  char *end = NULL;
  const double v = strtod(s, &end);
  if (end == s) return 0;
  if (!(v > 0.0)) return 0;

  *out = (float)v;
  return 1;
}

/**
 * @brief Get the effective RoPE position multiplier.
 *
 * @details
 * Resolution order:
 *  1) IE_ROPE_POS_SCALE:     multiplier m, effective_pos = pos * m
 *  2) IE_ROPE_LINEAR_SCALE:  factor f, effective_pos = pos / f
 *  3) default:              1.0
 *
 * @return Position multiplier to apply to @p pos.
 */
float ie_rope_pos_mul(void) {
  if (ie_rope_pos_mul_cached > 0.0f) return ie_rope_pos_mul_cached;

  float m = 1.0f;

  float pos_scale = 0.0f;
  if (ie_rope_env_pos_float("IE_ROPE_POS_SCALE", &pos_scale)) {
    m = pos_scale;
  } else {
    float linear_scale = 0.0f;
    if (ie_rope_env_pos_float("IE_ROPE_LINEAR_SCALE", &linear_scale)) {
      m = 1.0f / linear_scale;
    }
  }

  if (!(m > 0.0f)) m = 1.0f;
  ie_rope_pos_mul_cached = m;
  return m;
}

/* ------------------------------------------------------------------------- */
/* Math helpers                                                               */
/* ------------------------------------------------------------------------- */

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

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

int ie_rope_apply_one_f32(float *x, size_t head_dim, uint32_t pos, float theta) {
  if (!x) return -1;
  if (head_dim < 2u) return -2;
  if ((head_dim & 1u) != 0u) return -3;
  if (!(theta > 0.0f)) return -4;

  const size_t pairs = head_dim / 2u;

  /* Apply optional position scaling inside the helper. */
  const float p = (float)pos * ie_rope_pos_mul();

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
      const int rc = ie_rope_apply_one_f32(q + h * head_dim, head_dim, pos, theta);
      if (rc != 0) return rc;
    }
    if (k) {
      const int rc = ie_rope_apply_one_f32(k + h * head_dim, head_dim, pos, theta);
      if (rc != 0) return rc;
    }
  }

  return 0;
}
