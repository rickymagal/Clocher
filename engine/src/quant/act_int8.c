/**
 * @file act_int8.c
 * @brief INT8 activation quantization (per-tensor and per-group).
 *
 * This file implements fast saturating affine INT8 quantization paths
 * for activations. The API supports both per-tensor and per-group
 * parameterization with symmetric (zero_point=0) and asymmetric modes.
 */

#include "ie_quant_act.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* --------- small internal helpers (header-local) --------- */

static inline float ie_fabsf(float x) {
  return x < 0.0f ? -x : x;
}

static inline float ie_maxf(float a, float b) {
  return a > b ? a : b;
}

static inline float ie_minf(float a, float b) {
  return a < b ? a : b;
}

/* Round-to-nearest-even for better statistical behavior. */
static inline float ie_rne(float x) {
  /* Using nearbyintf honors current rounding mode; default is ties-to-even. */
  return nearbyintf(x);
}

static inline int8_t ie_sat_i8(int v) {
  if (v > 127) return 127;
  if (v < -128) return -128;
  return (int8_t)v;
}

/* Symmetric INT8 usually avoids -128 to keep symmetry. */
static inline int8_t ie_sat_sym_i8(int v) {
  if (v > 127) return 127;
  if (v < -127) return -127;
  return (int8_t)v;
}

/* --------- public API --------- */

void ie_act_i8_params_from_minmax(float minv, float maxv, int symmetric,
                                  float* out_scale, int8_t* out_zero) {
  /* Ensure valid range; if degenerate, pick unit scale. */
  if (!(maxv > minv)) {
    *out_scale = 1.0f;
    *out_zero  = 0;
    return;
  }

  if (symmetric) {
    const float amax = ie_maxf(ie_fabsf(minv), ie_fabsf(maxv));
    /* Map amax -> 127, avoid divide-by-zero. */
    const float denom = (amax > 0.0f) ? amax : 1.0f;
    *out_scale = denom / 127.0f;
    *out_zero  = 0;
  } else {
    /* Map minv -> -128, maxv -> 127 */
    const float qrange = 127.0f - (-128.0f); /* 255 */
    const float scale  = (maxv - minv) / qrange;
    const float zp_f   = -(-128.0f) - (minv / (scale > 0.0f ? scale : 1.0f));
    /* Round and clamp zero point into INT8 range. */
    int zp_i = (int)ie_rne(zp_f);
    if (zp_i < -128) zp_i = -128;
    if (zp_i > 127)  zp_i = 127;

    *out_scale = (scale > 0.0f ? scale : 1.0f);
    *out_zero  = (int8_t)zp_i;
  }
}

void ie_quantize_act_int8(const float* src, int8_t* dst, size_t n,
                          ie_act_i8_params params, int symmetric) {
  const float inv_scale = (params.scale > 0.0f) ? (1.0f / params.scale) : 1.0f;
  if (symmetric) {
    for (size_t i = 0; i < n; ++i) {
      const float v = src[i] * inv_scale;
      const int   q = (int)ie_rne(v);
      dst[i] = ie_sat_sym_i8(q);
    }
  } else {
    const int z = (int)params.zero_point;
    for (size_t i = 0; i < n; ++i) {
      const float v = src[i] * inv_scale + (float)z;
      const int   q = (int)ie_rne(v);
      dst[i] = ie_sat_i8(q);
    }
  }
}

void ie_dequantize_act_int8(const int8_t* src, float* dst, size_t n,
                            ie_act_i8_params params) {
  const float s = params.scale;
  const int   z = (int)params.zero_point;
  for (size_t i = 0; i < n; ++i) {
    dst[i] = s * ((int)src[i] - z);
  }
}

void ie_act_i8_group_params_from_data(const float* src, size_t n,
                                      size_t group_size, int symmetric,
                                      float* out_scales, int8_t* out_zeros) {
  if (group_size == 0) return;
  const size_t groups = (n + group_size - 1) / group_size;
  for (size_t g = 0; g < groups; ++g) {
    const size_t start = g * group_size;
    const size_t end   = (start + group_size <= n) ? (start + group_size) : n;
    float mn = src[start];
    float mx = src[start];
    for (size_t i = start + 1; i < end; ++i) {
      mn = ie_minf(mn, src[i]);
      mx = ie_maxf(mx, src[i]);
    }
    float s; int8_t z;
    ie_act_i8_params_from_minmax(mn, mx, symmetric, &s, &z);
    out_scales[g] = s;
    out_zeros[g]  = z;
  }
}

void ie_quantize_act_int8_per_group(const float* src, int8_t* dst, size_t n,
                                    size_t group_size,
                                    const float* scales, const int8_t* zeros,
                                    int symmetric) {
  if (group_size == 0) return;
  const size_t groups = (n + group_size - 1) / group_size;
  for (size_t g = 0; g < groups; ++g) {
    const size_t start = g * group_size;
    const size_t end   = (start + group_size <= n) ? (start + group_size) : n;
    const float inv_s  = (scales[g] > 0.0f) ? 1.0f / scales[g] : 1.0f;
    if (symmetric) {
      for (size_t i = start; i < end; ++i) {
        const int q = (int)ie_rne(src[i] * inv_s);
        dst[i] = ie_sat_sym_i8(q);
      }
    } else {
      const int z = (int)zeros[g];
      for (size_t i = start; i < end; ++i) {
        const int q = (int)ie_rne(src[i] * inv_s + (float)z);
        dst[i] = ie_sat_i8(q);
      }
    }
  }
}

void ie_dequantize_act_int8_per_group(const int8_t* src, float* dst, size_t n,
                                      size_t group_size,
                                      const float* scales, const int8_t* zeros) {
  if (group_size == 0) return;
  const size_t groups = (n + group_size - 1) / group_size;
  for (size_t g = 0; g < groups; ++g) {
    const size_t start = g * group_size;
    const size_t end   = (start + group_size <= n) ? (start + group_size) : n;
    const float s = scales[g];
    const int   z = (int)zeros[g];
    for (size_t i = start; i < end; ++i) {
      dst[i] = s * ((int)src[i] - z);
    }
  }
}
