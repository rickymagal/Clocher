/**
 * @file act_fp8.c
 * @brief FP8 activation quantization (E4M3 and E5M2).
 *
 * Software implementation of FP8 encoders/decoders to support runtime
 * activation compression (and KV-cache if desired) without external
 * dependencies. The E4M3 variant is treated as finite-only with
 * saturating overflow; E5M2 preserves Inf/NaN where possible.
 */

#include "ie_quant_act.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* ---- bit-level helpers ---- */

typedef union { float f; uint32_t u; } ie_f32u32;

static inline uint32_t ie_u32_from_f(float x) {
  ie_f32u32 v; v.f = x; return v.u;
}
static inline float ie_f_from_u32(uint32_t u) {
  ie_f32u32 v; v.u = u; return v.f;
}

static inline int ie_is_nan_u32(uint32_t u) {
  return ((u & 0x7FFFFFFFu) > 0x7F800000u);
}

static inline int ie_is_inf_u32(uint32_t u) {
  return ((u & 0x7FFFFFFFu) == 0x7F800000u);
}

/* ---- generic encode/decode scaffolding ----
 * We implement a simple normalized encode path:
 * - take absolute float, handle zero and non-finite
 * - extract exponent with frexpf: x = m * 2^e, m in [0.5, 1)
 * - convert to target exponent/mantissa with rounding-to-nearest-even
 * - clamp to range; optional Inf/NaN support for E5M2
 */

static inline uint8_t ie_pack_fp8_e4m3(float x) {
  /* Finite-only E4M3 with bias 7. */
  if (x == 0.0f) return 0u;
  uint32_t u = ie_u32_from_f(x);
  uint8_t sign = (u >> 31) & 0x1;

  if (ie_is_nan_u32(u)) {
    /* Treat NaN as max finite with payload dropped. */
    const uint8_t payload = 0x1; /* minimal mantissa to signal "non-zero" */
    return (uint8_t)((sign << 7) | (0xE << 3) | (payload & 0x7));
  }
  if (ie_is_inf_u32(u)) {
    /* Saturate to max finite. */
    return (uint8_t)((sign << 7) | (0xE << 3) | 0x7);
  }

  float ax = fabsf(x);
  int e2;
  float m = frexpf(ax, &e2); /* ax = m * 2^e2, m in [0.5,1) */

  /* Convert to 1.xxx with mantissa in [1,2) by shifting: */
  float m_shift = m * 2.0f; /* [1,2) */
  int   e = e2 - 1;         /* adjust because of the shift */

  const int bias = 7;
  int exp_fp8 = e + bias;

  /* Round mantissa to 3 bits on [1,2): mant = 1 + frac; q = round(frac * 2^3) */
  float frac = m_shift - 1.0f;
  float qf   = nearbyintf(frac * 8.0f); /* 2^3 = 8 */
  int   qm   = (int)qf;

  if (qm == 8) { /* carry into exponent */
    qm = 0;
    exp_fp8 += 1;
  }

  if (exp_fp8 <= 0) {
    /* Underflow to zero (we skip subnormals for simplicity/perf). */
    return (uint8_t)(sign << 7);
  }
  if (exp_fp8 >= 0xE) {
    /* 0xF would be reserved; clamp to max finite exponent (0xE) and mant=7 */
    return (uint8_t)((sign << 7) | (0xE << 3) | 0x7);
  }

  return (uint8_t)((sign << 7) | ((uint8_t)exp_fp8 << 3) | ((uint8_t)qm & 0x7));
}

static inline float ie_unpack_fp8_e4m3(uint8_t v) {
  if (v == 0u) return 0.0f;
  uint8_t sign = (v >> 7) & 0x1;
  uint8_t exp  = (v >> 3) & 0xF;
  uint8_t man  = (v & 0x7);

  if (exp == 0) {
    /* We do not encode subnormals; interpret as +/- 0. */
    return sign ? -0.0f : 0.0f;
  }
  /* Finite-only: exp==0xF unused; exp==0xE is largest finite. */
  const int bias = 7;
  int e = ((int)exp) - bias;

  float frac = (float)man / 8.0f; /* 2^3 */
  float val  = (1.0f + frac) * ldexpf(1.0f, e); /* (1 + frac) * 2^e */
  return sign ? -val : val;
}

static inline uint8_t ie_pack_fp8_e5m2(float x) {
  /* E5M2 with bias 15; supports Inf/NaN. */
  uint32_t u = ie_u32_from_f(x);
  uint8_t sign = (u >> 31) & 0x1;

  if (x == 0.0f) return (uint8_t)(sign << 7);
  if (ie_is_nan_u32(u)) {
    return (uint8_t)((sign << 7) | (0x1F << 2) | 0x1); /* quiet NaN surrogate */
  }
  if (ie_is_inf_u32(u)) {
    return (uint8_t)((sign << 7) | (0x1F << 2) | 0x0); /* infinity */
  }

  float ax = fabsf(x);
  int e2;
  float m = frexpf(ax, &e2);  /* ax = m * 2^e2, m in [0.5,1) */
  float m_shift = m * 2.0f;   /* [1,2) */
  int   e = e2 - 1;

  const int bias = 15;
  int exp_fp8 = e + bias;

  /* 2 mantissa bits on [1,2): */
  float frac = m_shift - 1.0f;
  float qf   = nearbyintf(frac * 4.0f); /* 2^2 = 4 */
  int   qm   = (int)qf;

  if (qm == 4) { /* carry */
    qm = 0;
    exp_fp8 += 1;
  }

  if (exp_fp8 <= 0) {
    /* Underflow -> flush to zero (skip subnormals). */
    return (uint8_t)(sign << 7);
  }
  if (exp_fp8 >= 0x1F) {
    /* Overflow -> infinity. */
    return (uint8_t)((sign << 7) | (0x1F << 2));
  }

  return (uint8_t)((sign << 7) | ((uint8_t)exp_fp8 << 2) | ((uint8_t)qm & 0x3));
}

static inline float ie_unpack_fp8_e5m2(uint8_t v) {
  uint8_t sign = (v >> 7) & 0x1;
  uint8_t exp  = (v >> 2) & 0x1F;
  uint8_t man  = (v & 0x3);

  if (exp == 0) {
    /* We do not produce subnormals; interpret as +/- 0. */
    return sign ? -0.0f : 0.0f;
  }
  if (exp == 0x1F) {
    if (man == 0) return sign ? -INFINITY : INFINITY;
    /* NaN: return a quiet NaN. */
    volatile float qnan = NAN;
    return qnan;
  }

  const int bias = 15;
  int e = ((int)exp) - bias;

  float frac = (float)man / 4.0f; /* 2^2 */
  float val  = (1.0f + frac) * ldexpf(1.0f, e);
  return sign ? -val : val;
}

/* ---- public array APIs ---- */

void ie_quantize_act_fp8(const float* src, uint8_t* dst, size_t n,
                         ie_fp8_format fmt) {
  if (fmt == IE_FP8_E4M3) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = ie_pack_fp8_e4m3(src[i]);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = ie_pack_fp8_e5m2(src[i]);
    }
  }
}

void ie_dequantize_act_fp8(const uint8_t* src, float* dst, size_t n,
                           ie_fp8_format fmt) {
  if (fmt == IE_FP8_E4M3) {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = ie_unpack_fp8_e4m3(src[i]);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      dst[i] = ie_unpack_fp8_e5m2(src[i]);
    }
  }
}
