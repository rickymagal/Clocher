/**
 * @file floatx.c
 * @brief FP32↔BF16/FP16 conversion helpers (vector forms).
 */

#include "ie_floatx.h"

#include <math.h>
#include <string.h>

/* ----------------------------- FP32 <-> BF16 ------------------------------ */

/**
 * @brief Convert FP32 → BF16 with round-to-nearest-even.
 *
 * @param in   FP32 input array.
 * @param out  BF16 output array (uint16_t per element).
 * @param n    Number of elements.
 */
void ie_fp32_to_bf16(const float *in, uint16_t *out, size_t n) {
  if (!in || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) {
    union { float f; uint32_t u; } v = { .f = in[i] };
    uint32_t x = v.u;

    /* Round-to-nearest-even on the lower 16 bits. */
    uint32_t lsb = (x >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    x += rounding_bias;

    out[i] = (uint16_t)(x >> 16);
  }
}

/**
 * @brief Convert BF16 → FP32.
 *
 * @param in   BF16 input array (uint16_t per element).
 * @param out  FP32 output array.
 * @param n    Number of elements.
 */
void ie_bf16_to_fp32(const uint16_t *in, float *out, size_t n) {
  if (!in || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)in[i]) << 16;
    out[i] = v.f;
  }
}

/* ----------------------------- FP32 <-> FP16 ------------------------------ */

/* Helpers adapted from common IEEE-754 half conversions. */

/**
 * @brief Convert FP32 → FP16 (IEEE 754 half) round-to-nearest-even.
 *
 * @param in   FP32 input array.
 * @param out  FP16 output array (uint16_t per element).
 * @param n    Number of elements.
 */
void ie_fp32_to_fp16(const float *in, uint16_t *out, size_t n) {
  if (!in || !out || n == 0) return;

  for (size_t i = 0; i < n; ++i) {
    union { float f; uint32_t u; } v = { .f = in[i] };
    uint32_t x = v.u;

    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127;   /* unbiased exp */
    uint32_t frac = x & 0x7FFFFFu;                        /* 23 bits */

    uint16_t h;
    if (exp == 128) { /* Inf / NaN */
      h = (uint16_t)((sign << 15) | 0x7C00u | (frac ? 0x200u : 0));
    } else if (exp > 15) { /* Overflow -> Inf */
      h = (uint16_t)((sign << 15) | 0x7C00u);
    } else if (exp >= -14) { /* Normalized half */
      /* Put implicit leading 1 back and round to 10 bits. */
      uint32_t mant = (frac | 0x800000u); /* 24 bits */
      int shift = 13 - exp;               /* align to 10-bit mantissa */
      uint32_t rounding = (shift > 0) ? ((1u << (shift - 1)) - 1u) : 0;
      uint32_t half_mant = (shift > 0) ? ((mant + rounding + ((mant >> shift) & 1u)) >> shift)
                                       : (mant << (-shift));
      if (half_mant & 0x400u) { /* mantissa overflow after rounding */
        half_mant >>= 1;
        exp += 1;
      }
      if (exp > 15) { /* overflow to Inf */
        h = (uint16_t)((sign << 15) | 0x7C00u);
      } else {
        uint16_t he = (uint16_t)((exp + 15) & 0x1Fu);
        h = (uint16_t)((sign << 15) | (he << 10) | (half_mant & 0x3FFu));
      }
    } else if (exp >= -24) { /* Subnormal half */
      uint32_t mant = (frac | 0x800000u);
      int rshift = (-14 - exp) + 1; /* +1 to position for subnormal */
      uint32_t sub = mant >> (rshift + 13); /* to 10 bits */
      /* round-to-nearest-even */
      uint32_t rem = mant & ((1u << (rshift + 13)) - 1u);
      uint32_t halfway = 1u << (rshift + 12);
      if (rem > halfway || (rem == halfway && (sub & 1u))) sub++;
      h = (uint16_t)((sign << 15) | (sub & 0x3FFu));
    } else {
      /* Underflow -> signed zero */
      h = (uint16_t)(sign << 15);
    }

    out[i] = h;
  }
}

/**
 * @brief Convert FP16 (IEEE 754 half) → FP32.
 *
 * @param in   FP16 input array (uint16_t per element).
 * @param out  FP32 output array.
 * @param n    Number of elements.
 */
void ie_fp16_to_fp32(const uint16_t *in, float *out, size_t n) {
  if (!in || !out || n == 0) return;

  for (size_t i = 0; i < n; ++i) {
    uint16_t h = in[i];
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;

    uint32_t out_sign = sign << 31;
    uint32_t out_exp, out_frac;

    if (exp == 0) {
      if (frac == 0) {
        out_exp = 0;
        out_frac = 0;
      } else {
        /* subnormal -> normalize */
        int e = -14;
        uint32_t f = frac;
        while ((f & 0x400u) == 0) { f <<= 1; e--; }
        f &= 0x3FFu;
        out_exp  = (uint32_t)(e + 127) << 23;
        out_frac = f << 13;
      }
    } else if (exp == 31) {
      out_exp  = 0xFFu << 23;
      out_frac = (frac ? (frac << 13) : 0);
    } else {
      out_exp  = (uint32_t)(exp - 15 + 127) << 23;
      out_frac = frac << 13;
    }

    union { uint32_t u; float f; } v = { .u = out_sign | out_exp | out_frac };
    out[i] = v.f;
  }
}
