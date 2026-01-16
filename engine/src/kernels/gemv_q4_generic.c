/* ============================================================================
 * File: engine/src/kernels/gemv_q4_generic.c
 * ============================================================================
 */
/**
 * @file gemv_q4_generic.c
 * @brief Portable Q4_0 GEMV (matrix-vector) kernel with runtime dispatch.
 *
 * @details
 * This module implements a Q4_0 matrix-vector multiply used by the GPT-OSS INT4
 * path. The exported symbol is @ref ie_gemv_q4_0_f32.
 *
 * The implementation provides:
 *  - A portable scalar fallback (always available).
 *  - An AVX2+FMA optimized implementation (compiled in a separate TU) selected
 *    at runtime when supported by the current CPU.
 *
 * The Q4_0 layout matches the engine's weight packing:
 *  - Each block covers 32 columns.
 *  - Blocks store 16 bytes of packed nibbles per 32 weights.
 *  - Each block has one scale (BF16, 2 bytes, or FP8 E4M3, 1 byte).
 *
 * The matrix is represented as two parallel streams:
 *  - W_blocks: rows * (cols/32) * 16 bytes
 *  - W_scales: rows * (cols/32) * scale_bytes
 *
 * Signed 4-bit decode (two's complement):
 *  - n = (nibble & 0xF)
 *  - w = (n >= 8) ? (n - 16) : n  -> range [-8, 7]
 */

#include "ie_kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

#include "ie_cpu.h"

/* Forward declaration for the AVX2 implementation (defined in gemv_q4_avx2.c). */
int ie_gemv_q4_0_f32_avx2_impl(const uint8_t *W_blocks,
                              const uint8_t *W_scales,
                              size_t scale_bytes,
                              const float *x,
                              float *y,
                              size_t rows,
                              size_t cols,
                              const uint16_t *bias_bf16);

/**
 * @brief Convert a BF16 value stored as a uint16_t to float.
 *
 * @param b BF16 bits.
 * @return The corresponding float value.
 */
static inline float ie_bf16_to_f32_scalar(uint16_t b) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = ((uint32_t)b) << 16;
  return v.f;
}

/**
 * @brief Decode log2(u8, q3) scale encoding to float.
 *
 * @details
 * Scale bytes for Q4_0 weights use a log2(u8, q3) encoding:
 *   exp = (v - 128) * 2^-3
 *   scale = 2^exp
 *
 * @param v Encoded scale byte.
 * @return Decoded scale value.
 */
static float ie_log2_u8_q3_lut_[256];
static pthread_once_t ie_log2_u8_q3_once_ = PTHREAD_ONCE_INIT;

static void ie_log2_u8_q3_init_(void) {
  for (int i = 0; i < 256; ++i) {
    const float exp = ((float)(i - 128)) * 0.125f;
    ie_log2_u8_q3_lut_[i] = exp2f(exp);
  }
}

static inline float ie_log2_u8_q3_to_f32(uint8_t v) {
  pthread_once(&ie_log2_u8_q3_once_, ie_log2_u8_q3_init_);
  return ie_log2_u8_q3_lut_[v];
}

/**
 * @brief Load a per-block scale value (BF16 or FP8 E4M3) as float.
 *
 * @param scales Pointer to scale storage for the block.
 * @param scale_bytes Bytes per scale (1 or 2).
 * @return Scale as float.
 */
static inline float ie_q4_load_scale_f32(const uint8_t *scales, size_t scale_bytes) {
  if (scale_bytes == 2) {
    uint16_t b;
    memcpy(&b, scales, sizeof(uint16_t));
    return ie_bf16_to_f32_scalar(b);
  }
  if (scale_bytes == 1) {
    return ie_log2_u8_q3_to_f32(scales[0]);
  }
  return 0.0f;
}

static int ie_q4_debug_nan_enabled_(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *s = getenv("IE_Q4_DEBUG_NAN");
    cached = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
  }
  return cached;
}

/**
 * @brief Compute one row dot product for Q4_0 weights.
 *
 * @param blocks Pointer to packed blocks for the row.
 * @param scales Pointer to per-block scales for the row.
 * @param scale_bytes Bytes per scale (1 or 2).
 * @param x Input vector, length cols.
 * @param cols Number of columns (multiple of 32).
 * @return Dot product result.
 */
static float ie_q4_0_row_dot_f32(const uint8_t *blocks,
                                const uint8_t *scales,
                                size_t scale_bytes,
                                const float *x,
                                size_t cols) {
  const size_t n_blocks = cols / 32;
  float acc = 0.0f;
  const int dbg_nan = ie_q4_debug_nan_enabled_();

  for (size_t b = 0; b < n_blocks; ++b) {
    const uint8_t *q = blocks + b * 16;
    const float d = ie_q4_load_scale_f32(scales + b * scale_bytes, scale_bytes);

    for (size_t j = 0; j < 32; ++j) {
      const uint8_t byte = q[j >> 1];
      const uint8_t nibble = (j & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
      const int32_t w = (nibble >= 8u) ? ((int32_t)nibble - 16) : (int32_t)nibble;
      acc += (d * (float)w) * x[b * 32 + j];
      if (dbg_nan && !isfinite(acc)) {
        fprintf(stderr,
                "[q4_nan] block=%zu j=%zu scale=%g w=%d x=%g acc=%g\n",
                b, j, (double)d, (int)w, (double)x[b * 32 + j], (double)acc);
        return acc;
      }
    }
  }

  return acc;
}

/**
 * @brief Portable scalar Q4_0 GEMV.
 *
 * @param W_blocks Packed blocks pointer.
 * @param W_scales Scales pointer.
 * @param scale_bytes Bytes per scale.
 * @param x Input vector.
 * @param y Output vector.
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param bias_bf16 Optional BF16 bias.
 * @return 0 on success, non-zero on invalid arguments.
 */
static int ie_gemv_q4_0_f32_generic_impl(const uint8_t *W_blocks,
                                        const uint8_t *W_scales,
                                        size_t scale_bytes,
                                        const float *x,
                                        float *y,
                                        size_t rows,
                                        size_t cols,
                                        const uint16_t *bias_bf16) {
  if (!W_blocks || !W_scales || !x || !y) {
    return 1;
  }
  if (cols == 0 || (cols % 32) != 0) {
    return 2;
  }
  if (!(scale_bytes == 1 || scale_bytes == 2)) {
    return 3;
  }

  const size_t n_blocks = cols / 32;
  const size_t row_block_bytes = n_blocks * 16;
  const size_t row_scale_bytes = n_blocks * scale_bytes;

  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *row_blocks = W_blocks + r * row_block_bytes;
    const uint8_t *row_scales = W_scales + r * row_scale_bytes;

    float v = ie_q4_0_row_dot_f32(row_blocks, row_scales, scale_bytes, x, cols);
    if (bias_bf16) {
      v += ie_bf16_to_f32_scalar(bias_bf16[r]);
    }
    y[r] = v;
  }

  return 0;
}

/**
 * @brief Public Q4_0 GEMV entrypoint with one-time runtime dispatch.
 *
 * @details
 * The dispatch decision is cached after the first call. This avoids the need to
 * modify global kernel initialization routines and ensures the symbol always
 * resolves at link time.
 */
int ie_gemv_q4_0_f32(const uint8_t *W_blocks,
                    const uint8_t *W_scales,
                    size_t scale_bytes,
                    const float *x,
                    float *y,
                    size_t rows,
                    size_t cols,
                    const uint16_t *bias_bf16) {
  typedef int (*gemv_fn_t)(const uint8_t *, const uint8_t *, size_t, const float *, float *,
                          size_t, size_t, const uint16_t *);

  static gemv_fn_t fn = NULL;

  if (!fn) {
    ie_cpu_features_t feat;
    ie_cpu_features_detect(&feat);
    const char *force_generic = getenv("IE_Q4_FORCE_GENERIC");
    if (force_generic && force_generic[0] && strcmp(force_generic, "0") != 0) {
      fn = ie_gemv_q4_0_f32_generic_impl;
    } else if (feat.avx2 && feat.fma) {
      fn = ie_gemv_q4_0_f32_avx2_impl;
    } else {
      fn = ie_gemv_q4_0_f32_generic_impl;
    }
  }

  return fn(W_blocks, W_scales, scale_bytes, x, y, rows, cols, bias_bf16);
}
