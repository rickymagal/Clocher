/* ============================================================================
 * File: engine/src/kernels/gemv_q4_avx2.c
 * ============================================================================
 */
/**
 * @file gemv_q4_avx2.c
 * @brief AVX2+FMA Q4_0 GEMV kernel.
 *
 * @details
 * This module implements an AVX2+FMA optimized variant of the Q4_0 matrix-vector
 * multiply used by the GPT-OSS INT4 path.
 *
 * The exported symbol is @ref ie_gemv_q4_0_f32_avx2_impl. The generic module
 * selects this implementation at runtime when the CPU supports AVX2 and FMA.
 *
 * Correctness model:
 *  - We unpack packed 4-bit weights into signed int8 values in [-8, 7].
 *  - We multiply by a per-block scale (BF16 or FP8 E4M3).
 *  - We compute a dot product against an FP32 input vector.
 */

#include "ie_kernels.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include <immintrin.h>

#ifndef IE_TARGET_AVX2_FMA
#if defined(__GNUC__) || defined(__clang__)
#define IE_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#define IE_TARGET_AVX2_FMA
#endif
#endif

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
static int ie_log2_u8_q3_ready_ = 0;

static void ie_log2_u8_q3_init_impl_(void) {
  for (int i = 0; i < 256; ++i) {
    const float exp = ((float)(i - 128)) * 0.125f;
    ie_log2_u8_q3_lut_[i] = exp2f(exp);
  }
  ie_log2_u8_q3_ready_ = 1;
}

void ie_q4_log2_u8_q3_init_avx2(void) {
  if (!ie_log2_u8_q3_ready_) {
    ie_log2_u8_q3_init_impl_();
  }
}

static inline float ie_log2_u8_q3_to_f32(uint8_t v) {
  if (!ie_log2_u8_q3_ready_) {
    ie_log2_u8_q3_init_impl_();
  }
  return ie_log2_u8_q3_lut_[v];
}

static inline float ie_log2_u8_q3_to_f32_fast(uint8_t v) {
  /* Assumes ie_log2_u8_q3_init_avx2() already ran. */
  return ie_log2_u8_q3_lut_[v];
}

static int ie_q4_use_aligned_loads(void) {
  static int cached = -1;
  if (cached >= 0) return cached;
  const char *s = getenv("IE_Q4_ALIGNED_LOADS");
  cached = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
  return cached;
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
  return ie_log2_u8_q3_to_f32(scales[0]);
}

/**
 * @brief Horizontal sum of an AVX2 register.
 *
 * @param v Input vector.
 * @return Sum of all 8 lanes.
 */
static IE_TARGET_AVX2_FMA inline float ie_hsum256_ps(__m256 v) {
  /* Horizontal sum without spilling to memory. */
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

/**
 * @brief Compute a dot product for one Q4_0 block (32 weights).
 *
 * @details
 * This uses the following unpack strategy:
 *  - Load 16 bytes packed nibbles.
 *  - Extract low and high nibbles.
 *  - Use pshufb with a 16-byte LUT to map nibbles -> signed weights [-8..7].
 *  - Interleave even/odd weights to obtain w0..w31 in order.
 *  - Convert 8 weights at a time to float and accumulate.
 *
 * @param q Pointer to 16 packed bytes.
 * @param x Pointer to 32 floats.
 * @param scale Block scale.
 * @return Dot(q, x) scaled.
 */
static IE_TARGET_AVX2_FMA float ie_q4_0_block_dot_f32_avx2(const uint8_t *q,
                                                          const float *x,
                                                          float scale) {
  const __m128i v = _mm_loadu_si128((const __m128i *)q);

  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(v, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);

  const __m128i lut =
      _mm_setr_epi8((char)0, (char)1, (char)2, (char)3, (char)4, (char)5, (char)6, (char)7,
                    (char)-8, (char)-7, (char)-6, (char)-5, (char)-4, (char)-3, (char)-2, (char)-1);

  const __m128i w_even = _mm_shuffle_epi8(lut, lo);
  const __m128i w_odd = _mm_shuffle_epi8(lut, hi);

  const __m128i w0_15 = _mm_unpacklo_epi8(w_even, w_odd);
  const __m128i w16_31 = _mm_unpackhi_epi8(w_even, w_odd);

  __m256 acc = _mm256_setzero_ps();

  /* 0..7 */
  {
    const __m128i w8 = _mm_cvtepi8_epi16(w0_15);
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_loadu_ps(x + 0);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  /* 8..15 */
  {
    const __m128i w8 = _mm_cvtepi8_epi16(_mm_srli_si128(w0_15, 8));
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_loadu_ps(x + 8);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  /* 16..23 */
  {
    const __m128i w8 = _mm_cvtepi8_epi16(w16_31);
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_loadu_ps(x + 16);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  /* 24..31 */
  {
    const __m128i w8 = _mm_cvtepi8_epi16(_mm_srli_si128(w16_31, 8));
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_loadu_ps(x + 24);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  return scale * ie_hsum256_ps(acc);
}

static IE_TARGET_AVX2_FMA float ie_q4_0_block_dot_f32_avx2_aligned(const uint8_t *q,
                                                                   const float *x,
                                                                   float scale) {
  const __m128i v = _mm_loadu_si128((const __m128i *)q);

  const __m128i mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(v, mask);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);

  const __m128i lut =
      _mm_setr_epi8((char)0, (char)1, (char)2, (char)3, (char)4, (char)5, (char)6, (char)7,
                    (char)-8, (char)-7, (char)-6, (char)-5, (char)-4, (char)-3, (char)-2, (char)-1);

  const __m128i w_even = _mm_shuffle_epi8(lut, lo);
  const __m128i w_odd = _mm_shuffle_epi8(lut, hi);

  const __m128i w0_15 = _mm_unpacklo_epi8(w_even, w_odd);
  const __m128i w16_31 = _mm_unpackhi_epi8(w_even, w_odd);

  __m256 acc = _mm256_setzero_ps();

  {
    const __m128i w8 = _mm_cvtepi8_epi16(w0_15);
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_load_ps(x + 0);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  {
    const __m128i w8 = _mm_cvtepi8_epi16(_mm_srli_si128(w0_15, 8));
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_load_ps(x + 8);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  {
    const __m128i w8 = _mm_cvtepi8_epi16(w16_31);
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_load_ps(x + 16);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  {
    const __m128i w8 = _mm_cvtepi8_epi16(_mm_srli_si128(w16_31, 8));
    const __m256i wi32 = _mm256_cvtepi16_epi32(w8);
    const __m256 wf = _mm256_cvtepi32_ps(wi32);
    const __m256 xs = _mm256_load_ps(x + 24);
    acc = _mm256_fmadd_ps(wf, xs, acc);
  }

  return scale * ie_hsum256_ps(acc);
}

/**
 * @brief AVX2+FMA implementation of @ref ie_gemv_q4_0_f32.
 *
 * @param W_blocks Packed blocks pointer.
 * @param W_scales Scales pointer.
 * @param scale_bytes Bytes per scale (1 or 2).
 * @param x Input vector.
 * @param y Output vector.
 * @param rows Number of rows.
 * @param cols Number of columns (multiple of 32).
 * @param bias_bf16 Optional BF16 bias.
 * @return 0 on success, non-zero on invalid arguments.
 */
IE_TARGET_AVX2_FMA int ie_gemv_q4_0_f32_avx2_impl(const uint8_t *W_blocks,
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

  const int x_aligned = ((((uintptr_t)x) & 31u) == 0u) && ie_q4_use_aligned_loads();

  if (scale_bytes == 1) {
    ie_q4_log2_u8_q3_init_avx2();
    for (size_t r = 0; r < rows; ++r) {
      const uint8_t *row_blocks = W_blocks + r * row_block_bytes;
      const uint8_t *row_scales = W_scales + r * row_scale_bytes;
      const float *x_ptr = x;

      float acc = 0.0f;
      size_t b = 0;
      for (; b + 3 < n_blocks; b += 4) {
        if ((b & 3u) == 0u) {
          __builtin_prefetch(row_blocks + 128u, 0, 1);
          __builtin_prefetch(row_scales + 16u, 0, 1);
        }
        const float d0 = ie_log2_u8_q3_to_f32_fast(row_scales[0]);
        const float d1 = ie_log2_u8_q3_to_f32_fast(row_scales[1]);
        const float d2 = ie_log2_u8_q3_to_f32_fast(row_scales[2]);
        const float d3 = ie_log2_u8_q3_to_f32_fast(row_scales[3]);
        if (x_aligned) {
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 16, x_ptr + 32, d1);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 32, x_ptr + 64, d2);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 48, x_ptr + 96, d3);
        } else {
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 16, x_ptr + 32, d1);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 32, x_ptr + 64, d2);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 48, x_ptr + 96, d3);
        }
        row_blocks += 64;
        row_scales += 4;
        x_ptr += 128;
      }
      for (; b + 1 < n_blocks; b += 2) {
        if ((b & 3u) == 0u) {
          __builtin_prefetch(row_blocks + 64u, 0, 1);
          __builtin_prefetch(row_scales + 8u, 0, 1);
        }
        const float d0 = ie_log2_u8_q3_to_f32_fast(row_scales[0]);
        const float d1 = ie_log2_u8_q3_to_f32_fast(row_scales[1]);
        if (x_aligned) {
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 16, x_ptr + 32, d1);
        } else {
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 16, x_ptr + 32, d1);
        }
        row_blocks += 32;
        row_scales += 2;
        x_ptr += 64;
      }
      if (b < n_blocks) {
        const float d0 = ie_log2_u8_q3_to_f32_fast(row_scales[0]);
        acc += x_aligned ? ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0)
                          : ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
      }

      if (bias_bf16) {
        acc += ie_bf16_to_f32_scalar(bias_bf16[r]);
      }
      y[r] = acc;
    }
  } else {
    for (size_t r = 0; r < rows; ++r) {
      const uint8_t *row_blocks = W_blocks + r * row_block_bytes;
      const uint8_t *row_scales = W_scales + r * row_scale_bytes;
      const float *x_ptr = x;

      float acc = 0.0f;
      size_t b = 0;
      for (; b + 3 < n_blocks; b += 4) {
        if ((b & 3u) == 0u) {
          __builtin_prefetch(row_blocks + 128u, 0, 1);
          __builtin_prefetch(row_scales + 32u, 0, 1);
        }
        uint16_t s0, s1, s2, s3;
        memcpy(&s0, row_scales, sizeof(uint16_t));
        memcpy(&s1, row_scales + 2, sizeof(uint16_t));
        memcpy(&s2, row_scales + 4, sizeof(uint16_t));
        memcpy(&s3, row_scales + 6, sizeof(uint16_t));
        const float d0 = ie_bf16_to_f32_scalar(s0);
        const float d1 = ie_bf16_to_f32_scalar(s1);
        const float d2 = ie_bf16_to_f32_scalar(s2);
        const float d3 = ie_bf16_to_f32_scalar(s3);
        if (x_aligned) {
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 16, x_ptr + 32, d1);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 32, x_ptr + 64, d2);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 48, x_ptr + 96, d3);
        } else {
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 16, x_ptr + 32, d1);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 32, x_ptr + 64, d2);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 48, x_ptr + 96, d3);
        }
        row_blocks += 64;
        row_scales += 8;
        x_ptr += 128;
      }
      for (; b + 1 < n_blocks; b += 2) {
        if ((b & 3u) == 0u) {
          __builtin_prefetch(row_blocks + 64u, 0, 1);
          __builtin_prefetch(row_scales + 16u, 0, 1);
        }
        uint16_t s0;
        uint16_t s1;
        memcpy(&s0, row_scales, sizeof(uint16_t));
        memcpy(&s1, row_scales + 2, sizeof(uint16_t));
        const float d0 = ie_bf16_to_f32_scalar(s0);
        const float d1 = ie_bf16_to_f32_scalar(s1);
        if (x_aligned) {
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2_aligned(row_blocks + 16, x_ptr + 32, d1);
        } else {
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
          acc += ie_q4_0_block_dot_f32_avx2(row_blocks + 16, x_ptr + 32, d1);
        }
        row_blocks += 32;
        row_scales += 4;
        x_ptr += 64;
      }
      if (b < n_blocks) {
        uint16_t s0;
        memcpy(&s0, row_scales, sizeof(uint16_t));
        const float d0 = ie_bf16_to_f32_scalar(s0);
        acc += x_aligned ? ie_q4_0_block_dot_f32_avx2_aligned(row_blocks, x_ptr, d0)
                          : ie_q4_0_block_dot_f32_avx2(row_blocks, x_ptr, d0);
      }

      if (bias_bf16) {
        acc += ie_bf16_to_f32_scalar(bias_bf16[r]);
      }
      y[r] = acc;
    }
  }

  return 0;
}
