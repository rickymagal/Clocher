/* ============================================================================
 * File: engine/src/kernels/gemv_avx2.c
 * ============================================================================
 */
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ie_kernels.h"

#if defined(__GNUC__) || defined(__clang__)
#define IE_TARGET_AVX2 __attribute__((target("avx2,fma,sse3")))
#else
#define IE_TARGET_AVX2
#endif

/* ------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* ------------------------------------------------------------------------- */

/**
 * @brief Horizontal sum of an AVX2 register.
 *
 * @param v Input vector.
 * @return Sum of all 8 lanes.
 */
static IE_TARGET_AVX2 inline float ie_hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);

  __m128 shuf = _mm_movehdup_ps(sum);
  sum = _mm_add_ps(sum, shuf);
  shuf = _mm_movehl_ps(shuf, sum);
  sum = _mm_add_ss(sum, shuf);

  return _mm_cvtss_f32(sum);
}

/**
 * @brief Convert 8 BF16 values to FP32.
 *
 * @details
 * BF16 is represented as the high 16 bits of an IEEE-754 FP32 value.
 * Conversion is performed by zero-extending to 32 bits and shifting left by 16.
 */
static IE_TARGET_AVX2 inline __m256 ie_bf16x8_to_f32(__m128i v_bf16) {
  __m256i v_u32 = _mm256_cvtepu16_epi32(v_bf16);
  v_u32 = _mm256_slli_epi32(v_u32, 16);
  return _mm256_castsi256_ps(v_u32);
}

/* ------------------------------------------------------------------------- */
/* Public AVX2 entry points                                                   */
/* ------------------------------------------------------------------------- */

IE_TARGET_AVX2
void ie_vec_bf16_to_f32_avx2_impl(const uint16_t *in, float *out, size_t n) {
  if (!in || !out || n == 0u) return;

  size_t i = 0;
  for (; i + 8u <= n; i += 8u) {
    __m128i v16 = _mm_loadu_si128((const __m128i *)(const void *)(in + i));
    __m256 vf = ie_bf16x8_to_f32(v16);
    _mm256_storeu_ps(out + i, vf);
  }

  for (; i < n; ++i) {
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)in[i]) << 16;
    out[i] = v.f;
  }
}

IE_TARGET_AVX2
int ie_gemv_bf16_f32_avx2_impl(const uint16_t *W_bf16, const float *x, float *y,
                              size_t rows, size_t cols,
                              const uint16_t *bias_bf16) {
  if (!W_bf16 || !x || !y || rows == 0u || cols == 0u) return -1;

  const size_t blk_k = 256u;

  for (size_t r = 0; r < rows; ++r) {
    const uint16_t *row = W_bf16 + r * cols;

    __m256 acc = _mm256_setzero_ps();
    float tail_sum = 0.0f;

    for (size_t k0 = 0; k0 < cols; k0 += blk_k) {
      const size_t kend = (k0 + blk_k <= cols) ? (k0 + blk_k) : cols;
      size_t c = k0;

      for (; c + 16u <= kend; c += 16u) {
        _mm_prefetch((const char *)(row + c + 64u), _MM_HINT_T0);
        _mm_prefetch((const char *)(x + c + 64u), _MM_HINT_T0);

        __m128i w16_0 = _mm_loadu_si128((const __m128i *)(const void *)(row + c));
        __m128i w16_1 = _mm_loadu_si128((const __m128i *)(const void *)(row + c + 8u));
        __m256 w0 = ie_bf16x8_to_f32(w16_0);
        __m256 w1 = ie_bf16x8_to_f32(w16_1);
        __m256 x0 = _mm256_loadu_ps(x + c);
        __m256 x1 = _mm256_loadu_ps(x + c + 8u);
        acc = _mm256_fmadd_ps(w0, x0, acc);
        acc = _mm256_fmadd_ps(w1, x1, acc);
      }

      for (; c + 8u <= kend; c += 8u) {
        _mm_prefetch((const char *)(row + c + 64u), _MM_HINT_T0);
        _mm_prefetch((const char *)(x + c + 64u), _MM_HINT_T0);
        __m128i w16 = _mm_loadu_si128((const __m128i *)(const void *)(row + c));
        __m256 w = ie_bf16x8_to_f32(w16);
        __m256 xv = _mm256_loadu_ps(x + c);
        acc = _mm256_fmadd_ps(w, xv, acc);
      }

      for (; c < kend; ++c) {
        union { uint32_t u; float f; } v;
        v.u = ((uint32_t)row[c]) << 16;
        tail_sum += v.f * x[c];
      }
    }

    float sum = ie_hsum256_ps(acc) + tail_sum;

    if (bias_bf16) {
      union { uint32_t u; float f; } b;
      b.u = ((uint32_t)bias_bf16[r]) << 16;
      sum += b.f;
    }

    y[r] = sum;
  }

  return 0;
}

IE_TARGET_AVX2
void ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y,
                          size_t rows, size_t cols,
                          const float *bias, size_t bias_stride) {
  for (size_t r = 0; r < rows; r++) {
    const float *row = W + r * cols;

    __m256 acc = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 8 <= cols; c += 8) {
      __m256 wv = _mm256_loadu_ps(row + c);
      __m256 xv = _mm256_loadu_ps(x + c);
      acc = _mm256_fmadd_ps(wv, xv, acc);
    }

    float sum = ie_hsum256_ps(acc);

    for (; c < cols; c++) {
      sum += row[c] * x[c];
    }

    if (bias) {
      sum += bias[r * bias_stride];
    }

    y[r] = sum;
  }
}

IE_TARGET_AVX2
void ie_gemv_qi8_avx2_impl(const int8_t *W_qi8, const float *W_scales,
                          const float *x, float *y,
                          size_t rows, size_t cols,
                          size_t block_cols,
                          const float *bias, size_t bias_stride) {
  (void)block_cols;

  for (size_t r = 0; r < rows; r++) {
    const int8_t *row = W_qi8 + r * cols;

    __m256 acc_ps = _mm256_setzero_ps();

    size_t c = 0;
    for (; c + 8 <= cols; c += 8) {
      __m128i v8 = _mm_loadl_epi64((const __m128i *)(const void *)(row + c));
      __m256i v32 = _mm256_cvtepi8_epi32(v8);

      __m256 w = _mm256_cvtepi32_ps(v32);
      __m256 xv = _mm256_loadu_ps(x + c);

      acc_ps = _mm256_fmadd_ps(w, xv, acc_ps);
    }

    float sum = ie_hsum256_ps(acc_ps);

    for (; c < cols; c++) {
      sum += (float)row[c] * x[c];
    }

    float s = W_scales ? W_scales[r] : 1.0f;
    sum *= s;

    if (bias) {
      sum += bias[r * bias_stride];
    }

    y[r] = sum;
  }
}
