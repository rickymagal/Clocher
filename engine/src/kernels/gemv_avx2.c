/* File: engine/src/kernels/gemv_avx2.c
 * -----------------------------------------------------------------------------
 * @file gemv_avx2.c
 * @brief AVX2/FMA-accelerated GEMV kernels (fp32 and INT8 activations).
 *
 * @details
 * Exposes only AVX2 implementation symbols:
 * - ::ie_gemv_f32_avx2_impl
 * - ::ie_gemv_qi8_f32_avx2_impl
 *
 * Public API and runtime dispatch live in gemv_generic.c.
 */

#include "ie_kernels.h"
#include "ie_quant_act.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Round n up to a multiple of a (a must be power of two).
 *
 * @param n Value to round up.
 * @param a Alignment (power of two).
 * @return Rounded-up value.
 */
static size_t ie_round_up_pow2(size_t n, size_t a) {
  return (n + (a - 1u)) & ~(a - 1u);
}

/**
 * @brief Allocate a 64-byte aligned buffer with C11 aligned_alloc rules.
 *
 * @details
 * aligned_alloc requires size to be a multiple of alignment; this helper rounds
 * up accordingly on non-MSVC toolchains.
 *
 * @param bytes Requested bytes.
 * @return Pointer to aligned allocation, or NULL on failure.
 */
static void *ie_aligned_malloc_64(size_t bytes) {
#if defined(_MSC_VER)
  return _aligned_malloc(bytes, 64);
#else
  const size_t need = ie_round_up_pow2(bytes, 64);
  return aligned_alloc(64, need);
#endif
}

/**
 * @brief Free a buffer allocated by ::ie_aligned_malloc_64.
 *
 * @param p Pointer to free (may be NULL).
 */
static void ie_aligned_free_64(void *p) {
#if defined(_MSC_VER)
  _aligned_free(p);
#else
  free(p);
#endif
}

/* -------------------------------------------------------------------------- */
/* AVX2 kernels                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief AVX2/FMA GEMV (fp32), blocked-K aware, optional bias epilogue.
 *
 * @details
 * Streams each row of W linearly and performs a dot product with x using AVX2
 * FMAs. Supports optional blocked-K traversal via blk_k.
 *
 * @param W     Weights (float).
 * @param x     Input vector (float).
 * @param y     Output vector (float).
 * @param rows  Rows.
 * @param cols  Cols.
 * @param bias  Optional bias or NULL.
 * @param blk_k Column block size; 0 disables blocking.
 */
__attribute__((target("avx2,fma"), used)) void
ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y, size_t rows,
                      size_t cols, const float *bias, size_t blk_k) {
  if (!W || !x || !y || rows == 0 || cols == 0) return;

  const size_t BK = (blk_k > 0 ? blk_k : cols);

#if defined(__GNUC__) || defined(__clang__)
  W = (const float *)__builtin_assume_aligned(W, 32);
  x = (const float *)__builtin_assume_aligned(x, 32);
  y = (float *)__builtin_assume_aligned(y, 32);
#endif

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    float acc_scalar = 0.0f;

    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();

    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + k0;
      const float *xblk = x + k0;

      if (k0 + 2 * BK < cols) {
        _mm_prefetch((const char *)(wblk + BK), _MM_HINT_T0);
        _mm_prefetch((const char *)(xblk + BK), _MM_HINT_T0);
      }

      size_t k = 0;
      for (; k + 32 <= klen; k += 32) {
        __m256 w0 = _mm256_loadu_ps(wblk + k + 0);
        __m256 x0 = _mm256_loadu_ps(xblk + k + 0);
        vacc0 = _mm256_fmadd_ps(w0, x0, vacc0);

        __m256 w1 = _mm256_loadu_ps(wblk + k + 8);
        __m256 x1 = _mm256_loadu_ps(xblk + k + 8);
        vacc1 = _mm256_fmadd_ps(w1, x1, vacc1);

        __m256 w2 = _mm256_loadu_ps(wblk + k + 16);
        __m256 x2 = _mm256_loadu_ps(xblk + k + 16);
        vacc2 = _mm256_fmadd_ps(w2, x2, vacc2);

        __m256 w3 = _mm256_loadu_ps(wblk + k + 24);
        __m256 x3 = _mm256_loadu_ps(xblk + k + 24);
        vacc3 = _mm256_fmadd_ps(w3, x3, vacc3);
      }

      for (; k + 8 <= klen; k += 8) {
        __m256 wv = _mm256_loadu_ps(wblk + k);
        __m256 xv = _mm256_loadu_ps(xblk + k);
        vacc0 = _mm256_fmadd_ps(wv, xv, vacc0);
      }

      for (; k < klen; ++k) {
        acc_scalar += wblk[k] * xblk[k];
      }
    }

    __m256 t01 = _mm256_add_ps(vacc0, vacc1);
    __m256 t23 = _mm256_add_ps(vacc2, vacc3);
    __m256 t = _mm256_add_ps(t01, t23);
    __m128 low = _mm256_castps256_ps128(t);
    __m128 high = _mm256_extractf128_ps(t, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 shf = _mm_movehdup_ps(sum);
    sum = _mm_add_ps(sum, shf);
    shf = _mm_movehl_ps(shf, sum);
    sum = _mm_add_ss(sum, shf);

    float acc = _mm_cvtss_f32(sum) + acc_scalar;
    if (bias) acc += bias[r];
    y[r] = acc;
  }
}

/**
 * @brief AVX2/FMA GEMV with per-tensor INT8 activations (dequant + fp32 GEMV).
 *
 * @details
 * Dequantizes x_q into a temporary 64-byte aligned float buffer, then calls
 * ::ie_gemv_f32_avx2_impl. The temp buffer is freed before returning.
 *
 * @param W         Weights.
 * @param x_q       INT8 activations.
 * @param y         Output.
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias or NULL.
 * @param blk_k     Column block size; 0 disables blocking.
 * @param params    INT8 params.
 * @param symmetric Informational flag (unused).
 */
__attribute__((target("avx2,fma"), used)) void
ie_gemv_qi8_f32_avx2_impl(const float *W, const int8_t *x_q, float *y,
                          size_t rows, size_t cols, const float *bias,
                          size_t blk_k, ie_act_i8_params params,
                          int symmetric) {
  (void)symmetric;
  if (!W || !x_q || !y || rows == 0 || cols == 0) return;

  float *x = (float *)ie_aligned_malloc_64(cols * sizeof(float));
  if (!x) return;

  const float s = params.scale;
  const int z = (int)params.zero_point;

  size_t i = 0;
  for (; i + 8 <= cols; i += 8) {
    __m128i q = _mm_loadl_epi64((const __m128i *)(x_q + i));
    __m256i qi = _mm256_cvtepi8_epi32(q);
    __m256 qf = _mm256_cvtepi32_ps(qi);
    __m256 zf = _mm256_set1_ps((float)z);
    __m256 sf = _mm256_set1_ps(s);
    __m256 xf = _mm256_mul_ps(sf, _mm256_sub_ps(qf, zf));
    _mm256_storeu_ps(x + i, xf);
  }
  for (; i < cols; ++i) x[i] = s * ((int)x_q[i] - z);

  ie_gemv_f32_avx2_impl(W, x, y, rows, cols, bias, blk_k);

  ie_aligned_free_64(x);
}
