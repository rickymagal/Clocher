/* File: engine/src/kernels/gemv_avx2.c
 * -----------------------------------------------------------------------------
 * @file gemv_avx2.c
 * @brief AVX2/FMA-accelerated kernels for GEMV (float) and INT8-activation GEMV.
 *
 * This translation unit exposes *only* AVX2 implementation symbols:
 *  - ::ie_gemv_f32_avx2_impl
 *  - ::ie_gemv_qi8_f32_avx2_impl
 *
 * The public API and the dispatcher live in gemv_generic.c.
 */

#include "ie_kernels.h"
#include "ie_quant_act.h"

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief AVX2/FMA GEMV (float path), blocked-K aware, with bias epilogue.
 *
 * The implementation accepts either plain row-major W or rows packed as
 * "blocked-K contiguous" (see pretranspose docs). In the latter case,
 * each row still has @p cols elements but the inner loop can iterate in
 * chunks of @p blk_k to promote cache locality.
 *
 * @param W     Weights (row-major or blocked-K contiguous per row).
 * @param x     Input vector (length @p cols).
 * @param y     Output vector (length @p rows).
 * @param rows  Number of rows.
 * @param cols  Number of cols.
 * @param bias  Optional bias (length @p rows), may be NULL.
 * @param blk_k Column block size; 0 => treat as single block of size @p cols.
 */
__attribute__((target("avx2,fma"), used)) void
ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y, size_t rows,
                      size_t cols, const float *bias, size_t blk_k) {
  const size_t BK = (blk_k > 0 ? blk_k : cols);

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

      /* Prefetch next chunk (temporal hint) */
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

    /* Reduce four accumulators to one scalar */
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
 * @brief AVX2/FMA GEMV with per-tensor INT8 activations
 * (two-pass: dequant then float GEMV).
 *
 * This path dequantizes activations into a temporary aligned float buffer, then
 * calls the AVX2 float GEMV core ::ie_gemv_f32_avx2_impl().
 *
 * @param W         Weights (row-major).
 * @param x_q       INT8 activations.
 * @param y         Output.
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias (may be NULL).
 * @param blk_k     Column block size; 0 => treat as single block.
 * @param params    INT8 per-tensor params (scale, zero_point).
 * @param symmetric Non-zero if symmetric (informational).
 */
__attribute__((target("avx2,fma"), used)) void
ie_gemv_qi8_f32_avx2_impl(const float *W, const int8_t *x_q, float *y,
                          size_t rows, size_t cols, const float *bias,
                          size_t blk_k, ie_act_i8_params params,
                          int symmetric) {
  (void)symmetric;
#if defined(_MSC_VER)
  float *x = (float *)_aligned_malloc(cols * sizeof(float), 64);
  if (!x) return;
#else
  float *x = (float *)aligned_alloc(64, cols * sizeof(float));
  if (!x) return;
#endif

  const float s = params.scale;
  const int z = (int)params.zero_point;

  size_t i = 0;
  for (; i + 8 <= cols; i += 8) {
    __m128i q = _mm_loadl_epi64((const __m128i *)(x_q + i)); /* 8x i8 */
    __m256i qi = _mm256_cvtepi8_epi32(q);                    /* -> 8x i32 */
    __m256 qf = _mm256_cvtepi32_ps(qi);                      /* -> 8x f32 */
    __m256 zf = _mm256_set1_ps((float)z);
    __m256 sf = _mm256_set1_ps(s);
    __m256 xf = _mm256_mul_ps(sf, _mm256_sub_ps(qf, zf));
    _mm256_storeu_ps(x + i, xf);
  }
  for (; i < cols; ++i) x[i] = s * ((int)x_q[i] - z);

  ie_gemv_f32_avx2_impl(W, x, y, rows, cols, bias, blk_k);

#if defined(_MSC_VER)
  _aligned_free(x);
#else
  free(x);
#endif
}
