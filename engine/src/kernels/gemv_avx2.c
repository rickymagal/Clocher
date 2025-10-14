/**
 * @file gemv_avx2.c
 * @brief AVX2-accelerated GEMV with optional column-blocking and epilogue bias.
 *
 * The generic dispatcher can use this implementation when AVX2/FMA is present.
 * We mark the function with `__attribute__((used))` so that toolchains running
 * with -Werror do not complain about “defined but not used” in translation units
 * where the reference is resolved at link-time.
 */

#include "ie_kernels.h"
#include <immintrin.h>
#include <stddef.h>

/**
 * @brief AVX2 GEMV core (blocked-K aware) with bias epilogue.
 *
 * @param W     Weights (row-major or blocked-K contiguous per row).
 * @param x     Input vector (length cols).
 * @param y     Output vector (length rows).
 * @param rows  Number of rows.
 * @param cols  Number of cols.
 * @param bias  Optional bias (length rows), may be NULL.
 * @param blk_k Column block size; 0 => treat as single block of size cols.
 */
__attribute__((target("avx2,fma"), used))
void ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y,
                           size_t rows, size_t cols,
                           const float *bias, size_t blk_k) {
  const size_t BK = (blk_k > 0 ? blk_k : cols);

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    __m256 vacc = _mm256_setzero_ps();

    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + k0;
      const float *xblk = x + k0;

      /* Prefetch next chunk (non-invasive hint) */
      if (k0 + 2*BK < cols) {
        _mm_prefetch((const char*)(wblk + BK), _MM_HINT_T0);
        _mm_prefetch((const char*)(xblk + BK), _MM_HINT_T0);
      }

      size_t k = 0;
      for (; k + 32 <= klen; k += 32) {
        __m256 w0 = _mm256_loadu_ps(wblk + k + 0);
        __m256 x0 = _mm256_loadu_ps(xblk + k + 0);
        vacc = _mm256_fmadd_ps(w0, x0, vacc);

        __m256 w1 = _mm256_loadu_ps(wblk + k + 8);
        __m256 x1 = _mm256_loadu_ps(xblk + k + 8);
        vacc = _mm256_fmadd_ps(w1, x1, vacc);

        __m256 w2 = _mm256_loadu_ps(wblk + k + 16);
        __m256 x2 = _mm256_loadu_ps(xblk + k + 16);
        vacc = _mm256_fmadd_ps(w2, x2, vacc);

        __m256 w3 = _mm256_loadu_ps(wblk + k + 24);
        __m256 x3 = _mm256_loadu_ps(xblk + k + 24);
        vacc = _mm256_fmadd_ps(w3, x3, vacc);
      }
      for (; k + 8 <= klen; k += 8) {
        __m256 wv = _mm256_loadu_ps(wblk + k);
        __m256 xv = _mm256_loadu_ps(xblk + k);
        vacc = _mm256_fmadd_ps(wv, xv, vacc);
      }
      float scalar = 0.0f;
      for (; k < klen; ++k) scalar += wblk[k] * xblk[k];

      /* Horizontal sum of vacc + scalar tail */
      __m256 t1 = _mm256_hadd_ps(vacc, vacc);
      __m256 t2 = _mm256_hadd_ps(t1, t1);
      __m128 low = _mm256_castps256_ps128(t2);
      __m128 high = _mm256_extractf128_ps(t2, 1);
      __m128 sum128 = _mm_add_ps(low, high);
      float acc = ((float*)&sum128)[0] + scalar;

      if (bias) acc += bias[r];
      y[r] = acc;

      vacc = _mm256_setzero_ps();
    }
  }
}
