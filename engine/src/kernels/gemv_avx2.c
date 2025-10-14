/**
 * @file gemv_avx2.c
 * @brief AVX2/FMA-optimized GEMV (FP32). No-ops if not compiled with __AVX2__.
 */

#include "ie_kernels.h"

#if defined(__AVX2__)
  #include <immintrin.h>

/**
 * @brief AVX2 implementation of row-major GEMV: y = W * x
 *
 * Compiled only when __AVX2__ is defined. The generic TU requests installation.
 *
 * @param W    Row-major weights [rows x cols].
 * @param x    Input vector [cols].
 * @param y    Output vector [rows].
 * @param rows Number of rows (outputs).
 * @param cols Number of columns (inputs).
 */
static void ie_gemv_f32_avx2(const float *W, const float *x, float *y,
                             size_t rows, size_t cols) {
  const size_t V = 8;
  for (size_t r = 0; r < rows; ++r) {
    const float *w = W + r * cols;
    __m256 vacc = _mm256_setzero_ps();
    size_t c = 0;
    for (; c + V <= cols; c += V) {
      __m256 vw = _mm256_loadu_ps(w + c);
      __m256 vx = _mm256_loadu_ps(x + c);
      vacc = _mm256_fmadd_ps(vw, vx, vacc);
    }
    float sum = 0.0f;
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum += _mm_cvtss_f32(s);
    for (; c < cols; ++c) sum += w[c] * x[c];
    y[r] = sum;
  }
}
#endif /* __AVX2__ */

/**
 * @brief Hook called by the generic TU to install AVX2 kernels when available.
 *
 * If compiled without AVX2, this function is a no-op.
 *
 * @param setter Callback provided by the generic TU to set the GEMV function.
 */
void ie_kernels_install_avx2(void (*setter)(void (*)(const float*, const float*, float*, size_t, size_t))) {
#if defined(__AVX2__)
  if (setter) setter(ie_gemv_f32_avx2);
#else
  (void)setter;
#endif
}
