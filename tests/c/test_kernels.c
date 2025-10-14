/**
 * @file test_kernels.c
 * @brief Unit test for GEMV kernel dispatcher (generic vs. AVX2).
 *
 * This test checks that:
 *  1) The generic GEMV (implicitly selected when `ie_kernels_install(0)`)
 *     produces a correct result against a simple reference implementation.
 *  2) The installed kernel (with `ie_kernels_install(1)`) produces the
 *     same results as the generic path (bitwise within 1e-4).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ie_kernels.h"

/**
 * @brief Reference GEMV: y = W * x (row-major), implemented in plain C.
 *
 * @param W Row-major weights of shape [R x C].
 * @param x Input vector of length C.
 * @param y Output vector of length R.
 * @param R Number of rows (outputs).
 * @param C Number of columns (inputs).
 */
static void gemv_ref(const float *W, const float *x, float *y, size_t R, size_t C) {
  for (size_t r = 0; r < R; ++r) {
    float acc = 0.0f;
    const float *w = W + r * C;
    for (size_t c = 0; c < C; ++c) acc += w[c] * x[c];
    y[r] = acc;
  }
}

/**
 * @brief Fill an array with a simple deterministic pattern.
 *
 * @param dst Destination buffer of length @p n.
 * @param n   Number of elements.
 */
static void fill_pattern(float *dst, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    dst[i] = (float)(((int)(i % 7) - 3)) * 0.25f;
  }
}

/**
 * @brief Test entry point.
 *
 * Allocates small matrices/vectors, computes reference result, then compares
 * with the dispatcher before and after installing the AVX2 kernel (if present).
 *
 * @return 0 on success, non-zero on failure.
 */
int main(void) {
  const size_t R = 7, C = 13;
  float *W  = (float*)malloc(R * C * sizeof(float));
  float *x  = (float*)malloc(C * sizeof(float));
  float *y0 = (float*)malloc(R * sizeof(float));
  float *y1 = (float*)malloc(R * sizeof(float));
  float *yr = (float*)malloc(R * sizeof(float));
  if (!W || !x || !y0 || !y1 || !yr) {
    fprintf(stderr, "alloc failed\n");
    free(W); free(x); free(y0); free(y1); free(yr);
    return 1;
  }

  /* Initialize inputs with deterministic patterns. */
  for (size_t i = 0; i < R*C; ++i) W[i] = (float)(((int)(i % 5) - 2)) * 0.1f;
  fill_pattern(x, C);

  /* Reference. */
  gemv_ref(W, x, yr, R, C);

  /* Generic path. */
  ie_kernels_install(0);             /* force generic */
  ie_gemv_f32(W, x, y0, R, C);

  /* “Best” path (AVX2 if compiled+supported, otherwise still generic). */
  ie_kernels_install(1);             /* allow AVX2 */
  ie_gemv_f32(W, x, y1, R, C);

  /* Compare against reference and between implementations. */
  for (size_t i = 0; i < R; ++i) {
    assert(fabsf(y0[i] - yr[i]) < 1e-4f);
    assert(fabsf(y1[i] - yr[i]) < 1e-4f);
  }

  free(W); free(x); free(y0); free(y1); free(yr);
  printf("ok test_kernels\n");
  return 0;
}
