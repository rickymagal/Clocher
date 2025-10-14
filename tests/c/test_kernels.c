/**
 * @file test_kernels.c
 * @brief Unit tests for GEMV kernels (row-major path, bias epilogue).
 *
 * This test validates:
 *  - Row-major GEMV without bias (blk_k = 0)
 *  - Row-major GEMV with epilogue bias (blk_k = 0)
 *
 * NOTE: The blocked-K layout is exercised indirectly by integration tests.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ie_kernels.h"

/**
 * @brief Compare two float vectors with absolute tolerance.
 *
 * @param a   First vector.
 * @param b   Second vector.
 * @param n   Number of elements to compare.
 * @param tol Absolute tolerance.
 * @return 0 if all elements are within @p tol; non-zero otherwise.
 */
static int vec_allclose(const float *a, const float *b, size_t n, float tol) {
  for (size_t i = 0; i < n; ++i) {
    float d = fabsf(a[i] - b[i]);
    if (d > tol) {
      fprintf(stderr, "mismatch at %zu: a=%.6f b=%.6f |diff|=%.6f > %.6f\n",
              i, a[i], b[i], d, tol);
      return 1;
    }
  }
  return 0;
}

/**
 * @brief Fill a small, deterministic row-major matrix W[r,c] = (r+1)*0.1 + (c+1)*0.01.
 *
 * @param W    Output buffer (size rows*cols).
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
static void fill_matrix(float *W, size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    for (size_t c = 0; c < cols; ++c) {
      W[r*cols + c] = (float)(0.1 * (double)(r + 1) + 0.01 * (double)(c + 1));
    }
  }
}

/**
 * @brief Fill a small, deterministic vector x[i] = 0.05 * (i+1).
 *
 * @param x Output buffer (length n).
 * @param n Number of elements.
 */
static void fill_vector(float *x, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] = 0.05f * (float)(i + 1);
}

/**
 * @brief Compute reference row-major GEMV y = W*x (no bias).
 *
 * @param W    Row-major weights (rows x cols).
 * @param x    Input vector (length cols).
 * @param y    Output vector (length rows).
 * @param rows Rows.
 * @param cols Cols.
 */
static void ref_gemv_nominal(const float *W, const float *x, float *y,
                             size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *wrow = W + r * cols;
    for (size_t c = 0; c < cols; ++c) acc += wrow[c] * x[c];
    y[r] = acc;
  }
}

/**
 * @brief Compute reference y = W*x + bias.
 *
 * @param W    Row-major weights (rows x cols).
 * @param x    Input vector (length cols).
 * @param y    Output vector (length rows).
 * @param bias Bias vector (length rows).
 * @param rows Rows.
 * @param cols Cols.
 */
static void ref_gemv_bias(const float *W, const float *x, float *y,
                          const float *bias, size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *wrow = W + r * cols;
    for (size_t c = 0; c < cols; ++c) acc += wrow[c] * x[c];
    y[r] = acc + bias[r];
  }
}

/**
 * @brief Test: GEMV without bias, row-major, blk_k=0.
 *
 * @return 0 on success; non-zero on failure.
 */
static int test_gemv_rowmajor_no_bias(void) {
  const size_t R = 3, C = 4;
  float W[R*C], x[C], y[R], y_ref[R];
  fill_matrix(W, R, C);
  fill_vector(x, C);

  ref_gemv_nominal(W, x, y_ref, R, C);

  /* bias=NULL, blk_k=0 => row-major path */
  memset(y, 0, sizeof(y));
  ie_gemv_f32(W, x, y, R, C, /*bias*/NULL, /*blk_k*/0);

  return vec_allclose(y, y_ref, R, 1e-6f);
}

/**
 * @brief Test: GEMV with bias epilogue, row-major, blk_k=0.
 *
 * @return 0 on success; non-zero on failure.
 */
static int test_gemv_rowmajor_with_bias(void) {
  const size_t R = 3, C = 4;
  float W[R*C], x[C], bias[R], y[R], y_ref[R];
  fill_matrix(W, R, C);
  fill_vector(x, C);
  for (size_t r = 0; r < R; ++r) bias[r] = 0.001f * (float)(r + 1);

  ref_gemv_bias(W, x, y_ref, bias, R, C);

  memset(y, 0, sizeof(y));
  ie_gemv_f32(W, x, y, R, C, /*bias*/bias, /*blk_k*/0);

  return vec_allclose(y, y_ref, R, 1e-6f);
}

/**
 * @brief Program entry: run all kernel tests and print "ok test_kernels" on success.
 *
 * @return 0 on success; non-zero on assertion failure.
 */
int main(void) {
  ie_kernels_install(/*use_avx2=*/0);

  if (test_gemv_rowmajor_no_bias() != 0) {
    fprintf(stderr, "FAIL: test_gemv_rowmajor_no_bias\n");
    return 1;
  }
  if (test_gemv_rowmajor_with_bias() != 0) {
    fprintf(stderr, "FAIL: test_gemv_rowmajor_with_bias\n");
    return 1;
  }

  puts("ok test_kernels");
  return 0;
}
