/**
 * @file gemv_generic.c
 * @brief Portable C GEMV with optional column-blocking and epilogue bias.
 */
#include "ie_kernels.h"

#include <stddef.h>
#include <string.h>

/** @brief Installed function pointer for GEMV (generic by default). */
static void (*s_gemv_f32)(const float*, const float*, float*, size_t, size_t, const float*, size_t) = NULL;

/**
 * @brief Generic C implementation: GEMV with optional blocked-K and bias epilogue.
 *
 * @param W     Pointer to weights (row-major or blocked-K).
 * @param x     Input vector (length cols).
 * @param y     Output vector (length rows).
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @param bias  Optional bias (length rows), may be NULL.
 * @param blk_k Column block size; 0 means plain row-major (no blocking).
 */
static void gemv_generic_impl(const float *W, const float *x, float *y,
                              size_t rows, size_t cols,
                              const float *bias, size_t blk_k) {
  const size_t BK = (blk_k > 0 ? blk_k : cols);

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols; /* blocked row is still contiguous */
    float acc = 0.0f;

    size_t kofs = 0;
    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + kofs;
      const float *xblk = x + k0;
      for (size_t k = 0; k < klen; ++k) acc += wblk[k] * xblk[k];
      kofs += klen;
    }
    if (bias) acc += bias[r];
    y[r] = acc;
  }
}

/**
 * @brief Install best kernels (generic fallback).
 *
 * @param use_avx2 Non-zero to prefer AVX2 if available (handled in gemv_avx2.c).
 */
void ie_kernels_install(int use_avx2) {
  (void)use_avx2;
  s_gemv_f32 = gemv_generic_impl;
}

/**
 * @brief Dispatch GEMV to the installed kernel.
 *
 * @param W     Weights pointer.
 * @param x     Input vector.
 * @param y     Output vector.
 * @param rows  Rows.
 * @param cols  Cols.
 * @param bias  Optional bias vector (may be NULL).
 * @param blk_k Column-block size for blocked layout; 0 for plain row-major.
 */
void ie_gemv_f32(const float *W, const float *x, float *y,
                 size_t rows, size_t cols,
                 const float *bias, size_t blk_k) {
  s_gemv_f32(W, x, y, rows, cols, bias, blk_k);
}
