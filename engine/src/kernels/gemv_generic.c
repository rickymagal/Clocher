/**
 * @file gemv_generic.c
 * @brief Generic C fallback implementation for FP32 row-major GEMV.
 */

#include "ie_kernels.h"

/** @brief Function pointer to the active GEMV implementation. */
static void (*g_ie_gemv_impl)(const float*, const float*, float*, size_t, size_t) = NULL;

/**
 * @brief Generic row-major GEMV implementation: y = W * x
 *
 * @param W    Row-major weights of shape [rows x cols].
 * @param x    Input vector of length cols.
 * @param y    Output vector of length rows.
 * @param rows Number of rows (outputs).
 * @param cols Number of columns (inputs).
 */
static void ie_gemv_f32_generic(const float *W, const float *x, float *y,
                                size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *w = W + r * cols;
    for (size_t c = 0; c < cols; ++c) acc += w[c] * x[c];
    y[r] = acc;
  }
}

/**
 * @brief Public dispatcher: call the currently installed GEMV kernel.
 *
 * Falls back to the generic implementation if no kernel has been installed.
 */
void ie_gemv_f32(const float *W, const float *x, float *y, size_t rows, size_t cols) {
  if (!g_ie_gemv_impl) g_ie_gemv_impl = ie_gemv_f32_generic;
  g_ie_gemv_impl(W, x, y, rows, cols);
}

/**
 * @brief Install a new GEMV implementation into the global dispatch pointer.
 *
 * @param pfn  Non-NULL function pointer for GEMV.
 */
static void ie_gemv_install(void (*pfn)(const float*, const float*, float*, size_t, size_t)) {
  if (pfn) g_ie_gemv_impl = pfn;
}

/**
 * @brief External installer for AVX2 kernels (defined in gemv_avx2.c).
 *
 * @param setter Callback to set the GEMV function pointer.
 */
void ie_kernels_install_avx2(void (*setter)(void (*)(const float*, const float*, float*, size_t, size_t)));

/**
 * @brief Install the best available kernel set according to runtime flags.
 *
 * @param enable_avx2  When non-zero, attempt to install AVX2 kernels.
 */
void ie_kernels_install(int enable_avx2) {
  if (enable_avx2) {
    ie_kernels_install_avx2(ie_gemv_install);
  } else {
    g_ie_gemv_impl = ie_gemv_f32_generic;
  }
}
