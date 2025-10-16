/**
 * @file test_int8_ptq.c
 * @brief C unit tests for INT8 PTQ min-max scaling and (de)quantization.
 *
 * These tests validate:
 *  - Per-tensor and per-row scale computation
 *  - Quantize -> Dequantize round-trip numerical behavior
 *  - Symmetric clamp behavior in [-127, +127]
 */

#include "ie_quant.h"

#include <math.h>    /* fabsf */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Compute mean squared error between two float arrays.
 *
 * @param a Pointer to the first float array.
 * @param b Pointer to the second float array.
 * @param n Number of elements in both arrays.
 * @return The mean squared error as a float.
 */
static float mse(const float *a, const float *b, size_t n) {
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return (n > 0) ? (s / (float)n) : 0.0f;
}

/**
 * @brief Check that all elements of an INT8 array lie within [-127, 127].
 *
 * @param q8 Pointer to INT8 data.
 * @param n Number of elements.
 * @return 1 if valid, 0 otherwise.
 */
static int all_in_sym_int8_range(const int8_t *q8, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (q8[i] < -127 || q8[i] > 127) return 0;
  }
  return 1;
}

/**
 * @brief Smoke test for per-tensor scale, quantize and dequantize.
 *
 * Generates a simple ramp, quantizes with per-tensor scaling and
 * checks basic properties (range, small MSE, non-zero scale).
 *
 * @return EXIT_SUCCESS on pass; EXIT_FAILURE otherwise.
 */
static int test_per_tensor_roundtrip(void) {
  const size_t rows = 4, cols = 8, n = rows * cols;
  float *w = (float*)malloc(n * sizeof(float));
  float *w_deq = (float*)malloc(n * sizeof(float));
  int8_t *q8 = (int8_t*)malloc(n * sizeof(int8_t));
  float scale[1] = {0.0f};

  for (size_t i = 0; i < n; ++i) {
    const float v = (float)((int)i - (int)n/2) / 13.0f; /* spread values */
    w[i] = v;
  }

  ie_ptq_compute_scales_minmax(w, rows, cols, IE_PTQ_SCALE_PER_TENSOR, scale);
  if (!(scale[0] > 0.0f)) {
    fprintf(stderr, "per-tensor: expected scale > 0\n");
    return EXIT_FAILURE;
  }

  ie_ptq_quantize_int8(w, rows, cols, IE_PTQ_SCALE_PER_TENSOR, scale, q8);
  if (!all_in_sym_int8_range(q8, n)) {
    fprintf(stderr, "per-tensor: q8 outside symmetric range\n");
    return EXIT_FAILURE;
  }

  ie_ptq_dequant_int8(q8, rows, cols, IE_PTQ_SCALE_PER_TENSOR, scale, w_deq);
  const float e = mse(w, w_deq, n);
  if (!(e >= 0.0f && e < 5e-3f)) {
    fprintf(stderr, "per-tensor: MSE too large: %g\n", (double)e);
    return EXIT_FAILURE;
  }

  free(w);
  free(w_deq);
  free(q8);
  return EXIT_SUCCESS;
}

/**
 * @brief Test per-row scaling on a matrix with different row magnitudes.
 *
 * Builds two rows with different dynamic range to ensure independent scales.
 *
 * @return EXIT_SUCCESS on pass; EXIT_FAILURE otherwise.
 */
static int test_per_row_independent_scales(void) {
  const size_t rows = 2, cols = 6;
  float w[12] = {
    -0.1f, -0.05f, 0.0f, 0.02f, 0.05f, 0.09f,   /* small row */
    -1.5f, -0.7f,  0.0f, 0.9f,  1.2f,  1.4f    /* large row */
  };
  float scales[2] = {0.0f, 0.0f};
  int8_t q8[12];
  float w_deq[12];

  ie_ptq_compute_scales_minmax(w, rows, cols, IE_PTQ_SCALE_PER_ROW, scales);
  if (!(scales[0] > 0.0f && scales[1] > 0.0f)) {
    fprintf(stderr, "per-row: expected both scales > 0\n");
    return EXIT_FAILURE;
  }
  if (!(scales[1] > scales[0])) {
    fprintf(stderr, "per-row: expected larger row to have larger scale\n");
    return EXIT_FAILURE;
  }

  ie_ptq_quantize_int8(w, rows, cols, IE_PTQ_SCALE_PER_ROW, scales, q8);
  if (!all_in_sym_int8_range(q8, rows * cols)) {
    fprintf(stderr, "per-row: q8 outside symmetric range\n");
    return EXIT_FAILURE;
  }

  ie_ptq_dequant_int8(q8, rows, cols, IE_PTQ_SCALE_PER_ROW, scales, w_deq);
  const float e = mse(w, w_deq, rows * cols);
  if (!(e >= 0.0f && e < 5e-3f)) {
    fprintf(stderr, "per-row: MSE too large: %g\n", (double)e);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

/**
 * @brief Validate symmetric clamping at the INT8 limits.
 *
 * Builds values above/below the representable range and checks
 * that quantization clamps to [-127, +127].
 *
 * @return EXIT_SUCCESS on pass; EXIT_FAILURE otherwise.
 */
static int test_saturation_symmetric(void) {
  const size_t rows = 1, cols = 6;
  const float w[6] = { -10.0f, -1.0f, -0.01f, 0.01f, 1.0f, 10.0f };
  float scale[1] = { 0.01f }; /* force saturation for extremes */
  int8_t q8[6];

  ie_ptq_quantize_int8(w, rows, cols, IE_PTQ_SCALE_PER_TENSOR, scale, q8);
  if (!(q8[0] == -127 && q8[5] == 127)) {
    fprintf(stderr, "saturation: expected symmetric clamp to [-127,127] got %d and %d\n",
            (int)q8[0], (int)q8[5]);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * @brief Program entry: runs all PTQ unit tests.
 *
 * @return EXIT_SUCCESS if all pass; EXIT_FAILURE otherwise.
 */
int main(void) {
  if (test_per_tensor_roundtrip() != EXIT_SUCCESS) return EXIT_FAILURE;
  if (test_per_row_independent_scales() != EXIT_SUCCESS) return EXIT_FAILURE;
  if (test_saturation_symmetric() != EXIT_SUCCESS) return EXIT_FAILURE;
  printf("ok test_int8_ptq\n");
  return EXIT_SUCCESS;
}
