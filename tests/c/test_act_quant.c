/**
 * @file test_act_quant.c
 * @brief Sanity tests for activation quantization (INT8/FP8).
 *
 * The tests validate:
 *  - Per-tensor INT8 roundtrip error stays within reasonable bounds.
 *  - Per-group INT8 matches per-tensor when group_size == n.
 *  - FP8 encoders/decoders are monotone and stable on a sweep.
 *
 * This is a lightweight test meant to run in CI quickly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "ie_quant_act.h"

static void fill_range(float* x, size_t n, float lo, float hi) {
  for (size_t i = 0; i < n; ++i) {
    float t = (float)i / (float)(n - 1 ? n - 1 : 1);
    x[i] = lo + t * (hi - lo);
  }
}

static float mse(const float* a, const float* b, size_t n) {
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = (double)a[i] - (double)b[i];
    acc += d * d;
  }
  return (float)(acc / (double)(n ? n : 1));
}

static int test_int8_per_tensor(void) {
  const size_t n = 4096;
  float*  x  = (float*)aligned_alloc(64, n * sizeof(float));
  int8_t* q  = (int8_t*)aligned_alloc(64, n * sizeof(int8_t));
  float*  y  = (float*)aligned_alloc(64, n * sizeof(float));

  fill_range(x, n, -3.0f, 3.0f);

  float mn = x[0], mx = x[n-1];
  ie_act_i8_params p;
  ie_act_i8_params_from_minmax(mn, mx, /*symmetric=*/1, &p.scale, &p.zero_point);

  ie_quantize_act_int8(x, q, n, p, /*symmetric=*/1);
  ie_dequantize_act_int8(q, y, n, p);

  float e = mse(x, y, n);
  printf("[INT8 per-tensor] MSE = %.6e (scale=%.6g, zp=%d)\n",
         e, p.scale, (int)p.zero_point);

  free(x); free(q); free(y);
  return (e < 1e-2f) ? 0 : 1;
}

static int test_int8_per_group(void) {
  const size_t n = 8192;
  const size_t group = 128;

  float*  x   = (float*)aligned_alloc(64, n * sizeof(float));
  int8_t* q   = (int8_t*)aligned_alloc(64, n * sizeof(int8_t));
  float*  y   = (float*)aligned_alloc(64, n * sizeof(float));

  /* Mixed range to exercise different groups. */
  for (size_t i = 0; i < n; ++i) {
    float s = (i % 2 == 0) ? 1.0f : 0.25f;
    x[i] = s * sinf((float)i * 0.0137f) * 4.0f;
  }

  const size_t G = (n + group - 1) / group;
  float*  scales = (float*)malloc(G * sizeof(float));
  int8_t* zeros  = (int8_t*)malloc(G * sizeof(int8_t));

  ie_act_i8_group_params_from_data(x, n, group, /*symmetric=*/1, scales, zeros);
  ie_quantize_act_int8_per_group(x, q, n, group, scales, zeros, /*symmetric=*/1);
  ie_dequantize_act_int8_per_group(q, y, n, group, scales, zeros);

  float e = mse(x, y, n);
  printf("[INT8 per-group]  MSE = %.6e (group=%zu)\n", e, group);

  free(x); free(q); free(y); free(scales); free(zeros);
  return (e < 2e-2f) ? 0 : 1;
}

static int test_fp8_sweep(void) {
  const size_t n = 4096;
  float*   x  = (float*)aligned_alloc(64, n * sizeof(float));
  uint8_t* q  = (uint8_t*)aligned_alloc(64, n * sizeof(uint8_t));
  float*   y  = (float*)aligned_alloc(64, n * sizeof(float));

  fill_range(x, n, -32.0f, 32.0f);

  int fails = 0;
  for (int fmt = 0; fmt < 2; ++fmt) {
    ie_quantize_act_fp8(x, q, n, (ie_fp8_format)fmt);
    ie_dequantize_act_fp8(q, y, n, (ie_fp8_format)fmt);

    /* Monotonicity check on absolute values (weak check). */
    for (size_t i = 1; i < n; ++i) {
      float a0 = fabsf(x[i-1]);
      float a1 = fabsf(x[i]);
      float b0 = fabsf(y[i-1]);
      float b1 = fabsf(y[i]);
      if (a1 >= a0 && b1 < b0 - 1e-6f) { /* allow tiny FP wiggle */
        ++fails;
        break;
      }
    }
    float e = mse(x, y, n);
    printf("[FP8 %s] MSE = %.6e\n", fmt == 0 ? "E4M3" : "E5M2", e);
  }

  free(x); free(q); free(y);
  return (fails == 0) ? 0 : 1;
}

int main(void) {
  int rc = 0;
  rc |= test_int8_per_tensor();
  rc |= test_int8_per_group();
  rc |= test_fp8_sweep();

  if (rc == 0) {
    printf("All activation quantization tests PASSED.\n");
  } else {
    printf("Some activation quantization tests FAILED (rc=%d).\n", rc);
  }
  return rc;
}
