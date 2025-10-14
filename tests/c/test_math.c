/**
 * @file test_math.c
 * @brief Validate fast tanh approximation and libm path bounds.
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ie_math.h"

int main(void) {
  const size_t N = 1000;
  float *v0 = (float*)malloc(N * sizeof(float));
  float *v1 = (float*)malloc(N * sizeof(float));
  assert(v0 && v1);
  for (size_t i = 0; i < N; ++i) {
    float x = ((int)i - 500) * 0.01f; /* [-5, 5] */
    v0[i] = x; v1[i] = x;
  }
  ie_vec_tanh_f32(v0, N, 0);
  ie_vec_tanh_f32(v1, N, 1);
  for (size_t i = 0; i < N; ++i) {
    assert(v0[i] <= 1.0f && v0[i] >= -1.0f);
    assert(v1[i] <= 1.0f && v1[i] >= -1.0f);
    /* Approx error bound (loose): */
    float diff = fabsf(v0[i] - v1[i]);
    assert(diff < 0.05f);
  }
  free(v0); free(v1);
  printf("ok test_math\n");
  return 0;
}
