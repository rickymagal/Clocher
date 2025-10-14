/**
 * @file fast_tanh.c
 * @brief Vector tanh with optional polynomial fast approximation.
 */

#include "ie_math.h"
#include <math.h>

/**
 * @brief Fast scalar tanh approximation for |x| <= ~3.
 *
 * Uses a rational/polynomial-like form with small max error in the central
 * region. Values are softly clamped to [-1, 1].
 *
 * @param x  Input value.
 * @return   Approximation to tanh(x).
 */
static inline float tanh_fast_scalar(float x) {
  const float x2 = x * x;
  const float p  = x * (27.0f + x2) / (27.0f + 9.0f * x2);
  if (p > 1.0f)  return 1.0f;
  if (p < -1.0f) return -1.0f;
  return p;
}

/**
 * @brief Apply tanh to a vector, optionally using a fast approximation.
 *
 * @param v     Pointer to the vector to transform (in-place).
 * @param n     Number of elements in @p v.
 * @param fast  Non-zero to use the approximate path; 0 for libm::tanhf.
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast) {
  if (!v || n == 0) return;
  if (!fast) {
    for (size_t i = 0; i < n; ++i) v[i] = tanhf(v[i]);
  } else {
    for (size_t i = 0; i < n; ++i) v[i] = tanh_fast_scalar(v[i]);
  }
}
