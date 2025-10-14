/**
 * @file fast_tanh.c
 * @brief Scalar/vector tanh implementations, including a fast approximation.
 *
 * This module exposes:
 *  - ie_fast_tanhf(x): a fast tanh approximation suitable for inference paths.
 *  - ie_vec_tanh_f32(v, n, fast): vector tanh (in-place), choosing between the
 *    standard libm tanhf() and the fast approximation.
 *
 * The fast approximation uses a clipped polynomial/rational form that provides
 * a good latency/accuracy trade-off for activation-like usage. For very large
 * magnitudes the result is saturated by the clamp prior to evaluation.
 */

#include <math.h>
#include <stddef.h>
#include "ie_math.h"

/**
 * @brief Fast scalar tanh approximation with input clamp.
 *
 * The approximation follows a clipped rational form that reduces the need for
 * libm while preserving monotonicity and the odd symmetry of tanh:
 *
 *   Let z = clamp(x, -3, +3).
 *   Use r(z) = z * (27 + z*z) / (27 + 9*z*z).
 *
 * The clamp limits error growth on large |x| where tanh(x) ~ ±1 anyway.
 *
 * @param x Input value.
 * @return Approximated tanh(x).
 */
float ie_fast_tanhf(float x) {
  /* Clamp to reduce error in tails and keep polynomial stable */
  float z = x;
  if (z > 3.0f)  z = 3.0f;
  if (z < -3.0f) z = -3.0f;

  /* Odd rational approximation: z * (27 + z^2) / (27 + 9 z^2) */
  const float z2 = z * z;
  const float num = z * (27.0f + z2);
  const float den = 27.0f + 9.0f * z2;
  return num / den;
}

/**
 * @brief In-place vector tanh for float32.
 *
 * When @p fast_tanh is non-zero, this function applies the fast approximation
 * ie_fast_tanhf() element-wise. Otherwise it falls back to the standard tanhf()
 * from libm for full accuracy.
 *
 * @param v         Pointer to the input/output vector (length @p n).
 * @param n         Number of elements in the vector.
 * @param fast_tanh Non-zero to use the fast approximation; zero to use tanhf().
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh) {
  if (!v || n == 0) return;

  if (fast_tanh) {
    for (size_t i = 0; i < n; ++i) {
      v[i] = ie_fast_tanhf(v[i]);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      v[i] = tanhf(v[i]);
    }
  }
}
