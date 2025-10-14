/**
 * @file fast_tanh.c
 * @brief Fast tanh approximation (scalar and vector forms) with strict [-1, 1] clamping.
 *
 * The scalar uses a cheap rational approximation for moderate |x| and early
 * saturation for large |x|. Vector helpers call the scalar per element to
 * keep the implementation simple and portable; SIMD paths can be added later.
 */

#include "ie_math.h"

#include <math.h>
#include <stddef.h>

/**
 * @brief Fast approximation of hyperbolic tangent with hard clamping.
 *
 * For large |x| we short-circuit to +/-1.0f. For moderate |x| we use a
 * small rational function that gives a smooth S-curve:
 *
 *     tanh(x) ≈ x * (27 + x^2) / (27 + 9 x^2)
 *
 * Finally, the result is clamped to [-1, 1] to guarantee range correctness.
 *
 * @param x Input value.
 * @return Approximated tanh(x) in [-1.0f, 1.0f].
 */
float ie_fast_tanhf(float x) {
  /* Early clamp for large magnitude to avoid overflow and extra work. */
  if (x > 5.0f)  return 1.0f;
  if (x < -5.0f) return -1.0f;

  /* Rational approximation (cheap and monotonic on [-3,3]) */
  const float x2  = x * x;
  const float num = x * (27.0f + x2);
  const float den = 27.0f + 9.0f * x2;
  float y = num / den;

  /* Safety clamp for strict range compliance. */
  if (y > 1.0f)  y = 1.0f;
  if (y < -1.0f) y = -1.0f;
  return y;
}

/**
 * @brief In-place vector tanh on a contiguous array (legacy API).
 *
 * If @p fast_tanh != 0, uses #ie_fast_tanhf; otherwise uses libm `tanhf`.
 * Output values are clamped to [-1, 1] in both modes to satisfy strict bounds.
 *
 * @param v         Pointer to the vector (length @p n). Modified in-place.
 * @param n         Number of elements in @p v.
 * @param fast_tanh Non-zero to use the fast approximation; 0 to use libm.
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh) {
  if (!v || n == 0) return;
  if (fast_tanh) {
    for (size_t i = 0; i < n; ++i) {
      v[i] = ie_fast_tanhf(v[i]);
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      float y = tanhf(v[i]);
      if (y > 1.0f)  y = 1.0f;
      if (y < -1.0f) y = -1.0f;
      v[i] = y;
    }
  }
}

/**
 * @brief Out-of-place vector tanh on a contiguous array (fast approximation).
 *
 * @param x    Pointer to input array (length @p n).
 * @param out  Pointer to output array (length @p n).
 * @param n    Number of elements.
 */
void ie_vec_tanh_f32_out(const float *x, float *out, size_t n) {
  if (!x || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) {
    out[i] = ie_fast_tanhf(x[i]);
  }
}

/**
 * @brief Out-of-place vector tanh on strided arrays (fast approximation).
 *
 * Computes for i in [0, n): out[i*out_stride] = tanh(x[i*x_stride]).
 *
 * @param x          Pointer to input array.
 * @param x_stride   Stride (in elements) between consecutive inputs.
 * @param out        Pointer to output array.
 * @param out_stride Stride (in elements) between consecutive outputs.
 * @param n          Number of elements to process.
 */
void ie_vec_tanh_f32_strided(const float *x, size_t x_stride,
                             float *out, size_t out_stride,
                             size_t n) {
  if (!x || !out || n == 0) return;
  for (size_t i = 0; i < n; ++i) {
    out[i * out_stride] = ie_fast_tanhf(x[i * x_stride]);
  }
}
