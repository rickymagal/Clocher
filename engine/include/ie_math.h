/**
 * @file ie_math.h
 * @brief Math utilities for activation functions and vector helpers.
 */

#ifndef IE_MATH_H_
#define IE_MATH_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * @brief Fast approximation of hyperbolic tanh with strict [-1, 1] range.
 *
 * @param x Input value.
 * @return Approximated tanh(x) in [-1.0f, 1.0f].
 */
float ie_fast_tanhf(float x);

/**
 * @brief In-place vector tanh on a contiguous array (legacy API).
 *
 * When @p fast_tanh != 0, uses #ie_fast_tanhf; otherwise uses the standard
 * library `tanhf` for higher accuracy.
 *
 * @param v         Pointer to the vector (length @p n). Modified in-place.
 * @param n         Number of elements in @p v.
 * @param fast_tanh Non-zero to use the fast approximation; 0 to use libm.
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh);

/**
 * @brief Out-of-place vector tanh on a contiguous array.
 *
 * Always uses the fast approximation (#ie_fast_tanhf). Output is strictly
 * clamped to [-1, 1].
 *
 * @param x    Pointer to input array (length @p n).
 * @param out  Pointer to output array (length @p n).
 * @param n    Number of elements.
 */
void ie_vec_tanh_f32_out(const float *x, float *out, size_t n);

/**
 * @brief Out-of-place vector tanh on strided arrays.
 *
 * Computes for i in [0, n): out[i*out_stride] = tanh(x[i*x_stride]).
 * Always uses the fast approximation with strict clamping.
 *
 * @param x          Pointer to input array.
 * @param x_stride   Stride (in elements) between consecutive inputs.
 * @param out        Pointer to output array.
 * @param out_stride Stride (in elements) between consecutive outputs.
 * @param n          Number of elements to process.
 */
void ie_vec_tanh_f32_strided(const float *x, size_t x_stride,
                             float *out, size_t out_stride,
                             size_t n);

#ifdef __cplusplus
}
#endif

#endif /* IE_MATH_H_ */
