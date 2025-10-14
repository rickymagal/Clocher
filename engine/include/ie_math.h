/**
 * @file ie_math.h
 * @brief Math helpers (vector/scalar tanh).
 */
#ifndef IE_MATH_H_
#define IE_MATH_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Vector tanh on float data (fp32).
 *
 * @param v         Pointer to input/output vector (in-place).
 * @param n         Number of elements.
 * @param fast_tanh Non-zero to use a fast approximation; zero to use tanhf().
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh);

/**
 * @brief Fast scalar tanh approximation used in fused loops.
 *
 * @param x Input value.
 * @return tanh(x) approximated by a polynomial/rational form.
 */
float ie_fast_tanhf(float x);

#ifdef __cplusplus
}
#endif

#endif /* IE_MATH_H_ */
