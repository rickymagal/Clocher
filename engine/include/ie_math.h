/**
 * @file ie_math.h
 * @brief Math helper routines and fast approximations.
 *
 * These helpers provide drop-in scalar/vector math operations that can
 * be tuned for accuracy vs. speed at runtime.
 */

#ifndef IE_MATH_H
#define IE_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply hyperbolic tangent to a FP32 vector (in-place).
 *
 * When @p fast is `0`, the function uses `tanhf` from libm for full accuracy.
 * When @p fast is non-zero, it uses a polynomial approximation that trades
 * tiny accuracy for speed (suitable when precision is reduced, e.g., bf16/fp16).
 *
 * @param[in,out] v    Vector pointer (length @p n).
 * @param[in]     n    Number of elements.
 * @param[in]     fast Non-zero to use the fast approximation, 0 for libm.
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast);

#ifdef __cplusplus
}
#endif

#endif /* IE_MATH_H */
