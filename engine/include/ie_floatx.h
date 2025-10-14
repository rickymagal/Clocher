/**
 * @file ie_floatx.h
 * @brief FP32↔BF16/FP16 conversion helpers (vector forms).
 *
 * All routines are pure and do not allocate. They operate on contiguous arrays.
 * Rounding:
 *  - FP32→BF16 uses round-to-nearest-even (ties to even).
 *  - FP32→FP16 uses IEEE 754 round-to-nearest-even; infinities/NaNs preserved.
 */

#ifndef IE_FLOATX_H_
#define IE_FLOATX_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Convert an array of float (FP32) to bfloat16 (BF16).
 *
 * @param in   Pointer to FP32 input array of length @p n.
 * @param out  Pointer to BF16 output array of length @p n (uint16_t per elem).
 * @param n    Number of elements to convert.
 */
void ie_fp32_to_bf16(const float *in, uint16_t *out, size_t n);

/**
 * @brief Convert an array of bfloat16 (BF16) to float (FP32).
 *
 * @param in   Pointer to BF16 input array of length @p n (uint16_t per elem).
 * @param out  Pointer to FP32 output array of length @p n.
 * @param n    Number of elements to convert.
 */
void ie_bf16_to_fp32(const uint16_t *in, float *out, size_t n);

/**
 * @brief Convert an array of float (FP32) to IEEE half (FP16).
 *
 * @param in   Pointer to FP32 input array of length @p n.
 * @param out  Pointer to FP16 output array of length @p n (uint16_t per elem).
 * @param n    Number of elements to convert.
 */
void ie_fp32_to_fp16(const float *in, uint16_t *out, size_t n);

/**
 * @brief Convert an array of IEEE half (FP16) to float (FP32).
 *
 * @param in   Pointer to FP16 input array of length @p n (uint16_t per elem).
 * @param out  Pointer to FP32 output array of length @p n.
 * @param n    Number of elements to convert.
 */
void ie_fp16_to_fp32(const uint16_t *in, float *out, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* IE_FLOATX_H_ */
