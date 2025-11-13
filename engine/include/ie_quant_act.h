#ifndef IE_QUANT_ACT_H
#define IE_QUANT_ACT_H

/**
 * @file ie_quant_act.h
 * @brief Activation quantization utilities (INT8 and FP8) for runtime paths.
 *
 * This module provides fast, allocation-free routines to quantize and
 * dequantize activations to INT8 and FP8 formats. It supports both
 * per-tensor and per-group (blockwise) parameterization for INT8, and
 * two FP8 formats (E4M3 and E5M2).
 *
 * Design goals:
 * - No dynamic allocation; all buffers owned by caller.
 * - Cache-friendly traversal with plain pointer arithmetic.
 * - Deterministic, saturating conversions.
 * - Small and easily inlinable helpers for hot loops.
 *
 * Notes on FP8:
 * - FP8 here implements software encoders/decoders for E4M3 and E5M2.
 *   The E4M3 variant is treated as finite-only (overflow saturates to
 *   the largest finite representable value). E5M2 supports Inf/NaN
 *   round-tripping from IEEE-754 float where possible; non-finite inputs
 *   to E4M3 are saturated to max finite.
 *
 * Thread-safety: all functions are pure with respect to input pointers
 * and are thread-safe if buffers do not alias.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @enum ie_fp8_format
 *  @brief FP8 format selector.
 */
typedef enum ie_fp8_format {
  IE_FP8_E4M3 = 0,  /**< 1 sign, 4 exponent (bias 7), 3 mantissa, finite-only saturation. */
  IE_FP8_E5M2 = 1   /**< 1 sign, 5 exponent (bias 15), 2 mantissa, Inf/NaN representable. */
} ie_fp8_format;

/** @struct ie_act_i8_params
 *  @brief Per-tensor INT8 affine parameters.
 *
 * The dequantization model is:
 *    real = scale * (q - zero_point)
 * The quantization model is the inverse with saturating rounding.
 */
typedef struct ie_act_i8_params {
  float  scale;       /**< Positive real scaling factor. */
  int8_t zero_point;  /**< Affine shift (0 for symmetric). */
} ie_act_i8_params;

/**
 * @brief Compute per-tensor INT8 parameters from min/max.
 *
 * The returned scale is strictly positive. If @p symmetric is non-zero,
 * zero_point is set to 0 and the scale is derived from max(|minv|,|maxv|)/127.
 * Otherwise, asymmetric parameters are derived to map minv→-128 and maxv→127.
 *
 * @param minv       Observed minimum (may be negative).
 * @param maxv       Observed maximum (may be positive).
 * @param symmetric  Non-zero for symmetric quantization.
 * @param out_scale  Output: scale.
 * @param out_zero   Output: zero point.
 */
void ie_act_i8_params_from_minmax(float minv, float maxv, int symmetric,
                                  float* out_scale, int8_t* out_zero);

/**
 * @brief Quantize activations to INT8 using per-tensor parameters.
 *
 * @param src         Pointer to source floats.
 * @param dst         Pointer to destination INT8 (same length as src).
 * @param n           Number of elements.
 * @param params      Quantization parameters (scale/zero_point).
 * @param symmetric   Non-zero if symmetric (enforces [-127,127] range).
 */
void ie_quantize_act_int8(const float* src, int8_t* dst, size_t n,
                          ie_act_i8_params params, int symmetric);

/**
 * @brief Dequantize INT8 activations to float using per-tensor parameters.
 *
 * @param src         Pointer to INT8 source.
 * @param dst         Pointer to float destination.
 * @param n           Number of elements.
 * @param params      Quantization parameters (scale/zero_point).
 */
void ie_dequantize_act_int8(const int8_t* src, float* dst, size_t n,
                            ie_act_i8_params params);

/**
 * @brief Compute per-group INT8 parameters for @p src.
 *
 * Partitions @p src into contiguous groups of size @p group_size
 * (the last group may be shorter) and computes parameters for each group
 * using @ref ie_act_i8_params_from_minmax with the chosen symmetry.
 *
 * @param src         Source floats.
 * @param n           Number of elements.
 * @param group_size  Group size (>=1).
 * @param symmetric   Non-zero for symmetric quantization.
 * @param out_scales  Output array of length ceil(n/group_size).
 * @param out_zeros   Output array of length ceil(n/group_size).
 */
void ie_act_i8_group_params_from_data(const float* src, size_t n,
                                      size_t group_size, int symmetric,
                                      float* out_scales, int8_t* out_zeros);

/**
 * @brief Quantize activations per-group to INT8.
 *
 * @param src         Source floats.
 * @param dst         Destination INT8.
 * @param n           Number of elements.
 * @param group_size  Group size (>=1).
 * @param scales      Array of scales (ceil(n/group_size)).
 * @param zeros       Array of zero_points (ceil(n/group_size)).
 * @param symmetric   Non-zero for symmetric (-127..127).
 */
void ie_quantize_act_int8_per_group(const float* src, int8_t* dst, size_t n,
                                    size_t group_size,
                                    const float* scales, const int8_t* zeros,
                                    int symmetric);

/**
 * @brief Dequantize per-group INT8 activations to float.
 *
 * @param src         Source INT8.
 * @param dst         Destination float.
 * @param n           Number of elements.
 * @param group_size  Group size (>=1).
 * @param scales      Array of scales (ceil(n/group_size)).
 * @param zeros       Array of zero_points (ceil(n/group_size)).
 */
void ie_dequantize_act_int8_per_group(const int8_t* src, float* dst, size_t n,
                                      size_t group_size,
                                      const float* scales, const int8_t* zeros);

/**
 * @brief Quantize activations to FP8 (E4M3 or E5M2).
 *
 * Non-finite inputs:
 * - E4M3: saturated to max finite.
 * - E5M2: pass through to Inf/NaN encodings when possible.
 *
 * @param src         Source floats.
 * @param dst         Destination bytes (FP8 bit-patterns).
 * @param n           Number of elements.
 * @param fmt         FP8 format selector.
 */
void ie_quantize_act_fp8(const float* src, uint8_t* dst, size_t n,
                         ie_fp8_format fmt);

/**
 * @brief Dequantize FP8 bytes to floats (E4M3 or E5M2).
 *
 * @param src         Source bytes (FP8 bit-patterns).
 * @param dst         Destination floats.
 * @param n           Number of elements.
 * @param fmt         FP8 format selector.
 */
void ie_dequantize_act_fp8(const uint8_t* src, float* dst, size_t n,
                           ie_fp8_format fmt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_QUANT_ACT_H */
