/* File: engine/include/ie_quant_act.h
 * -----------------------------------------------------------------------------
 * @file ie_quant_act.h
 * @brief Activation quantization utilities (INT8 and FP8) for runtime paths.
 *
 * @details
 * Allocation-free quantize/dequantize utilities for runtime activations.
 * Supports per-tensor and per-group INT8 affine quantization and software FP8
 * (E4M3, E5M2) encode/decode.
 */

#ifndef IE_QUANT_ACT_H
#define IE_QUANT_ACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ie_fp8_format
 * @brief FP8 format selector.
 */
typedef enum ie_fp8_format {
  /** 1 sign, 4 exponent (bias 7), 3 mantissa; finite-only saturation policy. */
  IE_FP8_E4M3 = 0,
  /** 1 sign, 5 exponent (bias 15), 2 mantissa; Inf/NaN representable. */
  IE_FP8_E5M2 = 1
} ie_fp8_format;

/**
 * @struct ie_act_i8_params
 * @brief Per-tensor INT8 affine parameters.
 *
 * Dequantization model:
 *   real = scale * (q - zero_point)
 */
typedef struct ie_act_i8_params {
  float  scale;       /**< Positive scaling factor. */
  int8_t zero_point;  /**< Affine shift (0 for symmetric). */
} ie_act_i8_params;

/**
 * @brief Compute per-tensor INT8 parameters from observed min/max.
 *
 * @details
 * If symmetric != 0:
 * - zero_point is set to 0
 * - scale is derived from max(|minv|,|maxv|)/127
 *
 * If symmetric == 0:
 * - parameters map minv -> -128 and maxv -> 127 using an affine transform
 *
 * @param minv       Observed minimum.
 * @param maxv       Observed maximum.
 * @param symmetric  Non-zero for symmetric quantization.
 * @param out_scale  Output scale (non-NULL).
 * @param out_zero   Output zero point (non-NULL).
 */
void ie_act_i8_params_from_minmax(float minv, float maxv, int symmetric,
                                  float* out_scale, int8_t* out_zero);

/**
 * @brief Quantize fp32 activations to INT8 using per-tensor parameters.
 *
 * @param src        Source float buffer.
 * @param dst        Destination INT8 buffer (length n).
 * @param n          Number of elements.
 * @param params     Quantization parameters.
 * @param symmetric  Non-zero to enforce symmetric saturation policy.
 */
void ie_quantize_act_int8(const float* src, int8_t* dst, size_t n,
                          ie_act_i8_params params, int symmetric);

/**
 * @brief Dequantize INT8 activations to fp32 using per-tensor parameters.
 *
 * @param src    Source INT8 buffer.
 * @param dst    Destination float buffer (length n).
 * @param n      Number of elements.
 * @param params Dequantization parameters.
 */
void ie_dequantize_act_int8(const int8_t* src, float* dst, size_t n,
                            ie_act_i8_params params);

/**
 * @brief Compute per-group INT8 parameters from data.
 *
 * @details
 * Splits src into contiguous groups of group_size (last group may be shorter)
 * and computes parameters for each group.
 *
 * @param src         Source float buffer.
 * @param n           Number of elements.
 * @param group_size  Group size (>= 1).
 * @param symmetric   Non-zero for symmetric quantization.
 * @param out_scales  Output scales array (ceil(n/group_size)).
 * @param out_zeros   Output zero-points array (ceil(n/group_size)).
 */
void ie_act_i8_group_params_from_data(const float* src, size_t n,
                                      size_t group_size, int symmetric,
                                      float* out_scales, int8_t* out_zeros);

/**
 * @brief Quantize fp32 activations to INT8 using per-group parameters.
 *
 * @param src         Source float buffer.
 * @param dst         Destination INT8 buffer.
 * @param n           Number of elements.
 * @param group_size  Group size (>= 1).
 * @param scales      Scales array (ceil(n/group_size)).
 * @param zeros       Zero-points array (ceil(n/group_size)).
 * @param symmetric   Non-zero to enforce symmetric saturation policy.
 */
void ie_quantize_act_int8_per_group(const float* src, int8_t* dst, size_t n,
                                    size_t group_size,
                                    const float* scales, const int8_t* zeros,
                                    int symmetric);

/**
 * @brief Dequantize per-group INT8 activations to fp32.
 *
 * @param src         Source INT8 buffer.
 * @param dst         Destination float buffer.
 * @param n           Number of elements.
 * @param group_size  Group size (>= 1).
 * @param scales      Scales array (ceil(n/group_size)).
 * @param zeros       Zero-points array (ceil(n/group_size)).
 */
void ie_dequantize_act_int8_per_group(const int8_t* src, float* dst, size_t n,
                                      size_t group_size,
                                      const float* scales, const int8_t* zeros);

/**
 * @brief Quantize fp32 activations to FP8.
 *
 * @details
 * - E4M3: non-finite inputs saturate to max finite.
 * - E5M2: attempts to preserve Inf/NaN encodings.
 *
 * @param src Source float buffer.
 * @param dst Destination FP8 bytes (bit patterns).
 * @param n   Number of elements.
 * @param fmt FP8 format selector.
 */
void ie_quantize_act_fp8(const float* src, uint8_t* dst, size_t n,
                         ie_fp8_format fmt);

/**
 * @brief Dequantize FP8 bytes to fp32.
 *
 * @param src Source FP8 bytes (bit patterns).
 * @param dst Destination float buffer.
 * @param n   Number of elements.
 * @param fmt FP8 format selector.
 */
void ie_dequantize_act_fp8(const uint8_t* src, float* dst, size_t n,
                           ie_fp8_format fmt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_QUANT_ACT_H */

