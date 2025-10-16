/**
 * @file ie_quant.h
 * @brief Post-Training Quantization (PTQ) helpers for INT8 weight-only paths.
 *
 * This header declares utilities to compute scaling factors and to
 * (de)quantize FP32 weight tensors into INT8 using simple, reproducible
 * rules. The API is dependency-free and backend-agnostic so it can be used
 * from CPU and GPU code paths.
 */

#ifndef IE_QUANT_H
#define IE_QUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum ie_ptq_scale_mode_t
 * @brief Scale granularity for PTQ.
 */
typedef enum ie_ptq_scale_mode_e {
  IE_PTQ_SCALE_PER_TENSOR = 0, /**< Single scale for the entire tensor. */
  IE_PTQ_SCALE_PER_ROW    = 1  /**< One scale per row (for row-major 2D matrices). */
} ie_ptq_scale_mode_t;

/**
 * @brief Compute INT8 scales using a symmetric min-max rule.
 *
 * For each scale domain D (tensor or row), we compute:
 *   scale(D) = max(abs(min(D)), abs(max(D))) / 127.0f
 * If D is all zeros, the scale is set to 1.0f to avoid division by zero.
 *
 * @param[in]  w        Pointer to FP32 weights in row-major layout.
 * @param[in]  rows     Number of rows (use 1 for a flat vector).
 * @param[in]  cols     Number of columns (use N for a flat vector).
 * @param[in]  mode     Scale granularity to use.
 * @param[out] scales   Output buffer for scales:
 *                      - length 1 if mode == IE_PTQ_SCALE_PER_TENSOR
 *                      - length rows if mode == IE_PTQ_SCALE_PER_ROW
 */
void ie_ptq_compute_scales_minmax(const float *w,
                                  size_t rows,
                                  size_t cols,
                                  ie_ptq_scale_mode_t mode,
                                  float *scales);

/**
 * @brief Quantize FP32 weights to INT8 using precomputed scales.
 *
 * When mode == IE_PTQ_SCALE_PER_TENSOR, scales[0] is applied to all elements.
 * When mode == IE_PTQ_SCALE_PER_ROW, scales[r] applies to row r.
 * Values are clamped to the symmetric INT8 range [-127, 127].
 *
 * @param[in]  w        Pointer to FP32 weights (row-major).
 * @param[in]  rows     Number of rows.
 * @param[in]  cols     Number of columns.
 * @param[in]  mode     Scale granularity, must match @p scales shape.
 * @param[in]  scales   Buffer of scales computed by ie_ptq_compute_scales_minmax.
 * @param[out] q8       Output buffer for INT8 weights (length rows*cols).
 */
void ie_ptq_quantize_int8(const float *w,
                          size_t rows,
                          size_t cols,
                          ie_ptq_scale_mode_t mode,
                          const float *scales,
                          int8_t *q8);

/**
 * @brief Dequantize INT8 weights back to FP32 using provided scales.
 *
 * @param[in]  q8       Pointer to INT8 weights (row-major).
 * @param[in]  rows     Number of rows.
 * @param[in]  cols     Number of columns.
 * @param[in]  mode     Scale granularity, must match @p scales shape.
 * @param[in]  scales   Buffer of scales used on quantization.
 * @param[out] w_out    Output buffer for FP32 weights (length rows*cols).
 */
void ie_ptq_dequant_int8(const int8_t *q8,
                         size_t rows,
                         size_t cols,
                         ie_ptq_scale_mode_t mode,
                         const float *scales,
                         float *w_out);

#ifdef __cplusplus
}
#endif

#endif /* IE_QUANT_H */
