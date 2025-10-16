/**
 * @file int8_ptq.c
 * @brief Implementation of INT8 PTQ min-max scaling and (de)quantization.
 *
 * This translation unit provides a minimal, reproducible Post-Training
 * Quantization (PTQ) path for weight-only INT8 using symmetric min-max
 * scaling. It is backend-agnostic and dependency-free, suitable for both
 * CPU and GPU/iGPU pipelines that only need offline quantization artifacts.
 */

#include "ie_quant.h"

#include <float.h>   /* FLT_MAX */
#include <math.h>    /* fabsf, lroundf, fmaxf */
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Clamp a float value to the symmetric INT8 range and round to nearest.
 *
 * The symmetric INT8 range is taken as [-127, +127] (note: not including
 * -128) to avoid asymmetric saturation that can bias dequantization.
 *
 * @param x Input value as 32-bit float.
 * @return int8_t The clamped and rounded integer in [-127, +127].
 */
static inline int8_t clamp_sym_int8(float x) {
  if (x > 127.0f)  return 127;
  if (x < -127.0f) return -127;
  return (int8_t)lroundf(x);
}

/**
 * @brief Compute symmetric min-max scales for INT8 PTQ.
 *
 * For each scaling domain @p D (the entire tensor when
 * IE_PTQ_SCALE_PER_TENSOR, or each row when IE_PTQ_SCALE_PER_ROW), this
 * routine computes:
 *
 * - @f$ a_{\max}(D) = \max(\lvert \min(D) \rvert, \lvert \max(D) \rvert) @f$
 * - @f$ \text{scale}(D) = \frac{a_{\max}(D)}{127.0} @f$
 *
 * If @f$ a_{\max}(D) = 0 @f$, the scale falls back to 1.0 to avoid division
 * by zero during quantization.
 *
 * @param[in]  w        Pointer to FP32 weights (row-major), length = rows*cols.
 * @param[in]  rows     Number of rows (use 1 for flat vectors).
 * @param[in]  cols     Number of columns (use N for flat vectors).
 * @param[in]  mode     Scale granularity (tensor- or row-wise).
 * @param[out] scales   Output buffer for scales:
 *                      - length 1 when mode == IE_PTQ_SCALE_PER_TENSOR
 *                      - length rows when mode == IE_PTQ_SCALE_PER_ROW
 *
 * @note Inputs are not validated beyond null/zero checks for performance.
 *       Callers must ensure the buffers are correctly sized.
 */
void ie_ptq_compute_scales_minmax(const float *w,
                                  size_t rows,
                                  size_t cols,
                                  ie_ptq_scale_mode_t mode,
                                  float *scales) {
  if (!w || !scales || rows == 0 || cols == 0) return;

  if (mode == IE_PTQ_SCALE_PER_TENSOR) {
    float lo = FLT_MAX, hi = -FLT_MAX;
    const size_t n = rows * cols;
    for (size_t i = 0; i < n; ++i) {
      const float v = w[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const float amax = fmaxf(fabsf(lo), fabsf(hi));
    scales[0] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    return;
  }

  /* Per-row scaling. */
  for (size_t r = 0; r < rows; ++r) {
    float lo = FLT_MAX, hi = -FLT_MAX;
    const float *row = w + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      const float v = row[c];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const float amax = fmaxf(fabsf(lo), fabsf(hi));
    scales[r] = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
  }
}

/**
 * @brief Quantize FP32 weights to INT8 using provided scales.
 *
 * When mode == IE_PTQ_SCALE_PER_TENSOR, @p scales[0] is used for all values.
 * When mode == IE_PTQ_SCALE_PER_ROW, @p scales[r] is used for row @p r.
 * Values are multiplied by the reciprocal scale, rounded to nearest, and
 * clamped to the symmetric INT8 range [-127, 127].
 *
 * @param[in]  w        Pointer to FP32 weights (row-major), length = rows*cols.
 * @param[in]  rows     Number of rows.
 * @param[in]  cols     Number of columns.
 * @param[in]  mode     Granularity of scales (tensor or per-row).
 * @param[in]  scales   Scale buffer from ie_ptq_compute_scales_minmax.
 * @param[out] q8       Output INT8 buffer (row-major), length = rows*cols.
 *
 * @note This is a pure weight-only quantization step: no zero-points; the
 *       quantization is symmetric around zero.
 */
void ie_ptq_quantize_int8(const float *w,
                          size_t rows,
                          size_t cols,
                          ie_ptq_scale_mode_t mode,
                          const float *scales,
                          int8_t *q8) {
  if (!w || !q8 || !scales || rows == 0 || cols == 0) return;

  if (mode == IE_PTQ_SCALE_PER_TENSOR) {
    const float inv = (scales[0] > 0.0f) ? (1.0f / scales[0]) : 1.0f;
    const size_t n = rows * cols;
    for (size_t i = 0; i < n; ++i) {
      q8[i] = clamp_sym_int8(w[i] * inv);
    }
    return;
  }

  /* Per-row scaling. */
  for (size_t r = 0; r < rows; ++r) {
    const float s = scales[r];
    const float inv = (s > 0.0f) ? (1.0f / s) : 1.0f;
    const float *row = w + r * cols;
    int8_t *qrow = q8 + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      qrow[c] = clamp_sym_int8(row[c] * inv);
    }
  }
}

/**
 * @brief Dequantize INT8 weights back to FP32 using provided scales.
 *
 * This function reverses ie_ptq_quantize_int8 by multiplying each INT8
 * element with its associated scale.
 *
 * @param[in]  q8       Pointer to INT8 weights (row-major), length = rows*cols.
 * @param[in]  rows     Number of rows.
 * @param[in]  cols     Number of columns.
 * @param[in]  mode     Granularity of scales (tensor or per-row).
 * @param[in]  scales   Scale buffer that was used during quantization.
 * @param[out] w_out    Output FP32 buffer (row-major), length = rows*cols.
 *
 * @note Dequantization is exact with respect to the chosen scale; any loss
 *       is due to the quantization rounding/clamping step.
 */
void ie_ptq_dequant_int8(const int8_t *q8,
                         size_t rows,
                         size_t cols,
                         ie_ptq_scale_mode_t mode,
                         const float *scales,
                         float *w_out) {
  if (!q8 || !w_out || !scales || rows == 0 || cols == 0) return;

  if (mode == IE_PTQ_SCALE_PER_TENSOR) {
    const float s = scales[0] > 0.0f ? scales[0] : 1.0f;
    const size_t n = rows * cols;
    for (size_t i = 0; i < n; ++i) {
      w_out[i] = (float)q8[i] * s;
    }
    return;
  }

  /* Per-row scaling. */
  for (size_t r = 0; r < rows; ++r) {
    const float s = scales[r] > 0.0f ? scales[r] : 1.0f;
    const int8_t *qrow = q8 + r * cols;
    float *row = w_out + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      row[c] = (float)qrow[c] * s;
    }
  }
}
