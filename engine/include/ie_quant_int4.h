/**
 * @file ie_quant_int4.h
 * @brief Public API for 4-bit (INT4) weight-only post-training quantization.
 *
 * This header declares a minimal, self-contained C API for exploring INT4
 * weight-only quantization of row-major 2D float32 matrices (e.g., linear
 * layer weight matrices). It provides:
 *
 *  - Per-tensor and per-row symmetric quantization in the integer range [-7, +7]
 *    with zero-point fixed at 0. (We deliberately avoid using -8 to keep the
 *    mapping symmetric around zero.)
 *  - Nibble packing: two signed 4-bit values per byte, stored as a biased
 *    unsigned nibble q_u = q + 8, where q ∈ [-7, +7]. The low nibble encodes
 *    column j (even index), the high nibble encodes column j + 1 (odd index).
 *  - Dequantization helpers.
 *  - Convenience utilities: row-bytes computation and simple error metrics.
 *
 * The API does not depend on any project-specific types. Callers supply plain
 * pointers and sizes. No dynamic memory allocation occurs inside; the caller
 * owns all buffers.
 *
 * ## Packing layout (row-major)
 * For a matrix of shape (rows, cols):
 *
 *   bytes_per_row = (cols + 1) / 2
 *
 * Each byte packs two 4-bit values:
 *   - dst[i]   = (q1_u << 4) | q0_u
 *   - q0_u     = (uint8_t)(q0 + 8)  // low nibble, column j
 *   - q1_u     = (uint8_t)(q1 + 8)  // high nibble, column j+1
 *
 * When cols is odd, the last high nibble of the row is set to the encoding of 0
 * (i.e., 8) to keep the representation canonical.
 *
 * ## Quantization rule
 *   scale = max(|W|) / 7
 *   q     = round(W / scale)
 *   q     = clamp(q, -7, +7)
 *
 * For per-row quantization, the scale is computed independently for each row.
 *
 * ## Error handling
 * All functions return 0 on success and a negative value on failure:
 *   - IE_INT4_STATUS_OK                (0)
 *   - IE_INT4_STATUS_BADARG           (-1)
 *   - IE_INT4_STATUS_NUMERIC          (-2)
 *
 * @note This API is intended for benchmarking and integration experiments. For
 *       production use, consider adding robust metadata containers, alignment
 *       control, and kernel-side fused decode paths.
 */

#ifndef IE_QUANT_INT4_H
#define IE_QUANT_INT4_H

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint8_t */
#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup ie_int4_quant
 *  @{
 */

/**
 * @enum ie_int4_status_t
 * @brief Status codes for INT4 quantization routines.
 */
typedef enum ie_int4_status_t {
  IE_INT4_STATUS_OK = 0,      /**< Success. */
  IE_INT4_STATUS_BADARG = -1, /**< Invalid argument (NULL pointers, zero sizes, etc.). */
  IE_INT4_STATUS_NUMERIC = -2 /**< Numeric issue (NaN scales, infinities, etc.). */
} ie_int4_status_t;

/**
 * @brief Compute the number of packed bytes required to store a single row
 *        of @p cols 4-bit values (two per byte).
 *
 * @param cols  Number of columns (elements) in the row.
 * @return      Bytes required to store the packed row.
 */
size_t ie_int4_rowbytes(size_t cols);

/**
 * @brief Compute the absolute-maximum value of a float32 matrix.
 *
 * @param src   Pointer to the source matrix (row-major), length = rows*cols.
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @param out_absmax  Output pointer for the absolute-maximum value.
 * @return 0 on success, negative on error.
 */
int ie_int4_absmax(const float *src, size_t rows, size_t cols, float *out_absmax);

/**
 * @brief Compute per-row absolute-maximums of a float32 matrix.
 *
 * @param src     Pointer to the source matrix (row-major), length = rows*cols.
 * @param rows    Number of rows.
 * @param cols    Number of columns.
 * @param out_absmax_rows  Output array of length @p rows.
 * @return 0 on success, negative on error.
 */
int ie_int4_absmax_per_row(const float *src, size_t rows, size_t cols, float *out_absmax_rows);

/**
 * @brief Quantize a float32 matrix to packed INT4 using a single (per-tensor) scale.
 *
 * The scale is computed as max(|W|)/7. If the matrix is all zeros, the scale is
 * set to 1.0f and the output is all zeros.
 *
 * @param src       Pointer to the source matrix (row-major), length = rows*cols.
 * @param rows      Number of rows.
 * @param cols      Number of columns.
 * @param dst_packed  Output buffer for packed INT4 data. Must be at least
 *                    rows * ie_int4_rowbytes(cols) bytes.
 * @param out_scale   Output pointer to receive the per-tensor scale.
 * @return 0 on success, negative on error.
 */
int ie_int4_quantize_per_tensor(const float *src,
                                size_t rows,
                                size_t cols,
                                uint8_t *dst_packed,
                                float *out_scale);

/**
 * @brief Dequantize a packed INT4 matrix using a single (per-tensor) scale.
 *
 * @param src_packed  Pointer to packed INT4 input buffer of size
 *                    rows * ie_int4_rowbytes(cols) bytes.
 * @param rows        Number of rows.
 * @param cols        Number of columns.
 * @param scale       Per-tensor scale that was used during quantization.
 * @param dst         Output buffer of length rows*cols (float32).
 * @return 0 on success, negative on error.
 */
int ie_int4_dequantize_per_tensor(const uint8_t *src_packed,
                                  size_t rows,
                                  size_t cols,
                                  float scale,
                                  float *dst);

/**
 * @brief Quantize a float32 matrix to packed INT4 with per-row scales.
 *
 * Scales[i] = max(|row_i|)/7. For an all-zero row, the scale is 1.0f and the
 * packed output row is encoded as zeros.
 *
 * @param src         Pointer to the source matrix (row-major), length = rows*cols.
 * @param rows        Number of rows.
 * @param cols        Number of columns.
 * @param dst_packed  Output buffer for packed INT4 data. Must be at least
 *                    rows * ie_int4_rowbytes(cols) bytes.
 * @param out_scales  Output buffer of length @p rows to receive per-row scales.
 * @return 0 on success, negative on error.
 */
int ie_int4_quantize_per_row(const float *src,
                             size_t rows,
                             size_t cols,
                             uint8_t *dst_packed,
                             float *out_scales);

/**
 * @brief Dequantize a packed INT4 matrix with per-row scales.
 *
 * @param src_packed  Pointer to packed INT4 input buffer of size
 *                    rows * ie_int4_rowbytes(cols) bytes.
 * @param rows        Number of rows.
 * @param cols        Number of columns.
 * @param scales      Pointer to per-row scales (length = rows).
 * @param dst         Output buffer of length rows*cols (float32).
 * @return 0 on success, negative on error.
 */
int ie_int4_dequantize_per_row(const uint8_t *src_packed,
                               size_t rows,
                               size_t cols,
                               const float *scales,
                               float *dst);

/**
 * @brief Compute error metrics between two float32 vectors of equal length:
 *        mean squared error (MSE) and cosine similarity.
 *
 * @param ref        Pointer to the reference vector.
 * @param test       Pointer to the test vector.
 * @param n          Number of elements.
 * @param out_mse    Output pointer for mean squared error (double).
 * @param out_cosine Output pointer for cosine similarity (double, in [-1,1]).
 * @return 0 on success, negative on error.
 */
int ie_int4_error_metrics(const float *ref,
                          const float *test,
                          size_t n,
                          double *out_mse,
                          double *out_cosine);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_QUANT_INT4_H */
