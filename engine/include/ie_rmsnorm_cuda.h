/**
 * @file ie_rmsnorm_cuda.h
 * @brief C ABI wrapper for CUDA RMSNorm (FP32).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief RMSNorm on device: y = (x / rms) * w.
 *
 * @param x    Device pointer to input, [rows, cols].
 * @param w    Device pointer to weight (cols) or NULL.
 * @param y    Device pointer to output, [rows, cols].
 * @param rows Rows (usually 1).
 * @param cols Cols (feature dim).
 * @param eps  Epsilon.
 * @return 0 on success, negative on error.
 */
int ie_rmsnorm_cuda_f32(const float *x, const float *w, float *y,
                        size_t rows, size_t cols, float eps);

#ifdef __cplusplus
} /* extern "C" */
#endif
