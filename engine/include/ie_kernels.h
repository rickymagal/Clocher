/**
 * @file ie_kernels.h
 * @brief CPU kernels and runtime dispatch interface.
 *
 * The module exposes a stable function name for GEMV (ie_gemv_f32) and
 * a one-time installer (ie_kernels_install) that selects the best available
 * implementation (e.g., AVX2) according to runtime checks.
 */

#ifndef IE_KERNELS_H
#define IE_KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Row-major GEMV (FP32): y = W[rows x cols] * x[cols].
 *
 * The active implementation (generic or AVX2) is selected via
 * ::ie_kernels_install at engine creation time.
 *
 * @param[in]  W     Row-major weight matrix of shape [rows x cols].
 * @param[in]  x     Input vector of length @p cols.
 * @param[out] y     Output vector of length @p rows.
 * @param[in]  rows  Number of rows (outputs).
 * @param[in]  cols  Number of columns (inputs).
 */
void ie_gemv_f32(const float *W,
                 const float *x,
                 float *y,
                 size_t rows,
                 size_t cols);

/**
 * @brief Install the best available kernel set for this process.
 *
 * Call this **once** at startup (e.g., during engine creation) after
 * detecting CPU features. If @p enable_avx2 is non-zero and the binary was
 * compiled with AVX2 support, the dispatcher will switch to the AVX2 kernel.
 * Otherwise, it falls back to the generic C implementation.
 *
 * @param[in] enable_avx2  Non-zero to enable AVX2 path when available.
 */
void ie_kernels_install(int enable_avx2);

#ifdef __cplusplus
}
#endif

#endif /* IE_KERNELS_H */
