/**
 * @file ie_kernels.h
 * @brief Kernel dispatch points (generic/AVX2) for GEMV and vector ops.
 */
#ifndef IE_KERNELS_H_
#define IE_KERNELS_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Install the best available CPU kernels at runtime.
 *
 * @param use_avx2 Non-zero to select AVX2-optimized paths where available.
 */
void ie_kernels_install(int use_avx2);

/**
 * @brief GEMV: y[r] = dot(W[r, :], x) [+ bias[r]] for r in [0, rows).
 *
 * The function chooses the best implementation (AVX2 or generic) selected by
 * ie_kernels_install(). The @p bias pointer may be NULL to skip epilogue bias.
 *
 * @param W     Pointer to weights in row-major layout or pretransposed-blocked
 *              layout accepted by the active kernel (see `ie_layout.h`).
 * @param x     Pointer to input vector of length @p cols.
 * @param y     Pointer to output vector of length @p rows.
 * @param rows  Number of rows to process.
 * @param cols  Number of columns per row.
 * @param bias  Optional pointer to bias vector of length @p rows; may be NULL.
 * @param blk_k Column-block size if @p W is in blocked-K layout; pass 0 for
 *              plain row-major (kernel will treat as unblocked).
 */
void ie_gemv_f32(const float *W, const float *x, float *y,
                 size_t rows, size_t cols,
                 const float *bias, size_t blk_k);

/**
 * @brief Vector tanh on float data (fp32).
 *
 * @param v         Pointer to input/output vector (in-place).
 * @param n         Number of elements.
 * @param fast_tanh Non-zero to use a fast approximation; zero to use tanhf().
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh);

/**
 * @brief Fast scalar tanh approximation (used for fused bias+tanh paths).
 *
 * @param x Input value.
 * @return tanh(x) approximated with a polynomial/rational form.
 */
float ie_fast_tanhf(float x);

#ifdef __cplusplus
}
#endif

#endif /* IE_KERNELS_H_ */
