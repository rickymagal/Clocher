/* ============================================================================
 * File: engine/src/kernels/rmsnorm_cpu.c
 * ============================================================================
 */
/**
 * @file rmsnorm_cpu.c
 * @brief CPU implementation of RMSNorm for fp32 activations.
 *
 * @details
 * RMSNorm normalizes an activation vector by its root-mean-square (RMS) and then
 * applies a learned per-channel scale:
 *
 *   rms = sqrt((1/N) * sum_i x[i]^2 + eps)
 *   y[i] = (x[i] / rms) * w[i]
 *
 * This file provides a correctness-first implementation intended for the first
 * real GPT-OSS forward pass on CPU. It is allocation-free and safe to call in a
 * tight token loop.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <stddef.h>

#include "ie_kernels.h"

/**
 * @brief Compute RMSNorm on CPU for fp32 vectors.
 *
 * @details
 * This function performs:
 *  - mean-square accumulation with a double accumulator for improved stability,
 *  - a single sqrtf and reciprocal,
 *  - per-element scaling by the provided weight vector.
 *
 * The function is designed to be allocation-free and deterministic. It does not
 * attempt SIMD vectorization; higher-performance backends may be added later.
 *
 * @param x    Input vector of length @p n.
 * @param w    Per-channel scale vector of length @p n.
 * @param n    Number of elements in @p x and @p w.
 * @param eps  Small epsilon added inside the sqrt for numerical stability.
 * @param y    Output vector of length @p n (may alias @p x).
 * @return 0 on success, non-zero on invalid arguments.
 */
int ie_rmsnorm_cpu_f32(const float *x,
                       const float *w,
                       size_t n,
                       float eps,
                       float *y) {
  if (!x || !w || !y || n == 0) return -1;

  double ss = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double v = (double)x[i];
    ss += v * v;
  }

  const double mean = ss / (double)n;
  const float inv_rms = 1.0f / sqrtf((float)mean + eps);

  for (size_t i = 0; i < n; ++i) {
    y[i] = (x[i] * inv_rms) * w[i];
  }

  return 0;
}
