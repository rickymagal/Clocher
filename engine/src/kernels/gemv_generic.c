/* File: engine/src/kernels/gemv_generic.c
 * -----------------------------------------------------------------------------
 * @file gemv_generic.c
 * @brief Portable GEMV kernels and the public GEMV API/dispatcher.
 *
 * @details
 * Owns the public entry points:
 * - ::ie_kernels_install
 * - ::ie_gemv_f32
 * - ::ie_gemv_qi8_f32
 * - ::ie_gemv_qi8pg_f32
 * - ::ie_gemv_qfp8_f32
 *
 * Optional ISA-specific implementations (e.g., AVX2) may be linked from other
 * translation units. Dispatch is decided at runtime via CPU feature checks.
 */

#include "ie_kernels.h"
#include "ie_quant_act.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Optional arch-specific implementations (resolved at link time if present). */
/* -------------------------------------------------------------------------- */

/**
 * @brief AVX2/FMA GEMV (fp32), provided by gemv_avx2.c.
 *
 * @param W     Weights (row-major or blocked-K contiguous per row).
 * @param x     Input vector (length cols).
 * @param y     Output vector (length rows).
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @param bias  Optional bias (length rows) or NULL.
 * @param blk_k Column block size; 0 disables blocking.
 */
void ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y,
                           size_t rows, size_t cols, const float *bias,
                           size_t blk_k);

/**
 * @brief AVX2/FMA GEMV with per-tensor INT8 activations, provided by gemv_avx2.c.
 *
 * @param W         Weights (float).
 * @param x_q       INT8 activations (length cols).
 * @param y         Output (float, length rows).
 * @param rows      Number of rows.
 * @param cols      Number of columns.
 * @param bias      Optional bias or NULL.
 * @param blk_k     Column block size; 0 disables blocking.
 * @param params    Per-tensor INT8 parameters.
 * @param symmetric Informational flag.
 */
void ie_gemv_qi8_f32_avx2_impl(const float *W, const int8_t *x_q, float *y,
                               size_t rows, size_t cols, const float *bias,
                               size_t blk_k, ie_act_i8_params params,
                               int symmetric);

/* -------------------------------------------------------------------------- */
/* Internal installed function pointer (fp32 path).                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Installed fp32 GEMV implementation.
 *
 * @details
 * Set by ::ie_kernels_install. ::ie_gemv_f32 will call install lazily if needed.
 */
static void (*s_gemv_f32)(const float *, const float *, float *, size_t, size_t,
                          const float *, size_t) = NULL;

/* ========================================================================== */
/*                               Generic C kernels                             */
/* ========================================================================== */

/**
 * @brief Generic C GEMV: y = W * x (+bias), optional blocked-K traversal.
 *
 * @details
 * Baseline portable implementation. W is treated as row-major with rows*cols
 * elements. When blk_k > 0, the inner loop traverses columns in blocks of BK.
 *
 * @param W     Weights (row-major).
 * @param x     Input vector (length cols).
 * @param y     Output vector (length rows).
 * @param rows  Rows.
 * @param cols  Cols.
 * @param bias  Optional bias (length rows) or NULL.
 * @param blk_k Column block size; 0 disables blocking.
 */
static void gemv_generic_impl(const float *W, const float *x, float *y,
                              size_t rows, size_t cols, const float *bias,
                              size_t blk_k) {
  const size_t BK = (blk_k > 0 ? blk_k : cols);

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    float acc = 0.0f;

    size_t kofs = 0;
    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + kofs;
      const float *xblk = x + k0;

      for (size_t k = 0; k < klen; ++k) acc += wblk[k] * xblk[k];
      kofs += klen;
    }

    if (bias) acc += bias[r];
    y[r] = acc;
  }
}

/**
 * @brief GEMV with per-tensor INT8 activations (fused dequantization), C path.
 *
 * @details
 * Interprets INT8 activations with:
 *   real = params.scale * (q - params.zero_point)
 * and multiplies float weights by decoded values without materializing x.
 *
 * @param W         Weights (float, row-major).
 * @param x_q       INT8 activations (length cols).
 * @param y         Output (float, length rows).
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias or NULL.
 * @param blk_k     Column block size; 0 disables blocking.
 * @param params    Per-tensor INT8 parameters.
 * @param symmetric Informational flag (unused).
 */
static void gemv_qi8_f32_fused_c_impl(const float *W, const int8_t *x_q,
                                      float *y, size_t rows, size_t cols,
                                      const float *bias, size_t blk_k,
                                      ie_act_i8_params params, int symmetric) {
  (void)symmetric;
  const size_t BK = (blk_k > 0 ? blk_k : cols);
  const float s = params.scale;
  const int z = (int)params.zero_point;

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    float acc = 0.0f;

    size_t kofs = 0;
    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + kofs;
      const int8_t *xblk = x_q + k0;

      for (size_t k = 0; k < klen; ++k) {
        const float xv = s * ((int)xblk[k] - z);
        acc += wblk[k] * xv;
      }
      kofs += klen;
    }

    if (bias) acc += bias[r];
    y[r] = acc;
  }
}

/**
 * @brief GEMV with per-group INT8 activations (fused dequantization), C path.
 *
 * @details
 * Each element uses group parameters:
 *   g = index / group_size
 *   real = scales[g] * (q - zeros[g])
 *
 * @param W           Weights (float, row-major).
 * @param x_q         INT8 activations.
 * @param y           Output.
 * @param rows        Rows.
 * @param cols        Cols.
 * @param bias        Optional bias or NULL.
 * @param blk_k       Column block size; 0 disables blocking.
 * @param group_size  Group size (>= 1).
 * @param scales      Group scales.
 * @param zeros       Group zero points.
 * @param symmetric   Informational flag (unused).
 */
static void gemv_qi8pg_f32_fused_c_impl(const float *W, const int8_t *x_q,
                                        float *y, size_t rows, size_t cols,
                                        const float *bias, size_t blk_k,
                                        size_t group_size, const float *scales,
                                        const int8_t *zeros, int symmetric) {
  (void)symmetric;
  const size_t BK = (blk_k > 0 ? blk_k : cols);

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    float acc = 0.0f;

    size_t kofs = 0;
    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + kofs;
      const int8_t *xblk = x_q + k0;

      for (size_t k = 0; k < klen; ++k) {
        const size_t g = (k0 + k) / group_size;
        const float s = scales[g];
        const int z = (int)zeros[g];
        const float xv = s * ((int)xblk[k] - z);
        acc += wblk[k] * xv;
      }
      kofs += klen;
    }

    if (bias) acc += bias[r];
    y[r] = acc;
  }
}

/**
 * @brief Decode FP8 E4M3 byte to float.
 *
 * @details
 * Software decode with flush-to-zero for exp==0. Uses exponent bias 7.
 *
 * @param v FP8 byte.
 * @return Decoded float value.
 */
static inline float ie_decode_fp8_e4m3(uint8_t v) {
  if (v == 0u) return 0.0f;
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 3) & 0xF;
  const uint8_t man = (v & 0x7);

  if (exp == 0) return sign ? -0.0f : 0.0f;

  const int bias = 7;
  const int e = (int)exp - bias;
  const float frac = (float)man / 8.0f;
  const float val = ldexpf(1.0f + frac, e);
  return sign ? -val : val;
}

/**
 * @brief Decode FP8 E5M2 byte to float.
 *
 * @details
 * - exp==0: returns signed zero (flush-to-zero policy).
 * - exp==all ones: returns Inf for man==0, quiet NaN otherwise.
 * - exponent bias is 15.
 *
 * @param v FP8 byte.
 * @return Decoded float value.
 */
static inline float ie_decode_fp8_e5m2(uint8_t v) {
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 2) & 0x1F;
  const uint8_t man = (v & 0x3);

  if (exp == 0) return sign ? -0.0f : 0.0f;

  if (exp == 0x1F) {
    if (man == 0) return sign ? -(__builtin_inff()) : (__builtin_inff());
    volatile float qnan = __builtin_nanf("0x1");
    return qnan;
  }

  const int bias = 15;
  const int e = (int)exp - bias;
  const float frac = (float)man / 4.0f;
  const float val = ldexpf(1.0f + frac, e);
  return sign ? -val : val;
}

/* ========================================================================== */
/*                                Public API                                   */
/* ========================================================================== */

/**
 * @brief Install the best available fp32 GEMV kernel.
 *
 * @details
 * On x86 (GCC/Clang), installs the AVX2/FMA implementation when allowed and
 * supported by the CPU. Otherwise, installs the generic C fallback.
 *
 * @param use_avx2 Non-zero to allow selecting AVX2/FMA implementation.
 */
void ie_kernels_install(int use_avx2) {
  int use_avx2_runtime = 0;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
  if (use_avx2) {
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
      use_avx2_runtime = 1;
    }
  }
#endif
#endif

  if (use_avx2_runtime) {
    s_gemv_f32 = ie_gemv_f32_avx2_impl;
  } else {
    s_gemv_f32 = gemv_generic_impl;
  }
}

/**
 * @brief Dispatch fp32 GEMV to the installed implementation.
 *
 * @details
 * If kernels were not installed yet, calls ::ie_kernels_install with AVX2 enabled.
 *
 * @param W     Weights.
 * @param x     Input vector.
 * @param y     Output vector.
 * @param rows  Rows.
 * @param cols  Cols.
 * @param bias  Optional bias.
 * @param blk_k Column block size; 0 disables blocking.
 */
void ie_gemv_f32(const float *W, const float *x, float *y, size_t rows,
                 size_t cols, const float *bias, size_t blk_k) {
  if (!s_gemv_f32) ie_kernels_install(/*use_avx2=*/1);
  s_gemv_f32(W, x, y, rows, cols, bias, blk_k);
}

/**
 * @brief GEMV with per-tensor INT8 activations (auto-dispatch).
 *
 * @details
 * Uses AVX2/FMA implementation when available; otherwise uses the C fused path.
 *
 * @param W         Weights.
 * @param x_q       INT8 activations.
 * @param y         Output.
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias.
 * @param blk_k     Column block size; 0 disables blocking.
 * @param params    INT8 params.
 * @param symmetric Informational flag.
 */
void ie_gemv_qi8_f32(const float *W, const int8_t *x_q, float *y, size_t rows,
                     size_t cols, const float *bias, size_t blk_k,
                     ie_act_i8_params params, int symmetric) {
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    ie_gemv_qi8_f32_avx2_impl(W, x_q, y, rows, cols, bias, blk_k, params,
                              symmetric);
    return;
  }
#endif
  gemv_qi8_f32_fused_c_impl(W, x_q, y, rows, cols, bias, blk_k, params,
                            symmetric);
}

/**
 * @brief GEMV with per-group INT8 activations (portable C implementation).
 *
 * @param W           Weights.
 * @param x_q         INT8 activations.
 * @param y           Output.
 * @param rows        Rows.
 * @param cols        Cols.
 * @param bias        Optional bias.
 * @param blk_k       Column block size; 0 disables blocking.
 * @param group_size  Group size.
 * @param scales      Group scales.
 * @param zeros       Group zero points.
 * @param symmetric   Informational flag.
 */
void ie_gemv_qi8pg_f32(const float *W, const int8_t *x_q, float *y,
                       size_t rows, size_t cols, const float *bias,
                       size_t blk_k, size_t group_size, const float *scales,
                       const int8_t *zeros, int symmetric) {
  gemv_qi8pg_f32_fused_c_impl(W, x_q, y, rows, cols, bias, blk_k, group_size,
                              scales, zeros, symmetric);
}

/**
 * @brief GEMV with FP8 activations (portable software decode).
 *
 * @param W      Weights.
 * @param x_fp8  FP8 activations (bytes).
 * @param y      Output.
 * @param rows   Rows.
 * @param cols   Cols.
 * @param bias   Optional bias.
 * @param blk_k  Column block size; 0 disables blocking.
 * @param fmt    FP8 format.
 */
void ie_gemv_qfp8_f32(const float *W, const uint8_t *x_fp8, float *y,
                      size_t rows, size_t cols, const float *bias,
                      size_t blk_k, ie_fp8_format fmt) {
  const size_t BK = (blk_k > 0 ? blk_k : cols);
  const int use_e4m3 = (fmt == IE_FP8_E4M3);

  for (size_t r = 0; r < rows; ++r) {
    const float *wrow = W + r * cols;
    float acc = 0.0f;

    size_t kofs = 0;
    for (size_t k0 = 0; k0 < cols; k0 += BK) {
      const size_t klen = (k0 + BK <= cols) ? BK : (cols - k0);
      const float *wblk = wrow + kofs;
      const uint8_t *xblk = x_fp8 + k0;

      for (size_t k = 0; k < klen; ++k) {
        const uint8_t b = xblk[k];
        const float xv = use_e4m3 ? ie_decode_fp8_e4m3(b) : ie_decode_fp8_e5m2(b);
        acc += wblk[k] * xv;
      }
      kofs += klen;
    }

    if (bias) acc += bias[r];
    y[r] = acc;
  }
}
