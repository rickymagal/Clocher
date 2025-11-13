/* File: engine/src/kernels/gemv_generic.c
 * -----------------------------------------------------------------------------
 * @file gemv_generic.c
 * @brief Portable GEMV kernels and the public GEMV API/dispatcher.
 *
 * This translation unit owns the public entry points:
 *  - ::ie_kernels_install
 *  - ::ie_gemv_f32
 *  - ::ie_gemv_qi8_f32
 *  - ::ie_gemv_qi8pg_f32
 *  - ::ie_gemv_qfp8_f32
 *
 * It may call architecture-specialized implementations provided by other
 * translation units (e.g., AVX2) when the platform supports them.
 */

#include "ie_kernels.h"
#include "ie_quant_act.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Forward declarations of optional arch-specific implementations              */
/* (resolved at link time when the corresponding TU is compiled).             */
/* -------------------------------------------------------------------------- */

/**
 * @brief AVX2/FMA GEMV (float path), provided by gemv_avx2.c.
 * @note This symbol is optional; the dispatcher checks CPU features before use.
 */
void ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y,
                           size_t rows, size_t cols, const float *bias,
                           size_t blk_k);

/**
 * @brief AVX2/FMA GEMV with INT8 activations (per-tensor params).
 * @note Optional symbol. The dispatcher guards its use with a CPU feature test.
 */
void ie_gemv_qi8_f32_avx2_impl(const float *W, const int8_t *x_q, float *y,
                               size_t rows, size_t cols, const float *bias,
                               size_t blk_k, ie_act_i8_params params,
                               int symmetric);

/* -------------------------------------------------------------------------- */
/* Internal installed function pointer (float path)                            */
/* -------------------------------------------------------------------------- */

/** @brief Installed float GEMV kernel (defaults to generic C). */
static void (*s_gemv_f32)(const float *, const float *, float *, size_t, size_t,
                          const float *, size_t) = NULL;

/* ========================================================================== */
/*                               Generic C kernels                             */
/* ========================================================================== */

/**
 * @brief Generic C GEMV: y = W * x (+bias), optional blocked-K layout.
 *
 * The routine multiplies a row-major (or "blocked-K contiguous") matrix W
 * by a dense vector x, accumulating into y. If @p blk_k is non-zero, the
 * inner loop processes chunks of size @p blk_k to better match cache lines.
 *
 * @param W     Pointer to weights (row-major; each row has @p cols floats).
 * @param x     Input vector (length @p cols).
 * @param y     Output vector (length @p rows).
 * @param rows  Number of rows in W and y.
 * @param cols  Number of columns in W and length of x.
 * @param bias  Optional bias to add per-output row (may be NULL).
 * @param blk_k Column block size; set 0 to disable blocking.
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
 * Dequantization model: real = scale * (q - zero_point).
 * This function multiplies float weights by dequantized INT8 activations
 * without materializing a separate float vector.
 *
 * @param W         Weights (row-major).
 * @param x_q       INT8 activations (length @p cols).
 * @param y         Output (length @p rows).
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias (may be NULL).
 * @param blk_k     Column block size; 0 means plain row-major.
 * @param params    Per-tensor INT8 parameters (scale, zero_point).
 * @param symmetric Non-zero if symmetric (zero_point=0). Informational only.
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
 * Each contiguous group of @p group_size elements in @p x_q uses its own
 * (scale, zero). The last group may be shorter.
 *
 * @param W           Weights (row-major).
 * @param x_q         INT8 activations.
 * @param y           Output.
 * @param rows        Rows.
 * @param cols        Cols.
 * @param bias        Optional bias (may be NULL).
 * @param blk_k       Column block size.
 * @param group_size  Activation quantization group size.
 * @param scales      Scales array, length ceil(cols/group_size).
 * @param zeros       Zero-points array, length ceil(cols/group_size).
 * @param symmetric   Non-zero if symmetric (informational only).
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
 * @brief Decode FP8 E4M3 value to float (flush subnormals).
 * @param v 8-bit FP8 E4M3 value.
 * @return 32-bit float approximation.
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
  const float val = (1.0f + frac) * (float)(1u << e);
  return sign ? -val : val;
}

/**
 * @brief Decode FP8 E5M2 value to float (IEEE-like; NaNs/Inf handled).
 * @param v 8-bit FP8 E5M2 value.
 * @return 32-bit float approximation (quiet NaN for FP8 NaN payloads).
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
  const float val = (1.0f + frac) * (float)(1u << e);
  return sign ? -val : val;
}

/* ========================================================================== */
/*                                Public API                                   */
/* ========================================================================== */

/**
 * @brief Install the best available float GEMV kernel for this process.
 *
 * On x86 with GCC/Clang, the dispatcher uses __builtin_cpu_supports("avx2")
 * and prefers the AVX2 implementation when available; otherwise it installs
 * the generic C fallback.
 *
 * @param use_avx2 Non-zero to allow AVX2 selection when supported.
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
 * @brief Dispatch GEMV (float path) to the installed kernel.
 *
 * @param W     Weights pointer.
 * @param x     Input vector.
 * @param y     Output vector.
 * @param rows  Rows in W/y.
 * @param cols  Cols in W/x.
 * @param bias  Optional bias (may be NULL).
 * @param blk_k Column-block size; 0 for plain row-major.
 */
void ie_gemv_f32(const float *W, const float *x, float *y, size_t rows,
                 size_t cols, const float *bias, size_t blk_k) {
  if (!s_gemv_f32) ie_kernels_install(/*use_avx2=*/1);
  s_gemv_f32(W, x, y, rows, cols, bias, blk_k);
}

/**
 * @brief GEMV with per-tensor INT8 activations, auto-dispatch (AVX2 or C).
 *
 * @param W         Weights (row-major or blocked-K).
 * @param x_q       INT8 activations (length cols).
 * @param y         Output (length rows).
 * @param rows      Rows.
 * @param cols      Cols.
 * @param bias      Optional bias (may be NULL).
 * @param blk_k     Column block size; 0 means plain row-major.
 * @param params    Per-tensor INT8 parameters (scale, zero_point).
 * @param symmetric Non-zero if symmetric (zero_point=0). Informational.
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
 * @brief GEMV with per-group INT8 activations (fused), C fallback.
 *
 * @param W           Weights.
 * @param x_q         INT8 activations.
 * @param y           Output.
 * @param rows        Rows.
 * @param cols        Cols.
 * @param bias        Optional bias (may be NULL).
 * @param blk_k       Column block size.
 * @param group_size  Activation quantization group size.
 * @param scales      Array of scales (ceil(cols/group_size)).
 * @param zeros       Array of zero points (ceil(cols/group_size)).
 * @param symmetric   Non-zero if symmetric (informational).
 */
void ie_gemv_qi8pg_f32(const float *W, const int8_t *x_q, float *y,
                       size_t rows, size_t cols, const float *bias,
                       size_t blk_k, size_t group_size, const float *scales,
                       const int8_t *zeros, int symmetric) {
  gemv_qi8pg_f32_fused_c_impl(W, x_q, y, rows, cols, bias, blk_k, group_size,
                              scales, zeros, symmetric);
}

/**
 * @brief GEMV with FP8 activations (E4M3/E5M2), fused decode (C fallback).
 *
 * @param W       Weights (row-major or blocked-K).
 * @param x_fp8   FP8 activations (bytes).
 * @param y       Output.
 * @param rows    Rows.
 * @param cols    Cols.
 * @param bias    Optional bias (may be NULL).
 * @param blk_k   Column block size; 0 means plain row-major.
 * @param fmt     FP8 format selector (::IE_FP8_E4M3 or ::IE_FP8_E5M2).
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
        const float xv =
            use_e4m3 ? ie_decode_fp8_e4m3(b) : ie_decode_fp8_e5m2(b);
        acc += wblk[k] * xv;
      }
      kofs += klen;
    }
    if (bias) acc += bias[r];
    y[r] = acc;
  }
}
