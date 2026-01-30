/* File: engine/src/kernels/gemv_q4_generic.c
 * -----------------------------------------------------------------------------
 * @file gemv_q4_generic.c
 * @brief Portable Q4_0 GEMV (matrix-vector) kernel with runtime dispatch and optional row-parallelism.
 *
 * @details
 * This module implements a Q4_0 matrix-vector multiply used by the GPT-OSS INT4 path.
 *
 * The implementation provides:
 *  - A portable scalar fallback (always available).
 *  - An AVX2+FMA optimized implementation (compiled in a separate TU) selected
 *    at runtime when supported by the current CPU.
 *
 * Optional parallelism:
 *  - If a thread pool is available via `ie_kernels_get_threadpool()`, this GEMV
 *    may split work by contiguous row ranges and execute in parallel.
 *
 * The Q4_0 layout matches the engine's weight packing:
 *  - Each block covers 32 columns.
 *  - Blocks store 16 bytes of packed nibbles per 32 weights.
 *  - Each block has one scale (BF16, 2 bytes, or log2(u8,q3), 1 byte).
 *
 * The matrix is represented as two parallel streams:
 *  - W_blocks: rows * (cols/32) * 16 bytes
 *  - W_scales: rows * (cols/32) * scale_bytes
 *
 * Signed 4-bit decode (two's complement):
 *  - n = (nibble & 0xF)
 *  - w = (n >= 8) ? (n - 16) : n  -> range [-8, 7]
 */

#include "ie_kernels.h"
#include "ie_threadpool.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <pthread.h>

#include "ie_cpu.h"

/* Forward declaration for the AVX2 implementation (defined in gemv_q4_avx2.c). */
int ie_gemv_q4_0_f32_avx2_impl(const uint8_t *W_blocks,
                              const uint8_t *W_scales,
                              size_t scale_bytes,
                              const float *x,
                              float *y,
                              size_t rows,
                              size_t cols,
                              const uint16_t *bias_bf16);

/* -------------------------------------------------------------------------- */
/* BF16 and scale decoding helpers                                            */
/* -------------------------------------------------------------------------- */

static inline float ie_bf16_to_f32_scalar(uint16_t b) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = ((uint32_t)b) << 16;
  return v.f;
}

static float ie_log2_u8_q3_lut_[256];
static int ie_log2_u8_q3_ready_ = 0;

static void ie_log2_u8_q3_init_impl_(void) {
  for (int i = 0; i < 256; ++i) {
    const float exp = ((float)(i - 128)) * 0.125f;
    ie_log2_u8_q3_lut_[i] = exp2f(exp);
  }
  ie_log2_u8_q3_ready_ = 1;
}

void ie_q4_log2_u8_q3_init_generic(void) {
  if (!ie_log2_u8_q3_ready_) {
    ie_log2_u8_q3_init_impl_();
  }
}

static inline float ie_log2_u8_q3_to_f32(uint8_t v) {
  if (!ie_log2_u8_q3_ready_) {
    ie_log2_u8_q3_init_impl_();
  }
  return ie_log2_u8_q3_lut_[v];
}

static inline float ie_fp8_e4m3_to_f32(uint8_t v) {
  if (v == 0u) return 0.0f;
  const uint8_t sign = (v >> 7) & 0x1;
  const uint8_t exp = (v >> 3) & 0xF;
  const uint8_t man = (v & 0x7);

  if (exp == 0) {
    return sign ? -0.0f : 0.0f;
  }
  const int bias = 7;
  const int e = ((int)exp) - bias;
  const float frac = (float)man / 8.0f;
  const float val = (1.0f + frac) * ldexpf(1.0f, e);
  return sign ? -val : val;
}

static inline float ie_q4_load_scale_f32_ex(const uint8_t *scales,
                                            size_t scale_bytes,
                                            int scale_fmt) {
  if (scale_bytes == 2) {
    uint16_t b;
    memcpy(&b, scales, sizeof(uint16_t));
    return ie_bf16_to_f32_scalar(b);
  }
  if (scale_bytes == 1) {
    return (scale_fmt == 1) ? ie_fp8_e4m3_to_f32(scales[0])
                            : ie_log2_u8_q3_to_f32(scales[0]);
  }
  return 0.0f;
}

static int ie_q4_debug_nan_enabled_(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *s = getenv("IE_Q4_DEBUG_NAN");
    cached = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
  }
  return cached;
}

/* -------------------------------------------------------------------------- */
/* Scalar dot and generic GEMV                                                */
/* -------------------------------------------------------------------------- */

static float ie_q4_0_row_dot_f32(const uint8_t *blocks,
                                const uint8_t *scales,
                                size_t scale_bytes,
                                int scale_fmt,
                                const float *x,
                                size_t cols) {
  const size_t n_blocks = cols / 32;
  float acc = 0.0f;
  const int dbg_nan = ie_q4_debug_nan_enabled_();

  for (size_t b = 0; b < n_blocks; ++b) {
    const uint8_t *q = blocks + b * 16;
    const float d = ie_q4_load_scale_f32_ex(scales + b * scale_bytes, scale_bytes, scale_fmt);

    for (size_t j = 0; j < 32; ++j) {
      const uint8_t byte = q[j >> 1];
      const uint8_t nibble = (j & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
      const int32_t w = (nibble >= 8u) ? ((int32_t)nibble - 16) : (int32_t)nibble;
      acc += (d * (float)w) * x[b * 32 + j];
      if (dbg_nan && !isfinite(acc)) {
        fprintf(stderr,
                "[q4_nan] block=%zu j=%zu scale=%g w=%d x=%g acc=%g\n",
                b, j, (double)d, (int)w, (double)x[b * 32 + j], (double)acc);
        return acc;
      }
    }
  }

  return acc;
}

static int ie_gemv_q4_0_f32_generic_impl_ex(const uint8_t *W_blocks,
                                           const uint8_t *W_scales,
                                           size_t scale_bytes,
                                           int scale_fmt,
                                           const float *x,
                                           float *y,
                                           size_t rows,
                                           size_t cols,
                                           const uint16_t *bias_bf16) {
  if (!W_blocks || !W_scales || !x || !y) {
    return 1;
  }
  if (cols == 0 || (cols % 32) != 0) {
    return 2;
  }
  if (!(scale_bytes == 1 || scale_bytes == 2)) {
    return 3;
  }

  const size_t n_blocks = cols / 32;
  const size_t row_block_bytes = n_blocks * 16;
  const size_t row_scale_bytes = n_blocks * scale_bytes;

  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *row_blocks = W_blocks + r * row_block_bytes;
    const uint8_t *row_scales = W_scales + r * row_scale_bytes;

    float v = ie_q4_0_row_dot_f32(row_blocks, row_scales, scale_bytes, scale_fmt, x, cols);
    if (bias_bf16) {
      v += ie_bf16_to_f32_scalar(bias_bf16[r]);
    }
    y[r] = v;
  }

  return 0;
}

static int ie_gemv_q4_0_f32_generic_impl(const uint8_t *W_blocks,
                                        const uint8_t *W_scales,
                                        size_t scale_bytes,
                                        const float *x,
                                        float *y,
                                        size_t rows,
                                        size_t cols,
                                        const uint16_t *bias_bf16) {
  return ie_gemv_q4_0_f32_generic_impl_ex(W_blocks, W_scales, scale_bytes, 0, x, y, rows, cols, bias_bf16);
}

/* -------------------------------------------------------------------------- */
/* Dispatch (thread-safe one-time init)                                       */
/* -------------------------------------------------------------------------- */

typedef int (*ie_q4_gemv_fn_t)(const uint8_t *, const uint8_t *, size_t,
                              const float *, float *, size_t, size_t, const uint16_t *);

static ie_q4_gemv_fn_t g_q4_fn_ = NULL;
static pthread_once_t g_q4_once_ = PTHREAD_ONCE_INIT;

static void ie_q4_dispatch_init_(void) {
  ie_cpu_features_t feat;
  ie_cpu_features_detect(&feat);

  const char *force_generic = getenv("IE_Q4_FORCE_GENERIC");
  if (force_generic && force_generic[0] && strcmp(force_generic, "0") != 0) {
    g_q4_fn_ = ie_gemv_q4_0_f32_generic_impl;
    return;
  }

  if (feat.avx2 && feat.fma) {
    g_q4_fn_ = ie_gemv_q4_0_f32_avx2_impl;
  } else {
    g_q4_fn_ = ie_gemv_q4_0_f32_generic_impl;
  }
}

/* -------------------------------------------------------------------------- */
/* Optional row-parallel wrapper                                              */
/* -------------------------------------------------------------------------- */

typedef struct {
  ie_q4_gemv_fn_t fn;
  const uint8_t *W_blocks;
  const uint8_t *W_scales;
  size_t scale_bytes;
  const float *x;
  float *y;
  size_t cols;
  const uint16_t *bias_bf16;
  size_t row_block_bytes;
  size_t row_scale_bytes;
} ie_q4_job_t;

static void ie_q4_job_run(void *ctx, unsigned start, unsigned end) {
  ie_q4_job_t *job = (ie_q4_job_t *)ctx;
  if (!job || start >= end) return;

  const size_t s = (size_t)start;
  const size_t n = (size_t)(end - start);

  const uint8_t *wb = job->W_blocks + s * job->row_block_bytes;
  const uint8_t *ws = job->W_scales + s * job->row_scale_bytes;
  float *y0 = job->y + s;
  const uint16_t *b0 = job->bias_bf16 ? (job->bias_bf16 + s) : NULL;

  (void)job->fn(wb, ws, job->scale_bytes, job->x, y0, n, job->cols, b0);
}

static int ie_should_parallel_rows(size_t rows, unsigned nth) {
  if (nth <= 1u) return 0;
  if (rows < 256u) return 0;
  if (rows < (size_t)nth * 64u) return 0;
  return 1;
}

static unsigned ie_q4_env_grainsize(unsigned rows, unsigned nth) {
  const char *s = getenv("IE_Q4_GRAINSIZE");
  if (!s || !*s) return 0;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || v <= 0) return 0;
  if ((unsigned)v > rows) return rows ? rows : 1u;
  (void)nth;
  return (unsigned)v;
}

static unsigned ie_q4_env_denom(unsigned nth) {
  const char *s = getenv("IE_Q4_PARALLEL_DENOM");
  if (!s || !*s) return 0;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || v <= 0) return 0;
  if ((unsigned)v < nth) return nth;
  return (unsigned)v;
}

/* -------------------------------------------------------------------------- */
/* Public entry point                                                         */
/* -------------------------------------------------------------------------- */

int ie_gemv_q4_0_f32(const uint8_t *W_blocks,
                    const uint8_t *W_scales,
                    size_t scale_bytes,
                    const float *x,
                    float *y,
                    size_t rows,
                    size_t cols,
                    const uint16_t *bias_bf16) {
  return ie_gemv_q4_0_f32_ex(W_blocks, W_scales, scale_bytes, 0, x, y, rows, cols, bias_bf16);
}

int ie_gemv_q4_0_f32_ex(const uint8_t *W_blocks,
                        const uint8_t *W_scales,
                        size_t scale_bytes,
                        int scale_fmt,
                        const float *x,
                        float *y,
                        size_t rows,
                        size_t cols,
                        const uint16_t *bias_bf16) {
  if (!W_blocks || !W_scales || !x || !y) {
    return 1;
  }
  if (cols == 0 || (cols % 32) != 0) {
    return 2;
  }
  if (!(scale_bytes == 1 || scale_bytes == 2)) {
    return 3;
  }
  if (scale_fmt != 0 && scale_fmt != 1) {
    return 4;
  }

  if (scale_fmt == 1) {
    return ie_gemv_q4_0_f32_generic_impl_ex(W_blocks, W_scales, scale_bytes, 1, x, y, rows, cols, bias_bf16);
  }

  (void)pthread_once(&g_q4_once_, ie_q4_dispatch_init_);
  if (!g_q4_fn_) {
    g_q4_fn_ = ie_gemv_q4_0_f32_generic_impl;
  }

  ie_threadpool_t *tp = ie_kernels_get_threadpool();
  unsigned nth = ie_kernels_get_threadpool_nth();

  if (tp && rows <= 0xFFFFFFFFu && ie_should_parallel_rows(rows, nth)) {
    const size_t n_blocks = cols / 32;
    ie_q4_job_t job = {
      .fn = g_q4_fn_,
      .W_blocks = W_blocks,
      .W_scales = W_scales,
      .scale_bytes = scale_bytes,
      .x = x,
      .y = y,
      .cols = cols,
      .bias_bf16 = bias_bf16,
      .row_block_bytes = n_blocks * 16,
      .row_scale_bytes = n_blocks * scale_bytes,
    };

    unsigned rows_u = (unsigned)rows;
    unsigned denom = (nth * 4u);
    {
      unsigned env_d = ie_q4_env_denom(nth);
      if (env_d) denom = env_d;
    }
    if (denom < 1u) denom = 1u;
    unsigned grainsize = (unsigned)((rows + (size_t)denom - 1u) / (size_t)denom);
    if (grainsize < 16u) grainsize = 16u;
    if (rows_u >= 512u && grainsize > 32u) grainsize = 32u;
    {
      unsigned env_gs = ie_q4_env_grainsize(rows_u, nth);
      if (env_gs) grainsize = env_gs;
    }

    ie_tp_parallel_for(tp, rows_u, grainsize, ie_q4_job_run, &job);
    return 0;
  }

  return g_q4_fn_(W_blocks, W_scales, scale_bytes, x, y, rows, cols, bias_bf16);
}
