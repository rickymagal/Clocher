/* File: engine/src/kernels/gemv_generic.c
 * -----------------------------------------------------------------------------
 * @file gemv_generic.c
 * @brief CPU GEMV dispatch (generic/AVX2) plus optional internal row-parallelism.
 *
 * @details
 * This module provides:
 *  - A lazily-initialized dispatch table for GEMV and BF16 helpers.
 *  - Optional row-parallel wrappers driven by an external thread pool.
 *
 * Threading model:
 *  - If a pool is provided via `ie_kernels_set_threadpool()`, GEMV wrappers
 *    may split work by contiguous row ranges and execute via `ie_tp_parallel_for()`.
 *  - If no pool is provided, kernels may lazily create an internal pool based
 *    on environment variables (best-effort).
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ie_cpu.h"
#include "ie_kernels.h"
#include "ie_threadpool.h"
#include "util_logging.h"

/* -------------------------------------------------------------------------- */
/* Dispatch table types                                                       */
/* -------------------------------------------------------------------------- */

typedef void (*ie_gemv_f32_impl_fn)(const float *W, const float *x, float *y,
                                   size_t rows, size_t cols,
                                   const float *bias, size_t bias_stride);

typedef int (*ie_gemv_bf16_impl_fn)(const uint16_t *W_bf16, const float *x, float *y,
                                   size_t rows, size_t cols,
                                   const uint16_t *bias_bf16);

typedef void (*ie_vec_bf16_to_f32_impl_fn)(const uint16_t *in, float *out, size_t n);

typedef void (*ie_gemv_qi8_impl_fn)(const int8_t *W_qi8, const float *W_scales,
                                   const float *x, float *y,
                                   size_t rows, size_t cols,
                                   size_t block_cols,
                                   const float *bias, size_t bias_stride);

/* -------------------------------------------------------------------------- */
/* Global dispatch pointers                                                   */
/* -------------------------------------------------------------------------- */

static ie_gemv_f32_impl_fn g_gemv_f32 = NULL;
static ie_gemv_bf16_impl_fn g_gemv_bf16 = NULL;
static ie_vec_bf16_to_f32_impl_fn g_vec_bf16_to_f32 = NULL;
static ie_gemv_qi8_impl_fn g_gemv_qi8 = NULL;

static int g_logged_once = 0;

/* -------------------------------------------------------------------------- */
/* Kernel threading state                                                     */
/* -------------------------------------------------------------------------- */

/* User-supplied pool (owned by the engine). */
static ie_threadpool_t *g_user_tp = NULL;
static unsigned g_user_tp_nth = 1u;

/* Optional env-created pool (owned by this module; intentionally leaked). */
static ie_threadpool_t *g_env_tp = NULL;
static unsigned g_env_tp_nth = 1u;

/* -------------------------------------------------------------------------- */
/* Logging helpers                                                            */
/* -------------------------------------------------------------------------- */

static void ie_log_kernel_once(const char *which_f32,
                               const char *which_bf16,
                               const char *which_vec_bf16,
                               const char *which_qi8) {
  if (g_logged_once) return;
  g_logged_once = 1;

  const char *env = getenv("IE_LOG_KERNELS");
  if (!env || env[0] != '1') return;

  ie_log_info("[kernels] gemv_f32=%s gemv_bf16=%s vec_bf16_to_f32=%s gemv_qi8=%s\n",
              which_f32 ? which_f32 : "unknown",
              which_bf16 ? which_bf16 : "unknown",
              which_vec_bf16 ? which_vec_bf16 : "unknown",
              which_qi8 ? which_qi8 : "unknown");
}

/* -------------------------------------------------------------------------- */
/* Generic fallback implementations                                           */
/* -------------------------------------------------------------------------- */

static void ie_gemv_f32_generic_impl(const float *W, const float *x, float *y,
                                     size_t rows, size_t cols,
                                     const float *bias, size_t bias_stride) {
  for (size_t r = 0; r < rows; r++) {
    const float *row = W + r * cols;
    float sum = 0.0f;

    for (size_t c = 0; c < cols; c++) {
      sum += row[c] * x[c];
    }

    if (bias) {
      sum += bias[r * bias_stride];
    }

    y[r] = sum;
  }
}

static inline float ie_bf16_bits_to_f32(uint16_t v) {
  union {
    uint32_t u;
    float f;
  } x;
  x.u = ((uint32_t)v) << 16;
  return x.f;
}

static void ie_vec_bf16_to_f32_generic_impl(const uint16_t *in, float *out, size_t n) {
  if (!in || !out || n == 0u) return;
  for (size_t i = 0; i < n; ++i) {
    out[i] = ie_bf16_bits_to_f32(in[i]);
  }
}

static int ie_gemv_bf16_generic_impl(const uint16_t *W_bf16, const float *x, float *y,
                                     size_t rows, size_t cols,
                                     const uint16_t *bias_bf16) {
  if (!W_bf16 || !x || !y || rows == 0u || cols == 0u) return -1;

  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const size_t base = r * cols;
    for (size_t c = 0; c < cols; ++c) {
      const float w = ie_bf16_bits_to_f32(W_bf16[base + c]);
      acc += w * x[c];
    }
    if (bias_bf16) acc += ie_bf16_bits_to_f32(bias_bf16[r]);
    y[r] = acc;
  }

  return 0;
}

static void ie_gemv_qi8_stub_impl(const int8_t *W_qi8, const float *W_scales,
                                 const float *x, float *y,
                                 size_t rows, size_t cols,
                                 size_t block_cols,
                                 const float *bias, size_t bias_stride) {
  (void)W_qi8;
  (void)W_scales;
  (void)x;
  (void)y;
  (void)rows;
  (void)cols;
  (void)block_cols;
  (void)bias;
  (void)bias_stride;
  ie_log_error("ie_gemv_qi8: no implementation installed\n");
}

/* -------------------------------------------------------------------------- */
/* Thread-pool selection                                                      */
/* -------------------------------------------------------------------------- */

static unsigned ie_parse_threads_env(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0u;

  long v = strtol(s, NULL, 10);
  if (v < 1) v = 1;
  if (v > 256) v = 256;
  return (unsigned)v;
}

static void ie_env_tp_init(void) {
  if (g_env_tp) return;

  unsigned nth = 0u;

  /* Prefer a dedicated variable if present, otherwise fall back to IE_THREADS. */
  nth = ie_parse_threads_env("IE_KERNEL_THREADS");
  if (nth == 0u) nth = ie_parse_threads_env("IE_THREADS");
  if (nth == 0u) nth = 1u;

  g_env_tp_nth = nth;

  if (nth > 1u) {
    g_env_tp = ie_tp_create(nth, "auto");
    if (!g_env_tp) {
      g_env_tp_nth = 1u;
      g_env_tp = NULL;
    }
  }
}

static ie_threadpool_t *ie_active_tp(unsigned *out_nth) {
  if (g_user_tp) {
    if (out_nth) *out_nth = (g_user_tp_nth ? g_user_tp_nth : 1u);
    return g_user_tp;
  }

  ie_env_tp_init();
  if (out_nth) *out_nth = (g_env_tp_nth ? g_env_tp_nth : 1u);
  return g_env_tp;
}

static int ie_should_parallel_rows(size_t rows, unsigned nth) {
  if (nth <= 1u) return 0;
  if (rows < 256u) return 0;
  if (rows < (size_t)nth * 64u) return 0;
  return 1;
}

/* -------------------------------------------------------------------------- */
/* Public: threadpool control                                                 */
/* -------------------------------------------------------------------------- */

void ie_kernels_set_threadpool(ie_threadpool_t *tp, unsigned nth) {
  if (nth < 1u) nth = 1u;
  g_user_tp = tp;
  g_user_tp_nth = nth;
}

ie_threadpool_t* ie_kernels_get_threadpool(void) {
  unsigned nth = 1u;
  return ie_active_tp(&nth);
}

unsigned ie_kernels_get_threadpool_nth(void) {
  unsigned nth = 1u;
  (void)ie_active_tp(&nth);
  if (nth < 1u) nth = 1u;
  return nth;
}

/* -------------------------------------------------------------------------- */
/* Installation and lazy init                                                 */
/* -------------------------------------------------------------------------- */

void ie_kernels_install(int use_avx2) {
  g_gemv_f32 = &ie_gemv_f32_generic_impl;
  g_gemv_bf16 = &ie_gemv_bf16_generic_impl;
  g_vec_bf16_to_f32 = &ie_vec_bf16_to_f32_generic_impl;
  g_gemv_qi8 = &ie_gemv_qi8_stub_impl;

  const char *sel_f32 = "generic";
  const char *sel_bf16 = "generic";
  const char *sel_vec_bf16 = "generic";
  const char *sel_qi8 = "stub";

  if (use_avx2) {
    ie_cpu_features_t f;
    ie_cpu_features_detect(&f);

    if (f.avx2 && f.fma) {
      extern void ie_gemv_f32_avx2_impl(const float *W, const float *x, float *y,
                                       size_t rows, size_t cols,
                                       const float *bias, size_t bias_stride);

      extern int ie_gemv_bf16_f32_avx2_impl(const uint16_t *W_bf16, const float *x, float *y,
                                           size_t rows, size_t cols,
                                           const uint16_t *bias_bf16);

      extern void ie_vec_bf16_to_f32_avx2_impl(const uint16_t *in, float *out, size_t n);

      extern void ie_gemv_qi8_avx2_impl(const int8_t *W_qi8, const float *W_scales,
                                       const float *x, float *y,
                                       size_t rows, size_t cols,
                                       size_t block_cols,
                                       const float *bias, size_t bias_stride);

      g_gemv_f32 = &ie_gemv_f32_avx2_impl;
      g_gemv_bf16 = &ie_gemv_bf16_f32_avx2_impl;
      g_vec_bf16_to_f32 = &ie_vec_bf16_to_f32_avx2_impl;
      g_gemv_qi8 = &ie_gemv_qi8_avx2_impl;

      sel_f32 = "avx2";
      sel_bf16 = "avx2";
      sel_vec_bf16 = "avx2";
      sel_qi8 = "avx2";
    }
  }

  ie_log_kernel_once(sel_f32, sel_bf16, sel_vec_bf16, sel_qi8);
}

static void ie_kernels_lazy_init(void) {
  if (g_gemv_f32 && g_gemv_bf16 && g_vec_bf16_to_f32 && g_gemv_qi8) return;
  ie_kernels_install(1);
}

/* -------------------------------------------------------------------------- */
/* Parallel wrappers                                                          */
/* -------------------------------------------------------------------------- */

typedef struct {
  const float *W;
  const float *x;
  float *y;
  size_t cols;
  const float *bias;
  size_t bias_stride;
  ie_gemv_f32_impl_fn impl;
} ie_gemv_f32_job_t;

static void ie_gemv_f32_job_run(void *ctx, unsigned start, unsigned end) {
  ie_gemv_f32_job_t *job = (ie_gemv_f32_job_t *)ctx;
  if (!job || start >= end) return;

  const size_t s = (size_t)start;
  const size_t n = (size_t)(end - start);

  const float *W0 = job->W + s * job->cols;
  float *y0 = job->y + s;
  const float *b0 = job->bias ? (job->bias + s * job->bias_stride) : NULL;

  job->impl(W0, job->x, y0, n, job->cols, b0, job->bias_stride);
}

typedef struct {
  const uint16_t *W_bf16;
  const float *x;
  float *y;
  size_t cols;
  const uint16_t *bias_bf16;
  ie_gemv_bf16_impl_fn impl;
} ie_gemv_bf16_job_t;

static void ie_gemv_bf16_job_run(void *ctx, unsigned start, unsigned end) {
  ie_gemv_bf16_job_t *job = (ie_gemv_bf16_job_t *)ctx;
  if (!job || start >= end) return;

  const size_t s = (size_t)start;
  const size_t n = (size_t)(end - start);

  const uint16_t *w0 = job->W_bf16 + s * job->cols;
  float *y0 = job->y + s;
  const uint16_t *b0 = job->bias_bf16 ? (job->bias_bf16 + s) : NULL;

  (void)job->impl(w0, job->x, y0, n, job->cols, b0);
}

/* -------------------------------------------------------------------------- */
/* Public kernel entry points                                                 */
/* -------------------------------------------------------------------------- */

void ie_vec_bf16_to_f32(const uint16_t *in, float *out, size_t n) {
  ie_kernels_lazy_init();
  g_vec_bf16_to_f32(in, out, n);
}

int ie_gemv_bf16_f32(const uint16_t *W_bf16, const float *x, float *y,
                     size_t rows, size_t cols,
                     const uint16_t *bias_bf16) {
  ie_kernels_lazy_init();

  unsigned nth = 1u;
  ie_threadpool_t *tp = ie_active_tp(&nth);

  if (tp && rows <= 0xFFFFFFFFu && ie_should_parallel_rows(rows, nth)) {
    ie_gemv_bf16_job_t job = {
      .W_bf16 = W_bf16,
      .x = x,
      .y = y,
      .cols = cols,
      .bias_bf16 = bias_bf16,
      .impl = g_gemv_bf16,
    };

    unsigned rows_u = (unsigned)rows;
    unsigned denom = (nth * 4u);
    if (denom < 1u) denom = 1u;
    unsigned grainsize = (unsigned)((rows + (size_t)denom - 1u) / (size_t)denom);
    if (grainsize < 16u) grainsize = 16u;

    ie_tp_parallel_for(tp, rows_u, grainsize, ie_gemv_bf16_job_run, &job);
    return 0;
  }

  return g_gemv_bf16(W_bf16, x, y, rows, cols, bias_bf16);
}

void ie_gemv_f32(const float *W, const float *x, float *y,
                 size_t rows, size_t cols,
                 const float *bias, size_t bias_stride) {
  ie_kernels_lazy_init();

  unsigned nth = 1u;
  ie_threadpool_t *tp = ie_active_tp(&nth);

  if (tp && rows <= 0xFFFFFFFFu && ie_should_parallel_rows(rows, nth)) {
    ie_gemv_f32_job_t job = {
      .W = W,
      .x = x,
      .y = y,
      .cols = cols,
      .bias = bias,
      .bias_stride = bias_stride,
      .impl = g_gemv_f32,
    };

    unsigned rows_u = (unsigned)rows;
    unsigned denom = (nth * 4u);
    if (denom < 1u) denom = 1u;
    unsigned grainsize = (unsigned)((rows + (size_t)denom - 1u) / (size_t)denom);
    if (grainsize < 16u) grainsize = 16u;

    ie_tp_parallel_for(tp, rows_u, grainsize, ie_gemv_f32_job_run, &job);
    return;
  }

  g_gemv_f32(W, x, y, rows, cols, bias, bias_stride);
}

void ie_gemv_qi8(const int8_t *W_qi8, const float *W_scales,
                 const float *x, float *y,
                 size_t rows, size_t cols,
                 size_t block_cols,
                 const float *bias, size_t bias_stride) {
  ie_kernels_lazy_init();
  g_gemv_qi8(W_qi8, W_scales, x, y, rows, cols, block_cols, bias, bias_stride);
}
