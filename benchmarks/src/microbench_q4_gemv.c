/**
 * @file microbench_q4_gemv.c
 * @brief Standalone microbenchmark for Q4_0 GEMV (INT4 weights, FP32 activations).
 *
 * Usage:
 *   ./build/microbench_q4_gemv [rows] [cols] [iters] [threads] [scale_bytes]
 * Defaults:
 *   rows=2048 cols=2880 iters=200 threads=$(nproc) scale_bytes=1
 */

#define _POSIX_C_SOURCE 200112L

#include "ie_kernels.h"
#include "ie_threadpool.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void *aligned_alloc_bytes(size_t align, size_t nbytes) {
  void *p = NULL;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (posix_memalign(&p, align, nbytes) != 0) return NULL;
  return p;
}

static uint32_t rng_next(uint32_t *s) {
  *s = (*s * 1664525u) + 1013904223u;
  return *s;
}

static void fill_rand_bytes(uint8_t *p, size_t n, uint32_t *seed) {
  for (size_t i = 0; i < n; ++i) {
    p[i] = (uint8_t)(rng_next(seed) & 0xFFu);
  }
}

static uint16_t f32_to_bf16(float v) {
  union { float f; uint32_t u; } x;
  x.f = v;
  return (uint16_t)(x.u >> 16);
}

static void fill_scales_u8(uint8_t *p, size_t n, uint32_t *seed) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t r = rng_next(seed) & 0xFFu;
    /* Center near 128 to keep scale ~1.0 */
    p[i] = (uint8_t)(96u + (r % 64u));
  }
}

static void fill_scales_bf16(uint16_t *p, size_t n, uint32_t *seed) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t r = rng_next(seed);
    float f = 0.25f + (float)(r & 0xFFFFu) / 65535.0f; /* ~0.25..1.25 */
    p[i] = f32_to_bf16(f);
  }
}

static void fill_x(float *x, size_t n, uint32_t *seed) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t r = rng_next(seed);
    float f = (float)(r & 0xFFFFu) / 65535.0f;
    x[i] = (f * 2.0f - 1.0f) * 0.5f;
  }
}

int main(int argc, char **argv) {
  size_t rows = (argc > 1) ? (size_t)strtoul(argv[1], NULL, 10) : 2048;
  size_t cols = (argc > 2) ? (size_t)strtoul(argv[2], NULL, 10) : 2880;
  size_t iters = (argc > 3) ? (size_t)strtoul(argv[3], NULL, 10) : 200;
  unsigned threads = (argc > 4) ? (unsigned)strtoul(argv[4], NULL, 10) : 0;
  size_t scale_bytes = (argc > 5) ? (size_t)strtoul(argv[5], NULL, 10) : 1;

  if (cols == 0 || (cols % 32) != 0) {
    fprintf(stderr, "cols must be >0 and divisible by 32\n");
    return 2;
  }
  if (!(scale_bytes == 1 || scale_bytes == 2)) {
    fprintf(stderr, "scale_bytes must be 1 or 2\n");
    return 2;
  }

  const size_t n_blocks = cols / 32;
  const size_t blocks_bytes = rows * n_blocks * 16;
  const size_t scales_bytes = rows * n_blocks * scale_bytes;

  uint8_t *W_blocks = (uint8_t *)aligned_alloc_bytes(64, blocks_bytes);
  uint8_t *W_scales = (uint8_t *)aligned_alloc_bytes(64, scales_bytes);
  float *x = (float *)aligned_alloc_bytes(64, cols * sizeof(float));
  float *y = (float *)aligned_alloc_bytes(64, rows * sizeof(float));
  if (!W_blocks || !W_scales || !x || !y) {
    fprintf(stderr, "alloc failed\n");
    free(W_blocks);
    free(W_scales);
    free(x);
    free(y);
    return 1;
  }

  uint32_t seed = 12345u;
  fill_rand_bytes(W_blocks, blocks_bytes, &seed);
  if (scale_bytes == 1) {
    fill_scales_u8(W_scales, scales_bytes, &seed);
  } else {
    fill_scales_bf16((uint16_t *)W_scales, scales_bytes / 2, &seed);
  }
  fill_x(x, cols, &seed);
  memset(y, 0, rows * sizeof(float));

  const char *aff = getenv("TP_AFFINITY");
  if (!aff || !aff[0]) aff = "compact";
  if (threads == 0) {
    const char *s = getenv("IE_THREADS");
    threads = (s && s[0]) ? (unsigned)strtoul(s, NULL, 10) : 0;
  }
  if (threads == 0) threads = 1;

  ie_threadpool_t *tp = ie_tp_create(threads, aff);
  if (!tp) {
    fprintf(stderr, "threadpool create failed\n");
    return 3;
  }
  ie_kernels_set_threadpool(tp, threads);

  /* Warmup */
  (void)ie_gemv_q4_0_f32(W_blocks, W_scales, scale_bytes, x, y, rows, cols, NULL);

  double t0 = now_s();
  for (size_t i = 0; i < iters; ++i) {
    (void)ie_gemv_q4_0_f32(W_blocks, W_scales, scale_bytes, x, y, rows, cols, NULL);
  }
  double t1 = now_s();

  const double dt = t1 - t0;
  const double us_per = (dt * 1e6) / (double)iters;
  const double rows_per_s = (double)rows * (double)iters / dt;

  printf("q4_gemv rows=%zu cols=%zu iters=%zu threads=%u scale_bytes=%zu\n",
         rows, cols, iters, threads, scale_bytes);
  printf("time: %.3f ms total, %.2f us/iter\n", dt * 1000.0, us_per);
  printf("throughput: %.2f rows/s\n", rows_per_s);

  ie_tp_destroy(tp);
  free(W_blocks);
  free(W_scales);
  free(x);
  free(y);
  return 0;
}
