/**
 * @file microbench_act_path.c
 * @brief Microbenchmark for activation quantization paths (INT8/FP8).
 *
 * This benchmark measures throughput of array-wide quantize/dequantize
 * kernels for both INT8 (per-tensor) and FP8 formats. It does not
 * include GEMV/MATMUL compute; it isolates conversion cost and memory
 * bandwidth behavior for large vectors.
 *
 * Build hint (example):
 *   cc -O3 -march=native -Iengine/include \
 *      benchmarks/src/microbench_act_path.c \
 *      engine/src/quant/act_int8.c engine/src/quant/act_fp8.c \
 *      -o microbench_act_path -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ie_quant_act.h"

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_gaussian(float* x, size_t n, unsigned seed) {
  /* Box–Muller transform: two uniforms -> two normals */
  srand(seed);
  for (size_t i = 0; i + 1 < n; i += 2) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    float r  = sqrtf(-2.0f * logf(1.0f - u1));
    float th = 2.0f * (float)M_PI * u2;
    x[i]   = r * cosf(th);
    x[i+1] = r * sinf(th);
  }
  if (n & 1) x[n-1] = 0.0f;
}

static double bench_int8(size_t n, int iters, int symmetric) {
  float*  src = (float*)aligned_alloc(64, n * sizeof(float));
  int8_t* q   = (int8_t*)aligned_alloc(64, n * sizeof(int8_t));
  float*  dst = (float*)aligned_alloc(64, n * sizeof(float));

  fill_gaussian(src, n, 1337u);
  ie_act_i8_params p;
  float mn = src[0], mx = src[0];
  for (size_t i = 1; i < n; ++i) {
    if (src[i] < mn) mn = src[i];
    if (src[i] > mx) mx = src[i];
  }
  ie_act_i8_params_from_minmax(mn, mx, symmetric, &p.scale, &p.zero_point);

  double t0 = now_seconds();
  for (int it = 0; it < iters; ++it) {
    ie_quantize_act_int8(src, q, n, p, symmetric);
    ie_dequantize_act_int8(q, dst, n, p);
  }
  double t1 = now_seconds();

  /* tiny checksum to avoid DCE */
  volatile float sink = 0.0f;
  for (size_t i = 0; i < n; ++i) sink += dst[i];
  fprintf(stderr, "INT8 checksum: %.6f\n", sink);

  free(src); free(q); free(dst);
  return (t1 - t0);
}

static double bench_fp8(size_t n, int iters, ie_fp8_format fmt) {
  float*   src = (float*)aligned_alloc(64, n * sizeof(float));
  uint8_t* q   = (uint8_t*)aligned_alloc(64, n * sizeof(uint8_t));
  float*   dst = (float*)aligned_alloc(64, n * sizeof(float));

  fill_gaussian(src, n, 4242u);

  double t0 = now_seconds();
  for (int it = 0; it < iters; ++it) {
    ie_quantize_act_fp8(src, q, n, fmt);
    ie_dequantize_act_fp8(q, dst, n, fmt);
  }
  double t1 = now_seconds();

  volatile float sink = 0.0f;
  for (size_t i = 0; i < n; ++i) sink += dst[i];
  fprintf(stderr, "FP8 checksum: %.6f\n", sink);

  free(src); free(q); free(dst);
  return (t1 - t0);
}

int main(int argc, char** argv) {
  size_t n = 64 * 1024 * 1024 / sizeof(float); /* ~16M floats (~64 MiB) */
  int iters = 5;

  if (argc > 1) n = (size_t)strtoull(argv[1], NULL, 10);
  if (argc > 2) iters = atoi(argv[2]);

  printf("Vector length: %zu, iters: %d\n", n, iters);

  double t_i8_sym = bench_int8(n, iters, /*symmetric=*/1);
  double t_i8_as  = bench_int8(n, iters, /*symmetric=*/0);
  double t_e4m3   = bench_fp8(n, iters, IE_FP8_E4M3);
  double t_e5m2   = bench_fp8(n, iters, IE_FP8_E5M2);

  double bytes = (double)n * (sizeof(float) + sizeof(int8_t) + sizeof(float));
  double tot   = (double)iters * bytes;

  printf("INT8 symmetric:  %.3f s, aggregate BW ~ %.2f GiB/s\n",
         t_i8_sym, (tot / t_i8_sym) / (1024.0 * 1024.0 * 1024.0));
  printf("INT8 asymmetric: %.3f s, aggregate BW ~ %.2f GiB/s\n",
         t_i8_as,  (tot / t_i8_as)  / (1024.0 * 1024.0 * 1024.0));

  bytes = (double)n * (sizeof(float) + sizeof(uint8_t) + sizeof(float));
  tot   = (double)iters * bytes;

  printf("FP8 E4M3:        %.3f s, aggregate BW ~ %.2f GiB/s\n",
         t_e4m3, (tot / t_e4m3) / (1024.0 * 1024.0 * 1024.0));
  printf("FP8 E5M2:        %.3f s, aggregate BW ~ %.2f GiB/s\n",
         t_e5m2, (tot / t_e5m2) / (1024.0 * 1024.0 * 1024.0));

  return 0;
}
