/**
 * @file microbench_gemv.c
 * @brief Standalone microbenchmark for GEMV/tanh/embed hotspots (FP32).
 *
 * Build via Makefile target `microbench`. Runs timed loops to mimic
 * the decode inner loop and prints plain-text metrics (time and GB/s).
 *
 * Usage:
 *   ./build/microbench_gemv [H] [V] [iters]
 * Defaults:
 *   H=256 (hidden), V=1024 (logits), iters=100
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/**
 * @brief Portable timestamp in seconds using C11 timespec_get.
 * @return Current time as double seconds.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief Initialize an array with pseudo-random FP32 in [-scale, +scale].
 * @param p     Destination pointer.
 * @param n     Number of elements.
 * @param seed  Seed (modified internally).
 * @param scale Half-range for uniform distribution.
 */
static void init_rand(float *p, size_t n, unsigned *seed, float scale) {
  for (size_t i = 0; i < n; ++i) {
    *seed = *seed * 1664525u + 1013904223u;
    float u = (float)((*seed & 0xFFFFFFu) / 16777216.0f);
    p[i] = (u * 2.0f - 1.0f) * scale;
  }
}

/**
 * @brief Row-major GEMV: y = W * x
 * @param W    Weights [rows x cols], row-major.
 * @param x    Input vector [cols].
 * @param y    Output vector [rows].
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
static void gemv_rowmajor(const float *W, const float *x, float *y,
                          size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *w = W + r * cols;
    for (size_t c = 0; c < cols; ++c) acc += w[c] * x[c];
    y[r] = acc;
  }
}

/**
 * @brief Apply tanhf elementwise (in-place).
 * @param v Vector pointer.
 * @param n Number of elements.
 */
static void vec_tanh(float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = tanhf(v[i]);
}

/**
 * @brief Synthesize an embedding from a token id (deterministic).
 * @param tok Token id.
 * @param x   Output embedding [H].
 * @param H   Hidden size.
 */
static void embed_token(unsigned tok, float *x, size_t H) {
  unsigned s = 0x9E3779B9u ^ (tok * 0x85EBCA6Bu);
  for (size_t i = 0; i < H; ++i) {
    s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
    float t = (float)(s & 0xFFFFu) / 65536.0f;
    x[i] = (t * 2.0f - 1.0f) + 0.1f * sinf((float)(i + (tok % 31)) * 0.07f);
  }
}

/**
 * @brief Entry point: allocate matrices/vectors; time embed+gemv+tanh loops.
 * @param argc Arg count.
 * @param argv Arg vector: [H] [V] [iters].
 * @return 0 on success; non-zero on failure.
 */
int main(int argc, char **argv) {
  size_t H = (argc > 1) ? (size_t)strtoul(argv[1], NULL, 10) : 256;
  size_t V = (argc > 2) ? (size_t)strtoul(argv[2], NULL, 10) : 1024;
  size_t iters = (argc > 3) ? (size_t)strtoul(argv[3], NULL, 10) : 100;

  float *Wxh = (float*)malloc(H * H * sizeof(float));
  float *Whh = (float*)malloc(H * H * sizeof(float));
  float *Woh = (float*)malloc(V * H * sizeof(float));
  float *bh  = (float*)malloc(H * sizeof(float));
  float *x   = (float*)malloc(H * sizeof(float));
  float *h   = (float*)malloc(H * sizeof(float));
  float *tmp = (float*)malloc(H * sizeof(float));
  float *y   = (float*)malloc(V * sizeof(float));
  if (!Wxh || !Whh || !Woh || !bh || !x || !h || !tmp || !y) {
    fprintf(stderr, "alloc failed\n"); 
    free(Wxh); free(Whh); free(Woh); free(bh);
    free(x); free(h); free(tmp); free(y);
    return 1;
  }

  unsigned seed = 1234;
  init_rand(Wxh, H*H, &seed, 1.0f/32.0f);
  init_rand(Whh, H*H, &seed, 1.0f/32.0f);
  init_rand(Woh, V*H, &seed, 1.0f/64.0f);
  init_rand(bh,  H,   &seed, 1.0f/32.0f);
  memset(h, 0, H*sizeof(float));

  /* Warmup (1 decode-like iteration) */
  embed_token(1000u, x, H);
  gemv_rowmajor(Wxh, x, tmp, H, H);
  gemv_rowmajor(Whh, h, h, H, H);
  for (size_t i = 0; i < H; ++i) h[i] += tmp[i] + bh[i];
  vec_tanh(h, H);
  gemv_rowmajor(Woh, h, y, V, H);

  /* Timed loop: mimic decode step */
  double t0 = now_s();
  for (size_t it = 0; it < iters; ++it) {
    embed_token((unsigned)(1000u + (it & 31u)), x, H);
    gemv_rowmajor(Wxh, x, tmp, H, H);
    gemv_rowmajor(Whh, h, h, H, H);
    for (size_t i = 0; i < H; ++i) h[i] += tmp[i] + bh[i];
    vec_tanh(h, H);
    gemv_rowmajor(Woh, h, y, V, H);
  }
  double t1 = now_s();

  const double dt = t1 - t0;                 /* seconds */
  const double us_per_iter = (dt * 1e6) / (double)iters;
  const double elems = (double)(H*H + H*H + V*H);
  const double ns_per_elem = (dt * 1e9) / (iters * elems);
  const double bytes_total = elems * sizeof(float); /* rough RW per iter */
  const double gbps = (bytes_total * (double)iters / dt) / 1e9;

  printf("H=%zu V=%zu iters=%zu\n", H, V, iters);
  printf("time: %.3f ms total, %.1f us/iter, %.2f ns/elem (Wxh+Whh+Woh)\n",
         dt*1000.0, us_per_iter, ns_per_elem);
  printf("throughput: %.2f GB/s (approx memory traffic)\n", gbps);

  free(Wxh); free(Whh); free(Woh); free(bh);
  free(x); free(h); free(tmp); free(y);
  return 0;
}
