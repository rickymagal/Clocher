/**
 * @file microbench_stream.c
 * @brief Sweep prefetch distances and NT ratios for streaming copy bandwidth.
 *
 * This microbenchmark:
 *  - Allocates two large float buffers with 64-byte alignment.
 *  - Warms up the cache and TLBs.
 *  - Sweeps a set of prefetch distances and nt_ratio values.
 *  - Measures effective bandwidth (GiB/s) for copy and prints CSV.
 *
 * Build (example):
 *   cc -std=c11 -O3 -Wall -Wextra -Werror -pedantic -Iengine/include \
 *      engine/src/opt/stream.c benchmarks/src/microbench_stream.c \
 *      -o build/microbench_stream -lpthread -lm
 *
 * Run (example):
 *   ./build/microbench_stream --mb 512 --iters 8
 *   # --mb is MiB of buffers, --iters is repeat count per configuration.
 */

#include "ie_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------ time helpers ------------------------------ */

/**
 * @brief Read high-resolution monotonic time (seconds).
 *
 * @return double seconds.
 */
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ------------------------------ cmdline args ------------------------------ */

/**
 * @brief Simple command-line parsing.
 *
 * Recognized flags:
 *   --mb <MiB>     : buffer size in MiB (per array). Default 256.
 *   --iters <N>    : iterations per configuration. Default 6.
 */
static void parse_args(int argc, char **argv, size_t *mb, int *iters) {
  *mb = 256;
  *iters = 6;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--mb") == 0 && i + 1 < argc) {
      *mb = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
      *iters = atoi(argv[++i]);
    }
  }
}

/* -------------------------------- main bench ------------------------------ */

/**
 * @brief Microbenchmark main.
 *
 * CSV header:
 *   distance_bytes,nt_ratio,MiB,GiBps,seconds
 */
int main(int argc, char **argv) {
  size_t mb;
  int iters;
  parse_args(argc, argv, &mb, &iters);

  const size_t bytes = mb * 1024ull * 1024ull;
  const size_t n = bytes / sizeof(float);

  /* 64-byte aligned allocations */
  float *src = NULL, *dst = NULL;
  if (posix_memalign((void **)&src, 64, bytes) != 0) return 1;
  if (posix_memalign((void **)&dst, 64, bytes) != 0) return 1;

  for (size_t i = 0; i < n; ++i) src[i] = (float)(i & 0xFF) * 0.25f;

  struct ie_stream_policy pol;
  if (ie_stream_policy_init(&pol) != 0) return 2;

  /* Sweep sets */
  const size_t distances[] = { 64, 128, 256, 384, 512, 768, 1024, 1536 };
  const double ratios[]    = { 0.0, 0.25, 0.50, 0.75, 1.0 };

  printf("distance_bytes,nt_ratio,MiB,GiBps,seconds\n");

  /* Warm up: a few scalar copies to fault in pages */
  for (int w = 0; w < 3; ++w) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
  }

  for (size_t di = 0; di < sizeof(distances)/sizeof(distances[0]); ++di) {
    const size_t dist = distances[di];
    for (size_t ri = 0; ri < sizeof(ratios)/sizeof(ratios[0]); ++ri) {
      const double ratio = ratios[ri];

      /* Use the chosen distance; disable distance capping for the bench. */
      pol.prefetch_distance = dist;
      pol.enable_prefetch = true;

      /* Time several iterations; take the best (or average). */
      double best = 1e9;
      for (int it = 0; it < iters; ++it) {
        /* Prefetch source (T0) a full pass ahead of copy. */
        ie_stream_prefetch_range_t0(src, bytes, IE_CACHELINE_BYTES, dist);

        const double t0 = now_sec();
        ie_stream_copy_f32(dst, src, n, &pol, ratio);
        const double t1 = now_sec();

        const double dt = t1 - t0;
        if (dt < best) best = dt;

        /* Touch a byte to avoid dead-store elimination. */
        volatile float guard = dst[(it + 17) % n];
        (void)guard;
      }

      const double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
      const double gibps = gib / best;
      printf("%zu,%.2f,%zu,%.6f,%.6f\n", dist, ratio, mb, gibps, best);
      fflush(stdout);
    }
  }

  free(src);
  free(dst);
  return 0;
}
