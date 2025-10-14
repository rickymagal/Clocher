/*
 * @file main_infer.c
 * @brief CLI entrypoint: creates the engine, runs a generation, prints JSON metrics.
 *
 * Usage:
 *   ./build/inference-engine "Your prompt here"
 *
 * This file is dependency-free and uses C11 timespec_get() for timing.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>        /* C11 timespec_get */
#include "ie_api.h"

/**
 * @brief Print a single JSON object with run metrics to stdout.
 *
 * @param tokens Number of tokens generated in this run.
 * @param wall_s End-to-end wall time in seconds for the run (prompt excluded).
 * @param m Snapshot of engine metrics (p50/p95, tps estimate, etc.).
 */
static void print_json_metrics(unsigned tokens, double wall_s, const ie_metrics_t *m) {
  printf("{\"tokens_generated\":%u,\"wall_time_s\":%.6f,"
         "\"tps_true\":%.6f,\"latency_p50_ms\":%.3f,\"latency_p95_ms\":%.3f,"
         "\"rss_peak_mb\":%zu,\"kv_hits\":%llu,\"kv_misses\":%llu}\n",
         tokens, wall_s, m->tps_true, m->latency_p50_ms, m->latency_p95_ms,
         m->rss_peak_mb, (unsigned long long)m->kv_hits, (unsigned long long)m->kv_misses);
}

/**
 * @brief Portable timestamp in seconds using C11 timespec_get.
 *
 * @return Current time in seconds (monotonic-ish UTC-based).
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief CLI main entry point.
 *
 * Creates the engine, runs a single generation, prints a JSON metrics line.
 *
 * @param argc Argument count.
 * @param argv Argument vector (argv[1] may hold the prompt).
 * @return 0 on success; non-zero on failure.
 */
int main(int argc, char **argv) {
  const char *prompt = (argc > 1) ? argv[1] : "Hello,";
  const uint32_t max_new = 16;

  ie_engine_params_t params = {
    .weights_path    = "models/gpt-oss-20b/model.ie.bin",
    .shape_json_path = "models/gpt-oss-20b/model.ie.json",
    .vocab_path      = "models/gpt-oss-20b/vocab.json",
    .threads         = 0,
    .affinity        = "auto",
    .precision       = "fp32"
  };

  ie_engine_t *eng = NULL;
  if (ie_engine_create(&params, &eng) != IE_OK) {
    fprintf(stderr, "engine: create failed\n");
    return 1;
  }

  uint32_t *buf = (uint32_t*)calloc(max_new, sizeof(uint32_t));
  if (!buf) {
    fprintf(stderr, "engine: out of memory\n");
    ie_engine_destroy(eng);
    return 1;
  }

  uint32_t n = 0;
  const double t0 = now_s();
  if (ie_engine_generate(eng, prompt, max_new, buf, &n) != IE_OK) {
    fprintf(stderr, "engine: generate failed\n");
    free(buf); ie_engine_destroy(eng);
    return 2;
  }
  const double t1 = now_s();

  ie_metrics_t m;
  if (ie_engine_metrics(eng, &m) != IE_OK) {
    fprintf(stderr, "engine: metrics failed\n");
    free(buf); ie_engine_destroy(eng);
    return 3;
  }

  print_json_metrics(n, (t1 - t0), &m);

  free(buf);
  ie_engine_destroy(eng);
  return 0;
}
