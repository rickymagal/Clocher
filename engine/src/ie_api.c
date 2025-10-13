/**
 * @file ie_api.c
 * @brief Baseline engine implementation (dummy generator).
 *
 * This file provides a dependency-free engine that:
 *  - Accepts params and constructs a handle.
 *  - Generates deterministic "fake" tokens quickly.
 *  - Tracks simple timing to compute True TPS.
 *  - Returns metrics via ie_engine_metrics.
 *
 * Replace the generation path in Step 2–3 with real kernel calls.
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>       /* C11 timespec_get */
#include "ie_api.h"

typedef struct ie_engine {
  unsigned long long tokens_generated;
  double last_latency_ms_p50;
  double last_latency_ms_p95;
} ie_engine;

/* Portable timestamp (seconds) using C11 timespec_get. */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return IE_ERR_INVALID_ARGUMENT;
  (void)p; /* unused in baseline */
  ie_engine *e = (ie_engine*)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_INTERNAL;
  *out = (ie_engine_t*)e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  free(h);
}

ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_tokens || !out_count) return IE_ERR_INVALID_ARGUMENT;
  (void)prompt; /* unused in baseline */

  /* Deterministic pseudo-work: spin ~0.2 ms per token to emulate latency. */
  const double t_start = now_s();
  const uint32_t n = (max_new_tokens == 0) ? 0u : max_new_tokens;
  for (uint32_t i = 0; i < n; ++i) {
    out_tokens[i] = 1000u + (i % 127u); /* stable fake token IDs */
    const double t0 = now_s();
    while ((now_s() - t0) < 0.0002) { /* ~0.2 ms busy-wait */ }
  }
  const double t_end = now_s();

  /* Update simple stats */
  ie_engine *e = (ie_engine*)h;
  e->tokens_generated += n;
  const double per_token_ms = (n ? ((t_end - t_start) * 1000.0 / (double)n) : 0.0);
  e->last_latency_ms_p50 = per_token_ms;
  e->last_latency_ms_p95 = per_token_ms * 1.2; /* naive scale */

  *out_count = n;
  return IE_OK;
}

ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return IE_ERR_INVALID_ARGUMENT;
  const ie_engine *e = (const ie_engine*)h;

  double tps = 0.0;
  if (e->last_latency_ms_p50 > 0.0) {
    tps = 1000.0 / e->last_latency_ms_p50;
  }

  out->tps_true       = tps;
  out->latency_p50_ms = e->last_latency_ms_p50;
  out->latency_p95_ms = e->last_latency_ms_p95;
  out->rss_peak_mb    = 0;                /* not tracked in baseline */
  out->kv_hits        = 0;
  out->kv_misses      = e->tokens_generated;
  return IE_OK;
}
