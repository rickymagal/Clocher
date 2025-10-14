/**
 * @file test_api.c
 * @brief Unit tests for the public engine API lifecycle and metrics.
 *
 * Validates engine creation/destruction, a simple generation run,
 * and the presence of expected metrics fields.
 */

#include <assert.h>
#include <stdio.h>
#include "ie_api.h"

/**
 * @brief Program entrypoint for API lifecycle tests.
 *
 * Creates an engine with minimal parameters, generates a small number
 * of tokens, fetches metrics, and validates sanity conditions.
 *
 * @return 0 on success.
 */
int main(void) {
  ie_engine_t *e = NULL;

  /* Minimal parameter set; paths may be stubs in the baseline. */
  ie_engine_params_t p = {0};
  p.precision = "fp32";

  /* Create engine */
  assert(ie_engine_create(&p, &e) == IE_OK);
  assert(e != NULL);

  /* Generate a small batch of tokens */
  uint32_t out[8];
  uint32_t n = 0;
  assert(ie_engine_generate(e, "hi", 8, out, &n) == IE_OK);
  assert(n == 8);

  /* Fetch metrics and validate */
  ie_metrics_t m;
  assert(ie_engine_metrics(e, &m) == IE_OK);
  assert(m.tps_true >= 0.0);
  assert(m.latency_p50_ms >= 0.0);
  assert(m.latency_p95_ms >= 0.0);

  /* Destroy engine (idempotent if called once) */
  ie_engine_destroy(e);

  printf("ok test_api\n");
  return 0;
}
