/**
 * @file test_api.c
 * @brief Unit test: engine create/generate/metrics.
 */
#include <assert.h>
#include <stdio.h>
#include "ie_api.h"

int main(void) {
  ie_engine_t *e = NULL;
  ie_engine_params_t p = {0};
  p.precision = "fp32";
  assert(ie_engine_create(&p, &e) == IE_OK && e != NULL);

  uint32_t out[8]; uint32_t n = 0;
  assert(ie_engine_generate(e, "hi", 8, out, &n) == IE_OK);
  assert(n == 8);

  ie_metrics_t m;
  assert(ie_engine_metrics(e, &m) == IE_OK);
  assert(m.tps_true > 0.0);

  ie_engine_destroy(e);
  printf("ok test_api\n");
  return 0;
}
