/**
 * @file test_api.c
 * @brief Unit tests for the public engine API lifecycle.
 *
 * This verifies:
 *  - engine creation/destruction
 *  - a simple generation call into a caller-provided buffer
 *
 * Note: This test is executed from inside the model directory by the Makefile:
 *   ( cd models/gpt-oss-20b && ../../build/test_api )
 * So we pass model_dir="." to ie_engine_create().
 */

#include <assert.h>
#include <stddef.h>
#include <stdio.h>

#include "ie_api.h"

int main(void) {
  ie_engine_t *e = NULL;

  ie_engine_params_t p = {0};
  p.precision = "fp32";
  p.threads = 0;

  /* Create engine (device="cpu", model_dir="." because we run from model dir). */
  assert(ie_engine_create(&p, "cpu", ".", &e) == IE_OK);
  assert(e != NULL);

  /* Generate a small number of tokens. */
  enum { MAX_NEW = 8 };
  int out_tokens[MAX_NEW];
  size_t n_tokens = 0;

  assert(ie_engine_generate(e, "hi", (size_t)MAX_NEW, out_tokens, &n_tokens) == IE_OK);
  assert(n_tokens <= (size_t)MAX_NEW);

  /* Destroy engine. */
  ie_engine_destroy(e);

  printf("ok test_api\n");
  return 0;
}
