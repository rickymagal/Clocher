/**
 * @file test_api.c
 * @brief Unit tests for the public engine API surface.
 *
 * Default mode (always runs in `make test`):
 *  - Validates basic argument checks and NULL-safety.
 *  - Does NOT require any model artifacts.
 *
 * Optional integration mode (opt-in):
 *  - Loads a model from the current working directory and runs a tiny generate.
 *  - Enable with: IE_TEST_INTEGRATION=1
 *
 * Notes:
 *  - The Makefile runs this binary from inside the model directory:
 *      ( cd models/gpt-oss-20b && ../../build/test_api )
 *    So model_dir="." is correct in integration mode.
 */

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_api.h"

static int env_truthy_(const char *name) {
  const char *v = getenv(name);
  if (!v || !*v) return 0;
  return (strcmp(v, "1") == 0) || (strcmp(v, "true") == 0) || (strcmp(v, "TRUE") == 0) ||
         (strcmp(v, "yes") == 0) || (strcmp(v, "YES") == 0) || (strcmp(v, "on") == 0) ||
         (strcmp(v, "ON") == 0);
}

static int file_exists_(const char *path) {
  if (!path || !*path) return 0;
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

static const char *pick_tokenizer_path_(void) {
  if (file_exists_("tokenizer.padded.json")) return "tokenizer.padded.json";
  if (file_exists_("tokenizer.json")) return "tokenizer.json";
  return NULL;
}

static const char *pick_precision_(void) {
  const char *v = getenv("IE_TEST_API_PRECISION");
  if (v && *v) return v;

  /* If int4 compat json is present, prefer the path you actually run day-to-day. */
  if (file_exists_("model.ie.compat.json")) return "int4";
  return "fp32";
}

static const char *pick_weights_json_(const char *precision) {
  if (precision && strcmp(precision, "int4") == 0) {
    if (file_exists_("model.ie.compat.json")) return "model.ie.compat.json";
  }
  if (file_exists_("model.ie.json")) return "model.ie.json";
  return NULL;
}

static void run_contract_tests_(void) {
  /* destroy(): must be NULL-safe */
  ie_engine_destroy(NULL);

  /* generate(): must reject NULL engine */
  {
    int out_tokens[1] = {0};
    size_t n = 123;
    ie_status_t st = ie_engine_generate(NULL, "hi", 1u, out_tokens, &n);
    assert(st == IE_ERR_BADARG);
  }

  /* create(): must reject NULL model_dir */
  {
    ie_engine_t *e = NULL;
    ie_status_t st = ie_engine_create(NULL, "cpu", NULL, &e);
    assert(st == IE_ERR_BADARG);
  }

  /* create(): must reject NULL out */
  {
    ie_status_t st = ie_engine_create(NULL, "cpu", ".", NULL);
    assert(st == IE_ERR_BADARG);
  }
}

static int run_integration_(void) {
  const char *tok = pick_tokenizer_path_();
  const char *prec = pick_precision_();
  const char *wjson = pick_weights_json_(prec);

  if (!tok || !wjson) {
    printf("ok test_api (integration skipped: missing tokenizer/model json in cwd)\n");
    return 0;
  }

  ie_engine_t *e = NULL;

  ie_engine_params_t p;
  memset(&p, 0, sizeof(p));
  p.precision = prec;
  p.threads = 0;
  p.tokenizer_path = tok;
  p.weights_json_path = wjson;

  ie_status_t st = ie_engine_create(&p, "cpu", ".", &e);
  if (st != IE_OK || !e) {
    fprintf(stderr,
            "test_api integration failed: ie_engine_create rc=%d (precision=%s tokenizer=%s weights_json=%s)\n",
            (int)st, prec, tok, wjson);
    return 1;
  }

  enum { MAX_NEW = 8 };
  int out_tokens[MAX_NEW];
  size_t n_tokens = 0;

  st = ie_engine_generate(e, "hi", (size_t)MAX_NEW, out_tokens, &n_tokens);
  if (st != IE_OK) {
    fprintf(stderr, "test_api integration failed: ie_engine_generate rc=%d\n", (int)st);
    ie_engine_destroy(e);
    return 1;
  }

  assert(n_tokens <= (size_t)MAX_NEW);

  ie_engine_destroy(e);
  printf("ok test_api (integration)\n");
  return 0;
}

int main(void) {
  run_contract_tests_();

  if (env_truthy_("IE_TEST_INTEGRATION")) {
    return run_integration_();
  }

  printf("ok test_api\n");
  return 0;
}
