/**
 * @file test_weights.c
 * @brief Unit tests for IEBIN v1 loader (baseline).
 */
#include <assert.h>
#include <stdio.h>
#include "ie_io.h"

int main(void) {
  ie_weights_t w;
  /* Expect success with our stub files in models/gpt-oss-20b */
  int rc = ie_weights_open("models/gpt-oss-20b/model.ie.json",
                           "models/gpt-oss-20b/model.ie.bin",
                           &w);
  assert(rc == 0);
  assert(w.version >= 1);
  /* dtype should be populated or defaulted to fp32 */
  assert(w.dtype[0] != '\0');

  /* Close is a no-op */
  ie_weights_close(&w);
  printf("ok test_weights\n");
  return 0;
}
