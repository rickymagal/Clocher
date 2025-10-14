/**
 * @file test_weights.c
 * @brief Unit tests for the IEBIN v1 lightweight loader.
 *
 * Ensures that the JSON metadata can be opened and parsed with relaxed scanning,
 * that defaults are populated, and that closing is safe.
 */

#include <assert.h>
#include <stdio.h>
#include "ie_io.h"

/**
 * @brief Program entrypoint for weights loader tests.
 *
 * Opens the stub metadata included in the repo, validates fields such as
 * version and dtype, and exercises the close path.
 *
 * @return 0 on success.
 */
int main(void) {
  ie_weights_t w;

  /* Expect success with the stub files shipped in models/gpt-oss-20b */
  int rc = ie_weights_open("models/gpt-oss-20b/model.ie.json",
                           "models/gpt-oss-20b/model.ie.bin",
                           &w);
  assert(rc == 0);

  /* version must be >= 1; dtype must not be empty (default "fp32" allowed) */
  assert(w.version >= 1);
  assert(w.dtype[0] != '\0');

  /* Close is a no-op in the baseline, but call for symmetry */
  ie_weights_close(&w);

  printf("ok test_weights\n");
  return 0;
}
