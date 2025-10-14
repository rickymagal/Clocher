/**
 * @file test_tensor.c
 * @brief Unit tests for basic tensor view helpers.
 *
 * This suite validates the minimal vector view constructor and ensures
 * the memory is referenced correctly without ownership semantics.
 */

#include <assert.h>
#include <stdio.h>
#include "ie_tensor.h"

/**
 * @brief Program entrypoint for the tensor tests.
 *
 * Creates a small buffer, wraps it in a vector view, and validates
 * length and element access invariants.
 *
 * @return 0 on success.
 */
int main(void) {
  float buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  /* Construct view and check invariants */
  ie_f32_vec_t v = ie_f32_vec(buf, 4);
  assert(v.len == 4);
  assert(v.data == buf);
  assert(v.data[0] == 1.0f);
  assert(v.data[2] == 3.0f);

  printf("ok test_tensor\n");
  return 0;
}
