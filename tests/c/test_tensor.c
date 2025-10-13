/**
 * @file test_tensor.c
 * @brief Unit test: tensor view basics.
 */
#include <assert.h>
#include <stdio.h>
#include "ie_tensor.h"

int main(void) {
  float buf[4] = {1,2,3,4};
  ie_f32_vec_t v = ie_f32_vec(buf, 4);
  assert(v.len == 4 && v.data[2] == 3.0f);
  printf("ok test_tensor\n");
  return 0;
}
