/**
 * @file test_cpu_features.c
 * @brief Smoke test for CPU feature detection (never crashes, sane flags).
 */
#include <assert.h>
#include <stdio.h>
#include "ie_cpu.h"

int main(void) {
  ie_cpu_features_t f;
  assert(ie_cpu_detect(&f));
  /* Flags must be either true/false without crashing; no further assert. */
  printf("ok test_cpu_features (avx2=%d fma=%d)\n", (int)f.avx2, (int)f.fma);
  return 0;
}
