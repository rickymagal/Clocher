/**
 * @file test_device.c
 * @brief Unit tests for ie_device API (selection and CPU fallback).
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "ie_device.h"

/**
 * @brief Verify device kind parsing for common and unknown strings.
 */
static void test_kind_parse(void) {
  assert(ie_device_kind_from_str(NULL) == IE_DEV_CPU);
  assert(ie_device_kind_from_str("cpu") == IE_DEV_CPU);
  assert(ie_device_kind_from_str("CUDA") == IE_DEV_CUDA);
  assert(ie_device_kind_from_str("ze") == IE_DEV_ZE);
  assert(ie_device_kind_from_str("unknown") == IE_DEV_CPU);
}

/**
 * @brief Verify CPU device exposes GEMV capability.
 */
static void test_cpu_caps(void) {
  ie_device_t *d = NULL;
  assert(ie_device_create(IE_DEV_CPU, &d) == 0 && d);
  ie_device_caps_t caps;
  assert(ie_device_caps(d, &caps) == 0);
  assert(caps.has_gemv_f32 == 1);
  ie_device_destroy(d);
}

/**
 * @brief Ensure GPU request gracefully falls back to a working GEMV.
 *
 * If CUDA/ZE are unavailable, create() returns a CPU-backed device and
 * gemv completes successfully.
 */
static void test_gpu_fallback(void) {
  ie_device_t *d = NULL;
  assert(ie_device_create(IE_DEV_CUDA, &d) == 0 && d);
  float W[4] = {1,2,3,4}, x[2] = {1,1}, y[2] = {0,0};
  int rc = ie_device_gemv_f32(d, W, x, y, 2, 2, NULL, 0);
  assert(rc == 0);
  assert((int)(y[0] + 1e-4f) == 3);
  assert((int)(y[1] + 1e-4f) == 7);
  ie_device_destroy(d);
}

/**
 * @brief Test entry point.
 *
 * @return 0 on success.
 */
int main(void) {
  test_kind_parse();
  test_cpu_caps();
  test_gpu_fallback();
  puts("ok test_device");
  return 0;
}
