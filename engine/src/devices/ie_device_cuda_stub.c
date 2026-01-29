// File: engine/src/devices/ie_device_cuda_stub.c
#include "ie_device_cuda.h"

#include <stddef.h>
#include <stdint.h>

int ie_cuda_is_available(void) {
  return 0;
}

int ie_cuda_init(int device_ordinal,
                 char *out_name, size_t name_cap,
                 char *out_driver, size_t driver_cap) {
  (void)device_ordinal;
  if (out_name && name_cap) out_name[0] = '\0';
  if (out_driver && driver_cap) out_driver[0] = '\0';
  return -1;
}

void *ie_cuda_malloc(size_t nbytes) {
  (void)nbytes;
  return NULL;
}

void ie_cuda_free(void *p) {
  (void)p;
}

int ie_cuda_memcpy(void *dst, const void *src, size_t nbytes, ie_cuda_copy_kind_t kind) {
  (void)dst;
  (void)src;
  (void)nbytes;
  (void)kind;
  return -1;
}

int ie_cuda_gemv_f32(const float *dW,
                     const float *dx,
                     float *dy,
                     size_t rows,
                     size_t cols,
                     const float *dbias) {
  (void)dW;
  (void)dx;
  (void)dy;
  (void)rows;
  (void)cols;
  (void)dbias;
  return -1;
}

int ie_cuda_gemv_q4_0_f32(const uint8_t *dW_q4,
                          const uint8_t *dW_scales,
                          size_t scale_bytes,
                          const float *dx,
                          float *dy,
                          size_t rows,
                          size_t cols,
                          const uint16_t *dbias_bf16) {
  (void)dW_q4;
  (void)dW_scales;
  (void)scale_bytes;
  (void)dx;
  (void)dy;
  (void)rows;
  (void)cols;
  (void)dbias_bf16;
  return -1;
}

const char *ie_cuda_last_error_string(void) {
  return "CUDA unavailable (stub)";
}

void ie_cuda_clear_last_error(void) {
}

int ie_cuda_mem_get_info(size_t *out_free, size_t *out_total) {
  if (out_free) *out_free = 0;
  if (out_total) *out_total = 0;
  return -1;
}
