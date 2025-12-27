// File: engine/src/devices/ie_device_cuda_stub.c
#include <stddef.h>

int ie_cuda_is_available(void) {
  return 0;
}

int ie_cuda_init(int gpu_id) {
  (void)gpu_id;
  return -1;
}

int ie_cuda_malloc(void **out, size_t bytes) {
  (void)bytes;
  if (out) *out = NULL;
  return -1;
}

int ie_cuda_free(void *ptr) {
  (void)ptr;
  return 0;
}

int ie_cuda_memcpy(void *dst, const void *src, size_t bytes, int kind) {
  (void)dst;
  (void)src;
  (void)bytes;
  (void)kind;
  return -1;
}

int ie_cuda_device_synchronize(void) {
  return -1;
}

const char *ie_cuda_get_error_string(int err) {
  (void)err;
  return "CUDA unavailable (stub)";
}

int ie_cuda_gemv_f32(void *out, const void *w, const void *x, int rows, int cols) {
  (void)out;
  (void)w;
  (void)x;
  (void)rows;
  (void)cols;
  return -1;
}
