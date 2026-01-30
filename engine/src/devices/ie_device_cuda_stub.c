// File: engine/src/devices/ie_device_cuda_stub.c
#include "ie_device_cuda.h"
#include "ie_attn_cuda.h"
#include "ie_elem_cuda.h"
#include "ie_rmsnorm_cuda.h"
#include "ie_rope.h"

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

int ie_cuda_gemv_q4_0_f32_ex(const uint8_t *dW_q4,
                             const uint8_t *dW_scales,
                             size_t scale_bytes,
                             int scale_fmt,
                             const float *dx,
                             float *dy,
                             size_t rows,
                             size_t cols,
                             const uint16_t *dbias_bf16) {
  (void)dW_q4;
  (void)dW_scales;
  (void)scale_bytes;
  (void)scale_fmt;
  (void)dx;
  (void)dy;
  (void)rows;
  (void)cols;
  (void)dbias_bf16;
  return -1;
}

int ie_cuda_gemv_bf16_f32(const uint16_t *dW_bf16,
                          const float *dx,
                          float *dy,
                          size_t rows,
                          size_t cols,
                          const uint16_t *dbias_bf16) {
  (void)dW_bf16;
  (void)dx;
  (void)dy;
  (void)rows;
  (void)cols;
  (void)dbias_bf16;
  return -1;
}

int ie_cuda_argmax_f32_reduce(const float *d_x,
                              size_t n,
                              float *d_out_vals,
                              uint32_t *d_out_idx) {
  (void)d_x;
  (void)n;
  (void)d_out_vals;
  (void)d_out_idx;
  return -1;
}

int ie_rmsnorm_cuda_f32(const float *x, const float *w, float *y,
                        size_t rows, size_t cols, float eps) {
  (void)x;
  (void)w;
  (void)y;
  (void)rows;
  (void)cols;
  (void)eps;
  return -1;
}

int ie_cuda_rope_f32(float *q, float *k, size_t heads, size_t head_dim,
                     uint32_t pos, float theta, float pos_mul) {
  (void)q;
  (void)k;
  (void)heads;
  (void)head_dim;
  (void)pos;
  (void)theta;
  (void)pos_mul;
  return -1;
}

int ie_attn_cuda_causal_f32(const float *q, const float *K, const float *V,
                            size_t seq_len, size_t n_heads, size_t d_head,
                            float inv_sqrt_d, float *out) {
  (void)q;
  (void)K;
  (void)V;
  (void)seq_len;
  (void)n_heads;
  (void)d_head;
  (void)inv_sqrt_d;
  (void)out;
  return -1;
}

int ie_attn_cuda_causal_gqa_f32(const float *q, const float *K, const float *V,
                                size_t seq_len, size_t n_heads, size_t n_kv_heads,
                                size_t d_head, float inv_sqrt_d, float *out) {
  (void)q;
  (void)K;
  (void)V;
  (void)seq_len;
  (void)n_heads;
  (void)n_kv_heads;
  (void)d_head;
  (void)inv_sqrt_d;
  (void)out;
  return -1;
}

int ie_cuda_add_inplace_f32(float *dst, const float *src, size_t n) {
  (void)dst;
  (void)src;
  (void)n;
  return -1;
}

int ie_cuda_add_scaled_inplace_f32(float *dst, const float *src, float a, size_t n) {
  (void)dst;
  (void)src;
  (void)a;
  (void)n;
  return -1;
}

int ie_cuda_zero_f32(float *dst, size_t n) {
  (void)dst;
  (void)n;
  return -1;
}

int ie_cuda_fix_nonfinite_f32(float *dst, size_t n) {
  (void)dst;
  (void)n;
  return -1;
}

int ie_cuda_silu_mul_f32(const float *gate_up, const float *up, float *out, size_t n) {
  (void)gate_up;
  (void)up;
  (void)out;
  (void)n;
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
