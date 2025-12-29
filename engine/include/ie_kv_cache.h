/* ============================================================================
 * File: engine/include/ie_kv_cache.h
 * ============================================================================
 */
#ifndef IE_KV_CACHE_H
#define IE_KV_CACHE_H

/**
 * @file ie_kv_cache.h
 * @brief Key/Value attention cache with INT8/FP8 compression and float round-trip.
 *
 * This module provides a compact and fast KV cache for transformer attention
 * that stores per-token K/V rows using either raw FP32, INT8 with affine
 * de/quantization, or FP8 (E4M3/E5M2). It is designed for:
 *   - Low-allocation hot paths: preallocate once for (heads, max_seq, head_dim).
 *   - Explicit memory layout for predictable bandwidth.
 *   - Per-token store/load helpers to integrate with decode loops.
 *
 * Memory layout:
 * Let H = heads, T = max_seq, D = head_dim, G = ceil(D / group_size).
 *
 * K and V payload layout:
 *   index(t,h,d) = ((t * H + h) * D + d)
 *
 * For INT8 storage, per-group parameters (scale, zero_point) are tracked
 * per token and per head for both K and V:
 *   param_index(t,h,g) = ((t * H + h) * G + g)
 */

#include <stddef.h>
#include <stdint.h>
#include "ie_quant_act.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ie_kv_storage_type {
  IE_KV_STORAGE_F32  = 0,
  IE_KV_STORAGE_INT8 = 1,
  IE_KV_STORAGE_FP8  = 2
} ie_kv_storage_type;

typedef struct ie_kv_opts {
  int heads;
  int head_dim;
  int max_seq;
  ie_kv_storage_type storage;
  size_t group_size;
  int symmetric;
  ie_fp8_format fp8_format;
} ie_kv_opts;

typedef struct ie_kv_cache {
  int heads;
  int head_dim;
  int max_seq;

  ie_kv_storage_type storage;
  size_t group_size;
  int symmetric;
  ie_fp8_format fp8_format;

  size_t elem_count;
  size_t group_count;

  void* K;
  void* V;

  float*  scales_K;
  int8_t* zeros_K;
  float*  scales_V;
  int8_t* zeros_V;
} ie_kv_cache;

int ie_kv_init(ie_kv_cache* kv, const ie_kv_opts* opts);
void ie_kv_free(ie_kv_cache* kv);

size_t ie_kv_element_size(ie_kv_storage_type storage);

int ie_kv_store_token_f32(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32);

int ie_kv_store_token_int8_per_tensor(ie_kv_cache* kv, size_t t,
                                      const float* K_f32, const float* V_f32);

int ie_kv_store_token_int8_per_group(ie_kv_cache* kv, size_t t,
                                     const float* K_f32, const float* V_f32);

int ie_kv_store_token_fp8(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32);

int ie_kv_load_token_f32(const ie_kv_cache* kv, size_t t,
                         float* K_out, float* V_out);

int ie_kv_raw_ptrs(ie_kv_cache* kv, void** out_K, void** out_V, size_t* out_lds);

/* Strides (in elements) for the [t][h][d] logical layout: d=1, h=D, t=H*D. */
int ie_kv_raw_strides(const ie_kv_cache* kv, size_t* stride_t, size_t* stride_h, size_t* stride_d);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_KV_CACHE_H */
