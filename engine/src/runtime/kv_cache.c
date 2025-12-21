/**
 * @file kv_cache.c
 * @brief Implementation of a compact Key/Value attention cache with INT8/FP8 compression.
 *
 * The cache stores per-token K/V slices laid out as [heads, head_dim] per token,
 * flattened with head_dim as the fastest-varying dimension:
 *
 *   index(t,h,d) = ((t * H + h) * D + d)
 *
 * For INT8 storage, per-group affine parameters are kept for each (token, head).
 * The parameter array index is:
 *
 *   param_index(t,h,g) = ((t * H + h) * G + g)
 *
 * where G = ceil(D / group_size).
 *
 * This file contains no dynamic threading or allocator hooks; callers control
 * parallelism outside and provide FP32 views on demand via ::ie_kv_load_token_f32.
 *
 * KV hit/miss instrumentation:
 *   This module can optionally report "KV hits" and "KV misses" via the metrics
 *   layer. We define safe no-op fallbacks so this file compiles regardless of
 *   whether the build exposes hit/miss macros.
 *
 *   Semantics (within this module):
 *     - hit:  ie_kv_load_token_f32() successfully loads token t
 *     - miss: load request is out of range (t >= max_seq) or storage unknown
 */

#include "ie_kv_cache.h"
#include "ie_quant_act.h"

/* Optional: metrics hooks (safe fallbacks). */
#include "ie_metrics.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ----------------------------- instrumentation ----------------------------- */

/**
 * @brief Add KV hit count (safe fallback).
 *
 * @param n Number of hits to add.
 */
static inline void ie_kv_hit_add_local(uint64_t n) {
#if defined(IE_KV_HIT)
  IE_KV_HIT(n);
#else
  (void)n;
#endif
}

/**
 * @brief Add KV miss count (safe fallback).
 *
 * @param n Number of misses to add.
 */
static inline void ie_kv_miss_add_local(uint64_t n) {
#if defined(IE_KV_MISS)
  IE_KV_MISS(n);
#else
  (void)n;
#endif
}

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Compute ceil_div(a,b) for size_t.
 */
static inline size_t ceil_div_sz(size_t a, size_t b) {
  return (a + b - 1u) / b;
}

/**
 * @brief Compute param index for (t,h,g) triplet.
 */
static inline size_t param_index(size_t t, size_t h, size_t g,
                                 size_t H, size_t G) {
  return ((t * H) + h) * G + g;
}

/**
 * @brief Compute payload index for (t,h,d) triplet.
 */
static inline size_t payload_index(size_t t, size_t h, size_t d,
                                   size_t H, size_t D) {
  return ((t * H) + h) * D + d;
}

/**
 * @brief Validate options and populate derived fields in @p kv (no allocation).
 *
 * @return 0 on success; non-zero on bad options.
 */
static int kv_validate_and_derive(ie_kv_cache* kv, const ie_kv_opts* opts) {
  if (!kv || !opts) return -1;
  if (opts->heads <= 0 || opts->head_dim <= 0 || opts->max_seq <= 0) return -1;
  if (opts->storage != IE_KV_STORAGE_F32 &&
      opts->storage != IE_KV_STORAGE_INT8 &&
      opts->storage != IE_KV_STORAGE_FP8) return -1;

  kv->heads      = opts->heads;
  kv->head_dim   = opts->head_dim;
  kv->max_seq    = opts->max_seq;
  kv->storage    = opts->storage;
  kv->group_size = (opts->group_size > 0 ? opts->group_size : 1);
  kv->symmetric  = opts->symmetric;
  kv->fp8_format = opts->fp8_format;

  kv->elem_count  = (size_t)kv->heads * (size_t)kv->max_seq * (size_t)kv->head_dim;
  kv->group_count = ceil_div_sz((size_t)kv->head_dim, kv->group_size);

  kv->K = kv->V = NULL;
  kv->scales_K = kv->scales_V = NULL;
  kv->zeros_K  = kv->zeros_V  = NULL;

  return 0;
}

/* --------------------------------- public API ------------------------------ */

size_t ie_kv_element_size(ie_kv_storage_type storage) {
  switch (storage) {
    case IE_KV_STORAGE_F32:  return sizeof(float);
    case IE_KV_STORAGE_INT8: return sizeof(int8_t);
    case IE_KV_STORAGE_FP8:  return sizeof(uint8_t);
    default: return 0;
  }
}

int ie_kv_init(ie_kv_cache* kv, const ie_kv_opts* opts) {
  if (kv_validate_and_derive(kv, opts) != 0) return -1;

  const size_t elem_size = ie_kv_element_size(kv->storage);
  if (elem_size == 0) return -1;

  const size_t bytes = kv->elem_count * elem_size;

  kv->K = calloc(1, bytes);
  kv->V = calloc(1, bytes);
  if (!kv->K || !kv->V) {
    ie_kv_free(kv);
    return -2;
  }

  if (kv->storage == IE_KV_STORAGE_INT8) {
    const size_t params_len = (size_t)kv->max_seq * (size_t)kv->heads * kv->group_count;
    kv->scales_K = (float*)calloc(params_len, sizeof(float));
    kv->zeros_K  = (int8_t*)calloc(params_len, sizeof(int8_t));
    kv->scales_V = (float*)calloc(params_len, sizeof(float));
    kv->zeros_V  = (int8_t*)calloc(params_len, sizeof(int8_t));
    if (!kv->scales_K || !kv->zeros_K || !kv->scales_V || !kv->zeros_V) {
      ie_kv_free(kv);
      return -2;
    }
  }

  return 0;
}

void ie_kv_free(ie_kv_cache* kv) {
  if (!kv) return;
  if (kv->K) { free(kv->K); kv->K = NULL; }
  if (kv->V) { free(kv->V); kv->V = NULL; }
  if (kv->scales_K) { free(kv->scales_K); kv->scales_K = NULL; }
  if (kv->zeros_K)  { free(kv->zeros_K);  kv->zeros_K  = NULL; }
  if (kv->scales_V) { free(kv->scales_V); kv->scales_V = NULL; }
  if (kv->zeros_V)  { free(kv->zeros_V);  kv->zeros_V  = NULL; }

  /* Keep metadata so double-free is harmless; caller may reuse the struct. */
  kv->elem_count = 0;
  kv->group_count = 0;
}

int ie_kv_store_token_f32(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_F32) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;

  float* Kdst = (float*)kv->K;
  float* Vdst = (float*)kv->V;

  /* Copy [heads, head_dim] block for token t. */
  for (size_t h = 0; h < H; ++h) {
    const size_t off = payload_index(t, h, 0, H, D);
    memcpy(Kdst + off, K_f32 + h * D, D * sizeof(float));
    memcpy(Vdst + off, V_f32 + h * D, D * sizeof(float));
  }
  return 0;
}

int ie_kv_store_token_int8_per_tensor(ie_kv_cache* kv, size_t t,
                                      const float* K_f32, const float* V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_INT8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;
  const size_t G = kv->group_count; /* per-tensor uses g=0 */

  int8_t* Kdst = (int8_t*)kv->K;
  int8_t* Vdst = (int8_t*)kv->V;

  for (size_t h = 0; h < H; ++h) {
    const float* Ksrc = K_f32 + h * D;
    const float* Vsrc = V_f32 + h * D;

    /* Compute per-head params from min/max. */
    float Kmn = Ksrc[0], Kmx = Ksrc[0];
    float Vmn = Vsrc[0], Vmx = Vsrc[0];
    for (size_t d = 1; d < D; ++d) {
      float kvv = Ksrc[d];
      float vvv = Vsrc[d];
      if (kvv < Kmn) Kmn = kvv;
      if (kvv > Kmx) Kmx = kvv;
      if (vvv < Vmn) Vmn = vvv;
      if (vvv > Vmx) Vmx = vvv;
    }

    ie_act_i8_params pK, pV;
    ie_act_i8_params_from_minmax(Kmn, Kmx, kv->symmetric, &pK.scale, &pK.zero_point);
    ie_act_i8_params_from_minmax(Vmn, Vmx, kv->symmetric, &pV.scale, &pV.zero_point);

    /* Quantize into destination payload. */
    const size_t off = payload_index(t, h, 0, H, D);
    ie_quantize_act_int8(Ksrc, Kdst + off, D, pK, kv->symmetric);
    ie_quantize_act_int8(Vsrc, Vdst + off, D, pV, kv->symmetric);

    /* Record parameters at g=0. */
    const size_t pi = param_index(t, h, 0, H, G);
    kv->scales_K[pi] = pK.scale;
    kv->zeros_K[pi]  = pK.zero_point;
    kv->scales_V[pi] = pV.scale;
    kv->zeros_V[pi]  = pV.zero_point;
  }
  return 0;
}

int ie_kv_store_token_int8_per_group(ie_kv_cache* kv, size_t t,
                                     const float* K_f32, const float* V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_INT8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;
  const size_t G = kv->group_count;
  const size_t Gs = kv->group_size;

  int8_t* Kdst = (int8_t*)kv->K;
  int8_t* Vdst = (int8_t*)kv->V;

  /* Temporary parameter buffers per head. */
  float*  scales = (float*)malloc(G * sizeof(float));
  int8_t* zeros  = (int8_t*)malloc(G * sizeof(int8_t));
  if (!scales || !zeros) { free(scales); free(zeros); return -4; }

  for (size_t h = 0; h < H; ++h) {
    const float* Ksrc = K_f32 + h * D;
    const float* Vsrc = V_f32 + h * D;
    const size_t off  = payload_index(t, h, 0, H, D);

    /* K */
    ie_act_i8_group_params_from_data(Ksrc, D, Gs, kv->symmetric, scales, zeros);
    ie_quantize_act_int8_per_group(Ksrc, Kdst + off, D, Gs, scales, zeros, kv->symmetric);
    for (size_t g = 0; g < G; ++g) {
      const size_t pi = param_index(t, h, g, H, G);
      kv->scales_K[pi] = scales[g];
      kv->zeros_K[pi]  = zeros[g];
    }

    /* V */
    ie_act_i8_group_params_from_data(Vsrc, D, Gs, kv->symmetric, scales, zeros);
    ie_quantize_act_int8_per_group(Vsrc, Vdst + off, D, Gs, scales, zeros, kv->symmetric);
    for (size_t g = 0; g < G; ++g) {
      const size_t pi = param_index(t, h, g, H, G);
      kv->scales_V[pi] = scales[g];
      kv->zeros_V[pi]  = zeros[g];
    }
  }

  free(scales);
  free(zeros);
  return 0;
}

int ie_kv_store_token_fp8(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_FP8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;

  uint8_t* Kdst = (uint8_t*)kv->K;
  uint8_t* Vdst = (uint8_t*)kv->V;

  /* Temporary FP8 buffers per head to avoid partial writes on error. */
  uint8_t* tmp = (uint8_t*)malloc(D * sizeof(uint8_t));
  if (!tmp) return -4;

  for (size_t h = 0; h < H; ++h) {
    const float* Ksrc = K_f32 + h * D;
    const float* Vsrc = V_f32 + h * D;
    const size_t off  = payload_index(t, h, 0, H, D);

    ie_quantize_act_fp8(Ksrc, tmp, D, kv->fp8_format);
    memcpy(Kdst + off, tmp, D * sizeof(uint8_t));

    ie_quantize_act_fp8(Vsrc, tmp, D, kv->fp8_format);
    memcpy(Vdst + off, tmp, D * sizeof(uint8_t));
  }

  free(tmp);
  return 0;
}

int ie_kv_load_token_f32(const ie_kv_cache* kv, size_t t,
                         float* K_out, float* V_out) {
  if (!kv || !K_out || !V_out) return -1;

  /* If request is out of the configured cache capacity, treat as miss. */
  if (t >= (size_t)kv->max_seq) {
    ie_kv_miss_add_local(1);
    return -2;
  }

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;

  if (kv->storage == IE_KV_STORAGE_F32) {
    const float* Ksrc = (const float*)kv->K;
    const float* Vsrc = (const float*)kv->V;
    for (size_t h = 0; h < H; ++h) {
      const size_t off = payload_index(t, h, 0, H, D);
      memcpy(K_out + h * D, Ksrc + off, D * sizeof(float));
      memcpy(V_out + h * D, Vsrc + off, D * sizeof(float));
    }
    ie_kv_hit_add_local(1);
    return 0;
  }

  if (kv->storage == IE_KV_STORAGE_INT8) {
    const int8_t* Kq = (const int8_t*)kv->K;
    const int8_t* Vq = (const int8_t*)kv->V;
    const size_t G   = kv->group_count;
    const size_t Gs  = kv->group_size;

    for (size_t h = 0; h < H; ++h) {
      const size_t off = payload_index(t, h, 0, H, D);

      /* Dequant K */
      for (size_t d = 0; d < D; ++d) {
        const size_t g  = d / Gs;
        const size_t pi = param_index(t, h, g, H, G);
        const float s   = kv->scales_K[pi];
        const int   z   = (int)kv->zeros_K[pi];
        K_out[h * D + d] = s * ((int)Kq[off + d] - z);
      }

      /* Dequant V */
      for (size_t d = 0; d < D; ++d) {
        const size_t g  = d / Gs;
        const size_t pi = param_index(t, h, g, H, G);
        const float s   = kv->scales_V[pi];
        const int   z   = (int)kv->zeros_V[pi];
        V_out[h * D + d] = s * ((int)Vq[off + d] - z);
      }
    }
    ie_kv_hit_add_local(1);
    return 0;
  }

  if (kv->storage == IE_KV_STORAGE_FP8) {
    const uint8_t* K8 = (const uint8_t*)kv->K;
    const uint8_t* V8 = (const uint8_t*)kv->V;

    for (size_t h = 0; h < H; ++h) {
      const size_t off = payload_index(t, h, 0, H, D);
      ie_dequantize_act_fp8(K8 + off, K_out + h * D, D, kv->fp8_format);
      ie_dequantize_act_fp8(V8 + off, V_out + h * D, D, kv->fp8_format);
    }
    ie_kv_hit_add_local(1);
    return 0;
  }

  /* Unknown storage: treat as miss. */
  ie_kv_miss_add_local(1);
  return -3;
}

int ie_kv_raw_ptrs(ie_kv_cache* kv, void** out_K, void** out_V, size_t* out_lds) {
  if (!kv) return -1;
  if (out_K)   *out_K   = kv->K;
  if (out_V)   *out_V   = kv->V;
  if (out_lds) *out_lds = (size_t)kv->head_dim;
  return 0;
}
