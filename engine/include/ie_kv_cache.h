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
 * ## Memory layout
 * Let:
 *   H = heads, T = max_seq, D = head_dim, G = ceil(D / group_size).
 *
 * The element layout for K and V is the same across all storage kinds:
 *   index(t,h,d) = ((t * H + h) * D + d)
 * Fastest dimension is head_dim (D), then head, then time.
 *
 * For INT8 storage, per-group parameters (scale, zero_point) are tracked
 * *per token and per head* for both K and V:
 *   param_index(t,h,g) = ((t * H + h) * G + g)
 *
 * Thread-safety: the cache object is not thread-safe for concurrent writers.
 * Readers may run concurrently if tokens are not being mutated.
 */

#include <stddef.h>
#include <stdint.h>
#include "ie_quant_act.h"  /* ie_fp8_format, ie_act_i8_params and helpers */

#ifdef __cplusplus
extern "C" {
#endif

/** @enum ie_kv_storage_type
 *  @brief Backing element format for K/V payloads.
 */
typedef enum ie_kv_storage_type {
  IE_KV_STORAGE_F32 = 0,      /**< Store K/V as FP32. */
  IE_KV_STORAGE_INT8 = 1,     /**< Store K/V as INT8 with affine de/quant. */
  IE_KV_STORAGE_FP8  = 2      /**< Store K/V as FP8 (E4M3 or E5M2), no scale arrays. */
} ie_kv_storage_type;

/** @struct ie_kv_opts
 *  @brief Initialization options for a KV cache instance.
 */
typedef struct ie_kv_opts {
  int heads;                  /**< Number of attention heads (H > 0). */
  int head_dim;               /**< Dimension per head (D > 0). */
  int max_seq;                /**< Maximum sequence length to store (T > 0). */
  ie_kv_storage_type storage; /**< Backing storage format. */
  size_t group_size;          /**< INT8 group size along D (>=1). Ignored if not INT8. */
  int symmetric;              /**< INT8 symmetric mode flag (zero_point = 0). */
  ie_fp8_format fp8_format;   /**< FP8 format (E4M3/E5M2). Used if storage == FP8. */
} ie_kv_opts;

/** @struct ie_kv_cache
 *  @brief Opaque-ish KV cache handle with preallocated buffers and metadata.
 *
 * Fields are exposed for introspection; callers should treat them as read-only
 * except via the API functions provided in this header.
 */
typedef struct ie_kv_cache {
  /* Dimensions */
  int heads;
  int head_dim;
  int max_seq;

  /* Storage policy */
  ie_kv_storage_type storage;
  size_t group_size;
  int symmetric;
  ie_fp8_format fp8_format;

  /* Derived geometry */
  size_t elem_count;   /**< Total elements per tensor: (size_t)heads*max_seq*head_dim. */
  size_t group_count;  /**< Groups per (token, head): ceil(head_dim / group_size). */

  /* Payload buffers (ownership belongs to this object) */
  void* K;             /**< K payload base pointer (element type depends on storage). */
  void* V;             /**< V payload base pointer (element type depends on storage). */

  /* INT8 per-group parameters for K and V (NULL unless storage == INT8). */
  float*  scales_K;    /**< length = T * H * G */
  int8_t* zeros_K;     /**< length = T * H * G */
  float*  scales_V;    /**< length = T * H * G */
  int8_t* zeros_V;     /**< length = T * H * G */
} ie_kv_cache;

/**
 * @brief Initialize a KV cache with the given options and allocate backing memory.
 *
 * On success, the cache owns all internal buffers and must be released with
 * ::ie_kv_free(). All buffers are zero-initialized.
 *
 * @param kv    Pointer to an uninitialized cache handle (output).
 * @param opts  Options controlling shape and storage policy.
 * @return 0 on success; non-zero on bad arguments or allocation failure.
 */
int ie_kv_init(ie_kv_cache* kv, const ie_kv_opts* opts);

/**
 * @brief Free all resources owned by a KV cache.
 *
 * It is safe to call with a NULL pointer or with a partially initialized handle.
 *
 * @param kv  Cache handle to be destroyed.
 */
void ie_kv_free(ie_kv_cache* kv);

/**
 * @brief Return the element size (in bytes) of the configured storage.
 *
 * @param storage  Storage kind.
 * @return Size in bytes of one element in K/V buffers.
 */
size_t ie_kv_element_size(ie_kv_storage_type storage);

/**
 * @brief Store one token’s worth of K/V rows provided in FP32 into the cache.
 *
 * The input slices must be contiguous and laid out as [heads, head_dim].
 * This function requires `storage == IE_KV_STORAGE_F32`.
 *
 * @param kv       Cache handle.
 * @param t        Token index in [0, max_seq).
 * @param K_f32    Pointer to input K slice (length heads*head_dim).
 * @param V_f32    Pointer to input V slice (length heads*head_dim).
 * @return 0 on success; non-zero on errors or type mismatch.
 */
int ie_kv_store_token_f32(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32);

/**
 * @brief Store one token’s K/V using INT8 per-tensor parameters (one scale/zp per head).
 *
 * For each head, parameters are derived from the K (or V) slice’s min/max and
 * recorded with the payload. Requires `storage == IE_KV_STORAGE_INT8` and
 * uses `kv->symmetric` to choose symmetric/asymmetric mapping.
 *
 * @param kv      Cache handle.
 * @param t       Token index in [0, max_seq).
 * @param K_f32   Input K slice (heads*head_dim).
 * @param V_f32   Input V slice (heads*head_dim).
 * @return 0 on success; non-zero on errors or type mismatch.
 */
int ie_kv_store_token_int8_per_tensor(ie_kv_cache* kv, size_t t,
                                      const float* K_f32, const float* V_f32);

/**
 * @brief Store one token’s K/V using INT8 per-group parameters along head_dim.
 *
 * Groups have size `kv->group_size`; parameters are computed and stored for each
 * (token, head, group). Requires `storage == IE_KV_STORAGE_INT8`.
 *
 * @param kv      Cache handle.
 * @param t       Token index in [0, max_seq).
 * @param K_f32   Input K slice (heads*head_dim).
 * @param V_f32   Input V slice (heads*head_dim).
 * @return 0 on success; non-zero on errors or type mismatch.
 */
int ie_kv_store_token_int8_per_group(ie_kv_cache* kv, size_t t,
                                     const float* K_f32, const float* V_f32);

/**
 * @brief Store one token’s K/V as FP8 (E4M3/E5M2), no scale arrays.
 *
 * Uses `kv->fp8_format` for encoding. Requires `storage == IE_KV_STORAGE_FP8`.
 *
 * @param kv      Cache handle.
 * @param t       Token index in [0, max_seq).
 * @param K_f32   Input K slice (heads*head_dim).
 * @param V_f32   Input V slice (heads*head_dim).
 * @return 0 on success; non-zero on errors or type mismatch.
 */
int ie_kv_store_token_fp8(ie_kv_cache* kv, size_t t,
                          const float* K_f32, const float* V_f32);

/**
 * @brief Load one token’s K/V as FP32, dequantizing/decoding if needed.
 *
 * The output buffers must have length `heads*head_dim`. Works with any
 * storage kind and performs the appropriate conversion on the fly.
 *
 * @param kv      Cache handle.
 * @param t       Token index in [0, max_seq).
 * @param K_out   Output buffer for K (FP32).
 * @param V_out   Output buffer for V (FP32).
 * @return 0 on success; non-zero on errors or bounds/type mismatch.
 */
int ie_kv_load_token_f32(const ie_kv_cache* kv, size_t t,
                         float* K_out, float* V_out);

/**
 * @brief Return raw pointers to the internal K/V payloads and the logical stride.
 *
 * This is a convenience for custom kernels that want to access the backing
 * arrays directly. The logical leading dimension along D is `head_dim`.
 *
 * @param kv          Cache handle.
 * @param out_K       Output pointer to the base of K buffer (may be NULL).
 * @param out_V       Output pointer to the base of V buffer (may be NULL).
 * @param out_lds     Output: leading dimension (equals `kv->head_dim`).
 * @return 0 on success; non-zero on bad arguments.
 */
int ie_kv_raw_ptrs(ie_kv_cache* kv, void** out_K, void** out_V, size_t* out_lds);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_KV_CACHE_H */
