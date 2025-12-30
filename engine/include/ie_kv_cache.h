/* ============================================================================
 * File: engine/include/ie_kv_cache.h
 * ============================================================================
 */
/**
 * @file ie_kv_cache.h
 * @brief Key/Value attention cache with INT8/FP8 compression and FP32 round-trip.
 *
 * @details
 * This module provides a compact KV cache for transformer attention. The cache
 * stores per-token K/V slices laid out as [heads, head_dim] per token, flattened
 * with head_dim as the fastest-varying dimension:
 *
 *   index(t,h,d) = ((t * H + h) * D + d)
 *
 * where:
 *   - H = heads
 *   - D = head_dim
 *   - t in [0, max_seq)
 *
 * Storage types:
 *   - IE_KV_STORAGE_F32: raw FP32 K/V payloads.
 *   - IE_KV_STORAGE_INT8: INT8 payloads with per-group affine parameters
 *                         (scale, zero_point) for each (token, head).
 *   - IE_KV_STORAGE_FP8: FP8 payloads (E4M3/E5M2), no per-group parameters.
 *
 * Per-group parameters (INT8):
 *   param_index(t,h,g) = ((t * H + h) * G + g)
 * where:
 *   - group_size is provided by opts (defaults to 1 for INT8 if not provided)
 *   - G = ceil(D / group_size)
 *
 * Layered KV specification:
 *   GPT-OSS inference uses an array of caches (one per layer). To maximize TPS,
 *   this module provides ::ie_kv_init_layers() which allocates one contiguous
 *   slab for all layers and slices it into per-layer ::ie_kv_cache views.
 *
 * Performance notes:
 *   - The implementation avoids per-token heap allocations by keeping small
 *     scratch buffers inside the cache for INT8-per-group and FP8 paths.
 *   - The [t][h][d] layout makes each token slice contiguous (H*D), enabling
 *     memcpy for FP32 store/load and predictable streaming access for attention.
 */

#ifndef IE_KV_CACHE_H
#define IE_KV_CACHE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "ie_quant_act.h"

/* ------------------------------------------------------------------------- */
/* Public types                                                               */
/* ------------------------------------------------------------------------- */

/**
 * @brief KV cache storage encoding.
 */
typedef enum ie_kv_storage_type {
  /** @brief Raw FP32 payloads. */
  IE_KV_STORAGE_F32  = 0,
  /** @brief INT8 payloads + affine params (per-group). */
  IE_KV_STORAGE_INT8 = 1,
  /** @brief FP8 payloads (E4M3/E5M2) stored as uint8_t codes. */
  IE_KV_STORAGE_FP8  = 2
} ie_kv_storage_type;

/**
 * @brief Options to initialize a KV cache.
 */
typedef struct ie_kv_opts {
  /** @brief Number of heads (H). */
  int heads;
  /** @brief Head dimension (D). */
  int head_dim;
  /** @brief Maximum sequence length (T). */
  int max_seq;

  /** @brief Storage type. */
  ie_kv_storage_type storage;

  /**
   * @brief INT8 group size for per-group affine parameters.
   *
   * @details
   * Only used when storage == IE_KV_STORAGE_INT8.
   * If 0, defaults to 1.
   */
  size_t group_size;

  /**
   * @brief Use symmetric quantization for INT8.
   *
   * @details
   * Only used when storage == IE_KV_STORAGE_INT8.
   *  - 0: asymmetric affine (scale + zero_point)
   *  - 1: symmetric (zero_point is effectively 0)
   */
  int symmetric;

  /**
   * @brief FP8 format for quantization/dequantization.
   *
   * @details
   * Only used when storage == IE_KV_STORAGE_FP8.
   */
  ie_fp8_format fp8_format;
} ie_kv_opts;

/**
 * @brief A KV cache storing K/V for one transformer layer.
 *
 * @details
 * The cache owns one backing allocation (a single slab) when created via
 * ::ie_kv_init() or as the owner slice of ::ie_kv_init_layers(). Non-owner
 * slices created by ::ie_kv_init_layers() point inside the owner's backing slab
 * and must not free it.
 */
typedef struct ie_kv_cache {
  /* Geometry */
  int heads;
  int head_dim;
  int max_seq;

  /* Storage configuration */
  ie_kv_storage_type storage;
  size_t group_size;
  int symmetric;
  ie_fp8_format fp8_format;

  /* Derived sizes */
  size_t elem_count;   /**< @brief Total payload elements: H*T*D. */
  size_t group_count;  /**< @brief INT8 groups per head: ceil(D/group_size). */

  /* Payload pointers (point inside backing slab) */
  void *K; /**< @brief K payload (F32/INT8/FP8). */
  void *V; /**< @brief V payload (F32/INT8/FP8). */

  /* INT8 parameter pointers (point inside backing slab; NULL unless INT8) */
  float  *scales_K;
  int8_t *zeros_K;
  float  *scales_V;
  int8_t *zeros_V;

  /* Small scratch buffers (point inside backing slab; optional) */
  float  *scratch_scales; /**< @brief [group_count] for INT8 per-group store. */
  int8_t *scratch_zeros;  /**< @brief [group_count] for INT8 per-group store. */
  uint8_t *scratch_u8;    /**< @brief [head_dim] for FP8 store. */

  /* Ownership of the backing slab */
  void *backing;          /**< @brief Base pointer of the backing slab. */
  size_t backing_bytes;   /**< @brief Backing slab size in bytes. */
  int backing_owner;      /**< @brief 1 if this instance owns backing and must free it. */
} ie_kv_cache;

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief Return element size in bytes for a given storage encoding.
 *
 * @param storage KV storage encoding.
 * @return Element size (bytes), or 0 for unknown storage.
 */
size_t ie_kv_element_size(ie_kv_storage_type storage);

/**
 * @brief Initialize a single KV cache, allocating its backing memory.
 *
 * @param kv   Cache object to initialize.
 * @param opts Initialization options.
 * @return 0 on success; non-zero on error.
 */
int ie_kv_init(ie_kv_cache *kv, const ie_kv_opts *opts);

/**
 * @brief Initialize an array of KV caches (one per layer) using one shared slab.
 *
 * @details
 * This is the preferred initialization for transformer inference. It allocates
 * one contiguous backing slab that contains K/V payloads and any parameter/scratch
 * arrays for all layers, then slices it into @p n_layers independent cache views.
 *
 * Ownership:
 *   - kv_layers[0] is marked as the backing owner and will free the slab.
 *   - kv_layers[i>0] are non-owners; calling ::ie_kv_free() on them is safe and
 *     will not free the slab.
 *
 * @param kv_layers Array of length @p n_layers.
 * @param n_layers  Number of layers.
 * @param opts      Initialization options (applies to every layer).
 * @return 0 on success; non-zero on error.
 */
int ie_kv_init_layers(ie_kv_cache *kv_layers, int n_layers, const ie_kv_opts *opts);

/**
 * @brief Free a KV cache and release its backing memory if owned.
 *
 * @param kv Cache to free.
 */
void ie_kv_free(ie_kv_cache *kv);

/**
 * @brief Convenience to free an array of KV caches created by ::ie_kv_init_layers().
 *
 * @param kv_layers Array of KV caches.
 * @param n_layers  Number of layers.
 */
void ie_kv_free_layers(ie_kv_cache *kv_layers, int n_layers);

/**
 * @brief Store a token's K/V slice into an FP32 cache.
 *
 * @details
 * Expects @p K_f32 and @p V_f32 laid out as [heads, head_dim] (contiguous).
 *
 * @param kv    Cache (must be IE_KV_STORAGE_F32).
 * @param t     Token index.
 * @param K_f32 K slice [H*D].
 * @param V_f32 V slice [H*D].
 * @return 0 on success; non-zero on error.
 */
int ie_kv_store_token_f32(ie_kv_cache *kv, size_t t,
                          const float *K_f32, const float *V_f32);

/**
 * @brief Store a token's K/V slice into an INT8 cache using per-tensor params.
 *
 * @details
 * Per-tensor means one (scale, zero) per (token, head), replicated across all
 * groups for that (token, head) to keep dequantization consistent.
 *
 * @param kv    Cache (must be IE_KV_STORAGE_INT8).
 * @param t     Token index.
 * @param K_f32 K slice [H*D].
 * @param V_f32 V slice [H*D].
 * @return 0 on success; non-zero on error.
 */
int ie_kv_store_token_int8_per_tensor(ie_kv_cache *kv, size_t t,
                                      const float *K_f32, const float *V_f32);

/**
 * @brief Store a token's K/V slice into an INT8 cache using per-group params.
 *
 * @details
 * Per-group means one (scale, zero) per (token, head, group).
 * This path uses scratch buffers inside @p kv to avoid per-token allocations.
 *
 * @param kv    Cache (must be IE_KV_STORAGE_INT8).
 * @param t     Token index.
 * @param K_f32 K slice [H*D].
 * @param V_f32 V slice [H*D].
 * @return 0 on success; non-zero on error.
 */
int ie_kv_store_token_int8_per_group(ie_kv_cache *kv, size_t t,
                                     const float *K_f32, const float *V_f32);

/**
 * @brief Store a token's K/V slice into an FP8 cache.
 *
 * @details
 * This path uses an internal scratch buffer to avoid per-token allocations.
 *
 * @param kv    Cache (must be IE_KV_STORAGE_FP8).
 * @param t     Token index.
 * @param K_f32 K slice [H*D].
 * @param V_f32 V slice [H*D].
 * @return 0 on success; non-zero on error.
 */
int ie_kv_store_token_fp8(ie_kv_cache *kv, size_t t,
                          const float *K_f32, const float *V_f32);

/**
 * @brief Load token t as FP32 K/V slices from any storage encoding.
 *
 * @param kv    Cache.
 * @param t     Token index.
 * @param K_out Output K slice [H*D].
 * @param V_out Output V slice [H*D].
 * @return 0 on success; non-zero on error.
 */
int ie_kv_load_token_f32(const ie_kv_cache *kv, size_t t,
                         float *K_out, float *V_out);

/**
 * @brief Get raw payload pointers and leading dimension for [t][h][d] layout.
 *
 * @param kv      Cache.
 * @param out_K   Output K payload pointer (may be NULL).
 * @param out_V   Output V payload pointer (may be NULL).
 * @param out_lds Output leading dimension for the d axis (equals head_dim).
 * @return 0 on success; non-zero on error.
 */
int ie_kv_raw_ptrs(ie_kv_cache *kv, void **out_K, void **out_V, size_t *out_lds);

/**
 * @brief Get logical strides (in elements) for the [t][h][d] layout.
 *
 * @details
 * Strides are independent of storage encoding and describe the logical layout:
 *   stride_d = 1
 *   stride_h = D
 *   stride_t = H*D
 *
 * @param kv       Cache.
 * @param stride_t Output stride in elements for token dimension (may be NULL).
 * @param stride_h Output stride in elements for head dimension (may be NULL).
 * @param stride_d Output stride in elements for dim dimension (may be NULL).
 * @return 0 on success; non-zero on error.
 */
int ie_kv_raw_strides(const ie_kv_cache *kv,
                      size_t *stride_t, size_t *stride_h, size_t *stride_d);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_KV_CACHE_H */
