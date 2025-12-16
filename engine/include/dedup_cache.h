#ifndef IE_DEDUP_CACHE_H
#define IE_DEDUP_CACHE_H

/**
 * @file dedup_cache.h
 * @brief Runtime cache and reconstruction helpers for lossless deduplicated weights.
 *
 * This module provides:
 * - Fast lookup: map a tensor_index -> (group, target) metadata.
 * - Lossless reconstruction: patch a default blob using mask + exceptions.
 * - Optional caching: keep reconstructed tensors in a fixed-capacity cache (LRU-ish).
 *
 * The API is byte-oriented and intentionally does not assume a specific tensor dtype.
 */

#include <stdint.h>
#include <stddef.h>

#include "dedup_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct ie_dedup_files_t
 * @brief Memory views of dedup artifact files (mmap'd or fully loaded).
 */
typedef struct ie_dedup_files_s {
  /** @brief Pointer to defaults.bin contents. */
  const uint8_t* defaults;

  /** @brief defaults.bin size in bytes. */
  uint64_t defaults_size;

  /** @brief Pointer to masks.bin contents. */
  const uint8_t* masks;

  /** @brief masks.bin size in bytes. */
  uint64_t masks_size;

  /** @brief Pointer to exceptions.bin contents. */
  const uint8_t* exceptions;

  /** @brief exceptions.bin size in bytes. */
  uint64_t exceptions_size;
} ie_dedup_files_t;

/**
 * @struct ie_dedup_cache_entry_t
 * @brief One cached reconstructed tensor blob.
 */
typedef struct ie_dedup_cache_entry_s {
  /** @brief Tensor index this entry corresponds to. */
  uint32_t tensor_index;

  /** @brief Size in bytes. */
  uint64_t nbytes;

  /** @brief Pointer to owned storage holding reconstructed bytes. */
  uint8_t* data;

  /** @brief Monotonic counter for simple eviction policy. */
  uint64_t stamp;
} ie_dedup_cache_entry_t;

/**
 * @struct ie_dedup_cache_t
 * @brief Dedup reconstruction cache.
 *
 * The cache is optional: you can call ie_dedup_reconstruct_into() directly if you
 * want on-demand reconstruction into a caller-provided buffer.
 */
typedef struct ie_dedup_cache_s {
  /** @brief Spec describing groups/targets. */
  const ie_dedup_spec_t* spec;

  /** @brief File views for defaults/masks/exceptions. */
  ie_dedup_files_t files;

  /** @brief Lookup table: tensor_index -> flat target pointer (or NULL). */
  const ie_dedup_target_t** target_by_index;

  /** @brief Size of target_by_index array (max tensor_index + 1). */
  uint32_t target_by_index_len;

  /** @brief Default tensor index -> group pointer lookup (optional fast path). */
  const ie_dedup_group_t** group_by_default_index;

  /** @brief Size of group_by_default_index array. */
  uint32_t group_by_default_index_len;

  /** @brief Cache entries (fixed capacity). */
  ie_dedup_cache_entry_t* entries;

  /** @brief Number of entries allocated. */
  uint32_t entry_cap;

  /** @brief Monotonic stamp counter. */
  uint64_t stamp_now;

  /** @brief Total bytes allocated for cached tensor buffers. */
  uint64_t bytes_cached;

  /** @brief Soft byte limit for cached buffers (0 disables caching). */
  uint64_t bytes_limit;
} ie_dedup_cache_t;

/**
 * @brief Initialize a dedup cache.
 *
 * This builds fast lookup tables for tensor_index -> target metadata.
 * If bytes_limit is 0, the cache storage is not used (but the lookup and reconstruction helpers still work).
 *
 * @param c Cache instance.
 * @param spec Parsed and validated dedup spec.
 * @param files Memory views of defaults/masks/exceptions (must stay valid for cache lifetime).
 * @param max_tensor_index Maximum tensor index in the model (used to size lookup tables).
 * @param entry_cap Number of cache slots to allocate (used only if bytes_limit > 0).
 * @param bytes_limit Soft cap on cached reconstructed bytes. Set to 0 to disable caching.
 * @return 0 on success, non-zero on error.
 */
int ie_dedup_cache_init(
  ie_dedup_cache_t* c,
  const ie_dedup_spec_t* spec,
  const ie_dedup_files_t* files,
  uint32_t max_tensor_index,
  uint32_t entry_cap,
  uint64_t bytes_limit);

/**
 * @brief Destroy a dedup cache and free any owned memory.
 * @param c Cache instance.
 */
void ie_dedup_cache_destroy(ie_dedup_cache_t* c);

/**
 * @brief Find a target record for a given tensor_index.
 * @param c Cache instance.
 * @param tensor_index Target tensor index.
 * @return Pointer to target metadata, or NULL if the tensor is not deduplicated.
 */
const ie_dedup_target_t* ie_dedup_cache_find_target(const ie_dedup_cache_t* c, uint32_t tensor_index);

/**
 * @brief Find the group that owns a given target record.
 *
 * @param c Cache instance.
 * @param t Target metadata pointer (must come from this cache/spec).
 * @return Pointer to owning group, or NULL if not found (should not happen if spec is valid).
 */
const ie_dedup_group_t* ie_dedup_cache_find_group_for_target(const ie_dedup_cache_t* c, const ie_dedup_target_t* t);

/**
 * @brief Reconstruct a target tensor into a caller-provided buffer (lossless).
 *
 * This performs:
 * - memcpy(default -> dst)
 * - scan mask bits; for each 1-bit at byte index i, overwrite dst[i] from exceptions stream
 *
 * @param dst Destination buffer (size must be t->nbytes).
 * @param default_bytes Pointer to the default blob bytes (size must be t->nbytes).
 * @param mask Pointer to mask bytes (size must be t->mask_nbytes).
 * @param exceptions Pointer to exception payload bytes (size must be t->exc_nbytes).
 * @param nbytes Tensor size in bytes.
 * @return 0 on success, non-zero on mismatch (e.g., ran out of exceptions).
 */
int ie_dedup_reconstruct_into(
  uint8_t* dst,
  const uint8_t* default_bytes,
  const uint8_t* mask,
  const uint8_t* exceptions,
  uint64_t nbytes);

/**
 * @brief Get a reconstructed weight blob for a tensor.
 *
 * If caching is enabled (bytes_limit > 0), this may return a pointer to cached storage.
 * Otherwise, this reconstructs into the provided scratch buffer.
 *
 * @param c Cache instance.
 * @param tensor_index Target tensor index.
 * @param scratch Scratch buffer (required if caching disabled, optional otherwise).
 * @param scratch_cap Scratch capacity in bytes.
 * @param out_nbytes Output tensor size in bytes.
 * @return Pointer to reconstructed bytes (either cache-owned or scratch), or NULL on error.
 */
const uint8_t* ie_dedup_cache_get_bytes(
  ie_dedup_cache_t* c,
  uint32_t tensor_index,
  uint8_t* scratch,
  uint64_t scratch_cap,
  uint64_t* out_nbytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEDUP_CACHE_H */

