#ifndef IE_DEDUP_CACHE_H
#define IE_DEDUP_CACHE_H

/**
 * @file dedup_cache.h
 * @brief In-memory cache for reconstructed (materialized) tensors.
 *
 * @details
 * This cache exists to convert the dedup storage format (defaults + mask +
 * exceptions) into a stable, directly usable byte buffer.
 *
 * Key requirements for real hits:
 *  - Cache keys must be stable (e.g. tensor name or tensor index).
 *  - The engine must request the same key repeatedly across tokens/steps.
 *  - Cached pointers must remain valid for the duration of the run.
 *
 * This is a generic cache: it does not know about NUMA replication; callers can
 * replicate "hot" cached blobs separately when needed.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque cache type. */
typedef struct ie_dedup_cache ie_dedup_cache_t;

/**
 * @struct ie_dedup_cache_opts_t
 * @brief Cache configuration options.
 */
typedef struct ie_dedup_cache_opts_t {
  /**
   * Maximum bytes the cache may retain.
   * If 0, the cache behaves like "no caching" (always misses).
   */
  size_t bytes_limit;

  /**
   * If non-zero, enables basic internal validation (bounds checks).
   * Keep this enabled for now; disable only once stable.
   */
  int enable_safety_checks;
} ie_dedup_cache_opts_t;

/**
 * @brief Create a dedup cache instance.
 *
 * @param opts Options (may be NULL for defaults).
 * @return New cache pointer, or NULL on OOM.
 */
ie_dedup_cache_t* ie_dedup_cache_create(const ie_dedup_cache_opts_t* opts);

/**
 * @brief Destroy a cache and release all retained buffers.
 *
 * @param c Cache (NULL allowed).
 */
void ie_dedup_cache_destroy(ie_dedup_cache_t* c);

/**
 * @brief Current total bytes retained in the cache.
 *
 * @param c Cache (non-NULL).
 * @return Retained bytes.
 */
size_t ie_dedup_cache_bytes_used(const ie_dedup_cache_t* c);

/**
 * @brief Maximum bytes allowed for this cache.
 *
 * @param c Cache (non-NULL).
 * @return Configured byte limit.
 */
size_t ie_dedup_cache_bytes_limit(const ie_dedup_cache_t* c);

/**
 * @brief Evict everything from the cache.
 *
 * @param c Cache (non-NULL).
 */
void ie_dedup_cache_clear(ie_dedup_cache_t* c);

/**
 * @brief Lookup a cached tensor by key.
 *
 * @details
 * The returned pointer is owned by the cache and remains valid until the entry
 * is evicted or the cache is destroyed.
 *
 * @param c Cache (non-NULL).
 * @param key Tensor key (stable string).
 * @param out_ptr Output pointer to cached bytes.
 * @param out_size Output size in bytes.
 * @return 1 if hit, 0 if miss.
 */
int ie_dedup_cache_get(const ie_dedup_cache_t* c,
                       const char* key,
                       const void** out_ptr,
                       size_t* out_size);

/**
 * @brief Insert (or replace) an entry in the cache.
 *
 * @details
 * The cache makes a private copy of @p data.
 *
 * @param c Cache (non-NULL).
 * @param key Tensor key (stable string).
 * @param data Bytes to copy into the cache.
 * @param size Number of bytes.
 * @return 0 on success, negative errno-like value on failure.
 */
int ie_dedup_cache_put(ie_dedup_cache_t* c,
                       const char* key,
                       const void* data,
                       size_t size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEDUP_CACHE_H */

