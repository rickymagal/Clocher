/**
 * @file ie_threadpool.h
 * @brief Minimal pthread-based thread pool with a blocking parallel_for API.
 *
 * The pool implements a deterministic split of an index range [0, n)
 * across workers using a fixed grain size. Affinity is best-effort on Linux
 * (compact or scatter); on other OSes the hints are ignored.
 */

#ifndef IE_THREADPOOL_H
#define IE_THREADPOOL_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque thread pool handle.
 *
 * Instances are created with ::ie_tp_create and destroyed with ::ie_tp_destroy.
 */
typedef struct ie_threadpool ie_threadpool_t;

/**
 * @brief Function signature for processing a half-open range [begin, end).
 *
 * The implementation must be side-effect free with respect to other shards
 * of the same task (i.e., parallel-safe) and should not block indefinitely.
 *
 * @param[in,out] ctx   User context pointer supplied to ::ie_tp_parallel_for.
 * @param[in]     begin First index in the range (inclusive).
 * @param[in]     end   One past the last index (exclusive).
 */
typedef void (*ie_range_task_fn)(void *ctx, size_t begin, size_t end);

/**
 * @brief Create a new thread pool.
 *
 * @param[in] nthreads  Number of worker threads. Use `0` for "auto" (nproc).
 * @param[in] affinity  Affinity hint: `"auto"`, `"compact"`, or `"scatter"`.
 *                      Non-Linux platforms ignore this hint.
 * @return A pointer to the pool, or `NULL` on allocation failure.
 */
ie_threadpool_t* ie_tp_create(unsigned nthreads, const char *affinity);

/**
 * @brief Destroy a previously created thread pool.
 *
 * Waits for all workers to exit before returning. Passing `NULL` is a no-op.
 *
 * @param[in,out] tp  Pool handle (may be NULL).
 */
void ie_tp_destroy(ie_threadpool_t *tp);

/**
 * @brief Execute a parallel for over the range [0, n).
 *
 * The range is split across the pool's workers using @p grainsize
 * (or an auto value if @p grainsize is `0`). If @p tp is `NULL` or the range
 * is trivial, the function runs the task single-threaded.
 *
 * @param[in,out] tp        Pool handle (NULL for single-thread fallback).
 * @param[in]     n         Number of elements in the range.
 * @param[in]     fn        Task callback to execute on each shard.
 * @param[in,out] ctx       User context passed to @p fn.
 * @param[in]     grainsize Minimum shard size (0 = auto).
 * @return `true` on success, `false` if @p fn is NULL.
 */
bool ie_tp_parallel_for(ie_threadpool_t *tp,
                        size_t n,
                        ie_range_task_fn fn,
                        void *ctx,
                        size_t grainsize);

#ifdef __cplusplus
}
#endif

#endif /* IE_THREADPOOL_H */
