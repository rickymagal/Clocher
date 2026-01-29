/**
 * @file ie_threadpool.h
 * @brief Minimal thread-pool interface with optional Linux CPU affinity.
 *
 * This API intentionally keeps the contract small:
 *  - Create / destroy a pool.
 *  - Run a synchronous parallel-for with contiguous partitioning.
 *
 * Key properties:
 *  - `ie_tp_parallel_for()` is synchronous; it returns only after all work is complete.
 *  - If `tp == NULL` or the pool has only one worker, it runs single-threaded.
 *  - Optional CPU affinity is applied only on Linux when `IE_TP_USE_AFFINITY=1`.
 *
 * Thread-safety:
 *  - The pool is safe to use from one caller at a time.
 *  - `ie_tp_parallel_for()` is not re-entrant and must not be called concurrently
 *    on the same pool (including from within `body_fn`).
 */

#ifndef IE_THREADPOOL_H_
#define IE_THREADPOOL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/** @brief Opaque thread-pool handle. */
typedef struct ie_threadpool ie_threadpool_t;

/**
 * @brief Create a thread pool.
 *
 * On success, creates up to `nth` long-lived worker threads. If thread creation
 * fails partway, the pool is still created with fewer workers; callers should
 * expect degraded performance but correct behavior.
 *
 * @param nth      Desired number of worker threads (>= 1). Values < 1 are clamped to 1.
 * @param affinity Optional policy string: "auto", "compact", or "scatter".
 *                 The value is advisory; non-Linux builds ignore affinity.
 *                 The string is copied internally.
 * @return Newly created thread-pool handle, or NULL on allocation failure.
 */
ie_threadpool_t* ie_tp_create(unsigned nth, const char *affinity);

/**
 * @brief Destroy a thread pool and join all worker threads.
 *
 * @param tp Thread-pool handle (NULL allowed; no-op).
 */
void ie_tp_destroy(ie_threadpool_t *tp);

/**
 * @brief Function type for a parallel-for body over a half-open range [start, end).
 *
 * The implementation guarantees that for each issued subrange:
 *  - `start < end` for non-empty subranges,
 *  - and the union of all subranges exactly covers `[0, n)` with no overlaps.
 *
 * @param ctx   User context pointer passed through from `ie_tp_parallel_for()`.
 * @param start Inclusive start index for this worker.
 * @param end   Exclusive end index for this worker.
 */
typedef void (*ie_tp_for_body_fn)(void *ctx, unsigned start, unsigned end);

/**
 * @brief Execute a synchronous parallel-for over `n` iterations.
 *
 * The iteration space `[0, n)` is partitioned into contiguous chunks. If
 * `grainsize > 0`, each chunk will be at least `grainsize` iterations (except
 * possibly the last). If `grainsize == 0`, the function partitions the space
 * as evenly as possible across the pool's workers.
 *
 * On non-Linux systems (or when `IE_TP_USE_AFFINITY != 1`), no CPU affinity
 * is applied. On Linux with `IE_TP_USE_AFFINITY=1`, worker threads may bind
 * to specific CPUs using the policy provided at creation ("auto"/"compact"/"scatter").
 *
 * @param tp        Thread-pool handle (may be NULL for single-threaded).
 * @param n         Number of iterations in the global range `[0, n)`.
 * @param grainsize Minimum chunk size (use 0 for auto).
 * @param body_fn   Function called for each chunk.
 * @param ctx       Opaque user pointer passed to `body_fn`.
 */
void ie_tp_parallel_for(ie_threadpool_t *tp,
                        unsigned n,
                        unsigned grainsize,
                        ie_tp_for_body_fn body_fn,
                        void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* IE_THREADPOOL_H_ */
