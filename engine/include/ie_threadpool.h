/**
 * @file ie_threadpool.h
 * @brief Minimal pthread-based thread pool API (blocking parallel_for).
 */
#ifndef IE_THREADPOOL_H_
#define IE_THREADPOOL_H_

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque thread pool type. */
typedef struct ie_threadpool ie_threadpool_t;

/**
 * @brief Range task callback type.
 *
 * The function must process indices in [begin, end).
 *
 * @param ctx   Opaque user context pointer.
 * @param begin Start index (inclusive).
 * @param end   End index (exclusive).
 */
typedef void (*ie_range_task_fn)(void *ctx, size_t begin, size_t end);

/**
 * @brief Create a thread pool with @p nthreads workers.
 *
 * Affinity is a best-effort hint effective on Linux only and only when
 * the environment variable IE_TP_USE_AFFINITY=1 is set at runtime.
 *
 * @param nthreads Number of threads (>=1). 0 = auto-detect (clamped to >=1).
 * @param affinity Hint string: "auto", "compact", or "scatter".
 * @return Pointer to a new thread pool or NULL on failure.
 */
ie_threadpool_t* ie_tp_create(unsigned nthreads, const char *affinity);

/**
 * @brief Destroy a thread pool and join all workers.
 *
 * @param tp Thread pool handle (NULL allowed; no-op).
 */
void ie_tp_destroy(ie_threadpool_t *tp);

/**
 * @brief Execute a blocking parallel-for over [0, n).
 *
 * The current baseline performs a deterministic contiguous partition across
 * threads; @p grainsize is informational only for now.
 *
 * @param tp        Thread pool (may be NULL => single-thread).
 * @param n         Total range length.
 * @param fn        User callback (must not be NULL).
 * @param ctx       Opaque user context.
 * @param grainsize Informational grainsize hint (currently unused).
 * @return true on success; false on invalid arguments.
 */
bool ie_tp_parallel_for(ie_threadpool_t *tp,
                        size_t n,
                        ie_range_task_fn fn,
                        void *ctx,
                        size_t grainsize);

#ifdef __cplusplus
}
#endif

#endif /* IE_THREADPOOL_H_ */
