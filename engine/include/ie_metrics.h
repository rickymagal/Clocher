/**
 * @file ie_metrics.h
 * @brief Public API for runtime metrics (KV cache and memory/RSS sampling).
 *
 * @details
 * This header exposes:
 *  - Cheap, thread-safe KV cache counters (hits/misses).
 *  - A portable best-effort function to sample peak RSS (resident set size).
 *
 * Typical usage:
 * - Call ::ie_metrics_reset at the start of a benchmark round.
 * - In hot paths, call ::IE_KV_HIT / ::IE_KV_MISS.
 * - At the end, call ::ie_metrics_snapshot(reset_after=1).
 * - Optionally call ::ie_metrics_sample_rss_peak() once per run/round to
 *   populate RSS numbers for reporting.
 */

#ifndef IE_METRICS_H
#define IE_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @brief Snapshot of engine metrics for a measurement interval.
 *
 * All fields are monotonic counters accumulated during an interval
 * (typically one benchmark/inference round).
 */
typedef struct ie_metrics_t {
  uint64_t kv_hits;   /**< Total KV cache hits observed in the interval. */
  uint64_t kv_misses; /**< Total KV cache misses observed in the interval. */
  uint64_t rsvd0;     /**< Reserved for future expansion. */
} ie_metrics_t;

/**
 * @brief Reset all metrics accumulators to zero.
 *
 * @post Subsequent snapshots will report zeros until producers add counts.
 *
 * @thread_safety Safe; uses C11 atomics.
 */
void ie_metrics_reset(void);

/**
 * @brief Atomically add to the KV hit counter.
 *
 * @param n Number of hits to add (may be zero).
 */
void ie_metrics_add_kv_hit(uint64_t n);

/**
 * @brief Atomically add to the KV miss counter.
 *
 * @param n Number of misses to add (may be zero).
 */
void ie_metrics_add_kv_miss(uint64_t n);

/**
 * @brief Atomically add to both KV counters.
 *
 * @param hits Hits to add.
 * @param misses Misses to add.
 */
void ie_metrics_add_kv(uint64_t hits, uint64_t misses);

/**
 * @brief Copy current counters into @p out and optionally reset them.
 *
 * @param[out] out          Destination snapshot structure (must not be NULL).
 * @param[in]  reset_after  If non-zero, zeroes accumulators after copying.
 */
void ie_metrics_snapshot(ie_metrics_t* out, int reset_after);

/**
 * @def IE_KV_HIT
 * @brief Cheap inline increment of the KV hit counter.
 *
 * @param n Number of hits to add.
 */
#define IE_KV_HIT(n)  ie_metrics_add_kv_hit((uint64_t)(n))

/**
 * @def IE_KV_MISS
 * @brief Cheap inline increment of the KV miss counter.
 *
 * @param n Number of misses to add.
 */
#define IE_KV_MISS(n) ie_metrics_add_kv_miss((uint64_t)(n))

/**
 * @brief Best-effort sampling of peak RSS (resident set size).
 *
 * @details
 * On Linux, prefers reading `/proc/self/status` (VmHWM). If unavailable,
 * falls back to `getrusage(RUSAGE_SELF).ru_maxrss`. On non-Linux, only the
 * `getrusage` fallback is used where available. Returns 0 if unsupported.
 *
 * @return Peak RSS in megabytes (MB), rounded down; 0 if unknown.
 */
uint32_t ie_metrics_sample_rss_peak(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_METRICS_H */
