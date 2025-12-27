/**
 * @file ie_metrics.h
 * @brief Public metrics struct and helpers for runtime reporting.
 *
 * @details
 * Metrics are intentionally small and stable so that the CLI can emit them as
 * JSON and the benchmark harness can aggregate results.
 *
 * This header defines:
 *  - @ref ie_metrics_t : snapshot struct for KV counters and RSS peak.
 *  - @ref ie_metrics_reset / add / snapshot : process-wide counters.
 *  - @ref ie_metrics_sample_rss_peak : best-effort RSS peak sampler (MiB).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Metrics snapshot structure.
 *
 * All fields are monotonically increasing counters except rss_peak_mb
 * which is a sampled value (MiB).
 */
struct ie_metrics {
  /** @brief KV cache hits since last reset/snapshot. */
  uint64_t kv_hits;

  /** @brief KV cache misses since last reset/snapshot. */
  uint64_t kv_misses;

  /** @brief Peak RSS in MiB (best-effort). */
  uint32_t rss_peak_mb;
};

/** @brief Alias for @ref ie_metrics. */
typedef struct ie_metrics ie_metrics_t;

/**
 * @brief Reset all process-wide counters (KV hits/misses).
 */
void ie_metrics_reset(void);

/**
 * @brief Add @p n KV hits to the global counter.
 * @param n Number of hits to add.
 */
void ie_metrics_add_kv_hit(uint64_t n);

/**
 * @brief Add @p n KV misses to the global counter.
 * @param n Number of misses to add.
 */
void ie_metrics_add_kv_miss(uint64_t n);

/**
 * @brief Add both hits and misses to the global counters.
 * @param hits Hits to add.
 * @param misses Misses to add.
 */
void ie_metrics_add_kv(uint64_t hits, uint64_t misses);

/**
 * @brief Snapshot KV counters into @p out.
 *
 * @param out Destination snapshot (required).
 * @param reset_after If nonzero, reset counters after snapshot.
 */
void ie_metrics_snapshot(ie_metrics_t *out, int reset_after);

/**
 * @brief Sample best-effort process peak RSS in MiB.
 * @return Peak RSS in MiB, or 0 if unavailable.
 */
uint32_t ie_metrics_sample_rss_peak(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
