/**
 * @file util_metrics.h
 * @brief Small runtime metrics API used by the CLI and benchmark harness.
 *
 * @details
 * This module provides lightweight, process-wide counters and RSS sampling.
 * The concrete snapshot structure is defined in @ref ie_metrics.h.
 */

#pragma once

#include "ie_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reset all process-wide metrics counters.
 */
void ie_metrics_reset(void);

/**
 * @brief Add KV cache hits to the process-wide counter.
 * @param n Hits to add.
 */
void ie_metrics_add_kv_hit(uint64_t n);

/**
 * @brief Add KV cache misses to the process-wide counter.
 * @param n Misses to add.
 */
void ie_metrics_add_kv_miss(uint64_t n);

/**
 * @brief Add KV cache hits and misses to the process-wide counters.
 * @param hits Hits to add.
 * @param misses Misses to add.
 */
void ie_metrics_add_kv(uint64_t hits, uint64_t misses);

/**
 * @brief Snapshot KV counters into @p out.
 * @param out Destination snapshot.
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
