/**
 * @file util_metrics.h
 * @brief Public helpers for runtime metrics (KV counters + RSS sampling).
 *
 * This header complements the engine metrics exposed by `ie_api.h`.
 * It provides:
 *  - Process-wide KV hit/miss accumulators using C11 atomics.
 *  - A best-effort function to retrieve the process peak RSS (in MiB).
 *
 * Design:
 *  - We intentionally include `ie_api.h` (and NOT `ie_metrics.h`) so the
 *    project uses the canonical `ie_metrics_t` owned by the engine API.
 */

#ifndef UTIL_METRICS_H
#define UTIL_METRICS_H

#include <stdint.h>
#include "ie_api.h" /* for ie_metrics_t */

/**
 * @defgroup IE_METRICS Utilities: KV counters and RSS sampler
 * @{
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reset process-wide KV counters to zero.
 *
 * Thread-safe (uses relaxed atomics). Intended for tests or to delimit rounds.
 */
void ie_metrics_reset(void);

/**
 * @brief Add @p n KV hits to the process-wide counter.
 * @param n Number of hits to add (no-op if 0).
 */
void ie_metrics_add_kv_hit(uint64_t n);

/**
 * @brief Add @p n KV misses to the process-wide counter.
 * @param n Number of misses to add (no-op if 0).
 */
void ie_metrics_add_kv_miss(uint64_t n);

/**
 * @brief Add both hits and misses in one call (each may be zero).
 * @param hits   Hits to add.
 * @param misses Misses to add.
 */
void ie_metrics_add_kv(uint64_t hits, uint64_t misses);

/**
 * @brief Snapshot process-wide KV counters into @p out.
 *
 * @param out         Destination snapshot (must be non-NULL).
 * @param reset_after If nonzero, counters are reset after reading.
 *
 * Only KV fields inside @p out are written by this function. Other fields
 * of @p out are left untouched (so callers can compose metrics).
 */
void ie_metrics_snapshot(ie_metrics_t *out, int reset_after);

/**
 * @brief Best-effort sampling of process peak RSS, in MiB.
 *
 * Platform behavior:
 *  - Linux: tries `/proc/self/status` `VmHWM:` (kB), falls back to `VmRSS:`
 *    (kB), then `/proc/self/smaps_rollup` `Rss:` (kB), and finally
 *    `getrusage()` (kB). kB are rounded up to MiB.
 *  - macOS: uses `mach_task_basic_info` (bytes) and falls back to
 *    `getrusage()` (bytes). Bytes are rounded up to MiB.
 *  - Other OS: falls back to `getrusage()` and conservatively treats the
 *    value as kB.
 *
 * Debugging:
 *  - Set `IE_DEBUG_RSS=1` to print sampler decisions to stderr.
 *
 * @return Peak RSS in MiB (0 if unavailable or reported < 1 MiB).
 */
uint32_t ie_metrics_sample_rss_peak(void);

#ifdef __cplusplus
}
#endif

/** @} */ /* end of group IE_METRICS */

#endif /* UTIL_METRICS_H */

