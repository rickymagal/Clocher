/**
 * @file util_metrics.h
 * @brief Lightweight, dependency-free runtime metrics helpers.
 *
 * This header exposes small helpers for collecting **host-observable**
 * signals such as peak resident set size (RSS) and simple KV hit/miss
 * accounting plumbing. It intentionally keeps the surface area small and
 * ABI-stable.
 *
 * Design notes:
 * - The canonical metrics structure is `ie_metrics_t` from @ref ie_api.h.
 *   This header only *updates* that struct; it does not redefine it.
 * - On non-Linux platforms, the `/proc`-based functions become no-ops
 *   that return success and leave outputs as zero.
 * - No dynamic allocation; all updates happen on caller-owned storage.
 */

#ifndef UTIL_METRICS_H
#define UTIL_METRICS_H

/* Single, portable include. */
#include <stddef.h>          /* size_t */
#include <stdint.h>          /* uint64_t */
#include "ie_api.h"          /* ie_metrics_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read `VmHWM` (high-water mark of RSS) and `VmRSS` from `/proc/self/status`.
 *
 * Linux-only implementation: parses `/proc/self/status` to extract both values.
 * If a pointer is `NULL`, that output is skipped. Units returned are **KiB**.
 *
 * On non-Linux builds, this function returns 0 and sets outputs to 0.
 *
 * @param[out] VmHWM_kb  Optional output for peak RSS (KiB).
 * @param[out] VmRSS_kb  Optional output for current RSS (KiB).
 * @return 0 on success; negative on error (e.g., file missing/unreadable).
 */
int ie_metrics_read_proc_status_kb(size_t *VmHWM_kb, size_t *VmRSS_kb);

/**
 * @brief Sample and update the `rss_peak_mb` field of an @ref ie_metrics_t.
 *
 * This function is **cheap** and **non-intrusive**. It never walks process
 * mappings or touches model pages; it only parses `/proc/self/status` and
 * converts `VmHWM` (KiB) to MiB.
 *
 * @param m Pointer to a valid @ref ie_metrics_t.
 * @return 0 on success; negative on error (invalid pointer or parse error).
 */
int ie_metrics_sample_rss_peak(ie_metrics_t *m);

/**
 * @brief Reset the KV counters (hits, misses) inside an @ref ie_metrics_t.
 *
 * This is a *plumbing* helper for engines that wish to expose KV stats via
 * the public metrics object without forcing the CLI to maintain separate
 * counters. It does **not** perform any memory “touching” or probing.
 *
 * @param m Pointer to a valid @ref ie_metrics_t.
 */
void ie_metrics_kv_reset(ie_metrics_t *m);

/**
 * @brief Additive update to KV hit/miss counters in @ref ie_metrics_t.
 *
 * Engines may call this to accumulate KV statistics while they run. The CLI
 * merely snapshots the resulting numbers; it does not generate or probe KV.
 *
 * @param m       Pointer to a valid @ref ie_metrics_t.
 * @param hits    Number of hits to add.
 * @param misses  Number of misses to add.
 */
void ie_metrics_kv_add(ie_metrics_t *m, uint64_t hits, uint64_t misses);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* UTIL_METRICS_H */
