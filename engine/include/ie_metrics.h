/**
 * @file ie_metrics.h
 * @brief Metrics data structures for the Inference Engine.
 *
 * @defgroup IE_METRICS Metrics
 * @brief Snapshot structure returned by @ref ie_engine_metrics.
 * @{
 */
#ifndef IE_METRICS_H
#define IE_METRICS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Monotonic metrics captured across the last generation run.
 *
 * All timings are reported in milliseconds; sizes in megabytes; counters are monotonic.
 */
typedef struct {
  double   tps_true;        /**< Engine-estimated tokens/s from p50 (harness reports wall-clock TPS separately). */
  double   latency_p50_ms;  /**< Median per-token latency over the last run. */
  double   latency_p95_ms;  /**< 95th percentile per-token latency over the last run. */
  size_t   rss_peak_mb;     /**< Peak RSS in MB (baseline: 0; can be filled via /proc/self/statm). */
  uint64_t kv_hits;         /**< KV-cache hits (baseline: 0). */
  uint64_t kv_misses;       /**< KV-cache misses (baseline: equals tokens_generated). */
} ie_metrics_t;

#ifdef __cplusplus
}
#endif
/** @} */ /* end of IE_METRICS */
#endif /* IE_METRICS_H */
