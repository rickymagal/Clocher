/**
 * @file ie_metrics.h
 * @brief Metrics data structures for the Inference Engine.
 */
#ifndef IE_METRICS_H
#define IE_METRICS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Monotonic metrics snapshot. */
typedef struct {
  double   tps_true;         /**< True tokens-per-second (generated only). */
  double   latency_p50_ms;   /**< Median per-token latency in milliseconds. */
  double   latency_p95_ms;   /**< 95th percentile per-token latency in ms. */
  size_t   rss_peak_mb;      /**< Peak resident set size in megabytes. */
  uint64_t kv_hits;          /**< KV-cache hits (baseline: zero). */
  uint64_t kv_misses;        /**< KV-cache misses (baseline: tokens). */
} ie_metrics_t;

#ifdef __cplusplus
}
#endif
#endif /* IE_METRICS_H */
