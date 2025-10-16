/**
 * @file util_metrics.h
 * @brief Metrics result types for CLI/json reporting and tests.
 *
 * This header intentionally exposes only the POD result structures used by
 * the CLI and tests. The corresponding implementation utilities may live in
 * util_metrics.c (or elsewhere) and are free to keep their APIs internal.
 *
 * Rationale:
 * - Main CLI prints JSON using these fields.
 * - Tests parse these exact keys.
 * - Avoid leaking helper function signatures to keep linkage flexible.
 */

#ifndef UTIL_METRICS_H
#define UTIL_METRICS_H

#include <stddef.h> /* size_t */

/**
 * @brief Latency percentile snapshot (milliseconds).
 *
 * This structure carries precomputed latency percentiles from a ring-buffer
 * (or any other collector). Units are milliseconds.
 */
typedef struct ie_latency_stats {
  double p50_ms;  /**< Median latency (p50), in milliseconds. */
  double p95_ms;  /**< Tail latency (p95), in milliseconds. */
} ie_latency_stats_t;

/**
 * @brief Aggregate metrics commonly emitted by the CLI in JSON form.
 *
 * Field names mirror the JSON keys produced by the binary, e.g.:
 * {
 *   "tokens_generated": 8,
 *   "tokens": [ ... ],
 *   "wall_time_s": 0.123,
 *   "tps_true": 6500.0,
 *   "latency_p50_ms": 0.21,
 *   "latency_p95_ms": 0.31,
 *   "rss_peak_mb": 123,
 *   "kv_hits": 0,
 *   "kv_misses": 8
 * }
 */
typedef struct ie_metrics_result {
  int    tokens_generated;   /**< Number of tokens generated in the run. */
  double wall_time_s;        /**< Elapsed wall time in seconds. */
  double tps_true;           /**< True tokens-per-second (tokens_generated / wall_time_s). */
  double latency_p50_ms;     /**< p50 latency in milliseconds. */
  double latency_p95_ms;     /**< p95 latency in milliseconds. */
  size_t rss_peak_mb;        /**< Peak resident set size observed (MiB). */
  size_t kv_hits;            /**< KV-cache hits observed (if applicable). */
  size_t kv_misses;          /**< KV-cache misses observed (if applicable). */
} ie_metrics_result_t;

#endif /* UTIL_METRICS_H */
