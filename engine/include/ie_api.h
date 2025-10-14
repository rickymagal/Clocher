/**
 * @file ie_api.h
 * @brief Public C API for the inference engine (parameters, run, metrics).
 */
#ifndef IE_API_H_
#define IE_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes returned by engine functions. */
typedef enum ie_status {
  IE_OK = 0,                 /**< Success. */
  IE_ERR_INVALID_ARGUMENT=1, /**< A parameter was invalid. */
  IE_ERR_INTERNAL=2          /**< Internal error (allocation, etc.). */
} ie_status_t;

/** @brief Forward-declared opaque engine handle. */
typedef struct ie_engine ie_engine_t;

/**
 * @brief Engine creation parameters (borrowed strings; no ownership taken).
 */
typedef struct ie_engine_params {
  /** Optional: path to JSON with model shape metadata. */
  const char *shape_json_path;
  /** Optional: path to weights binary. */
  const char *weights_path;
  /** Optional: path to vocab JSON. */
  const char *vocab_path;

  /** Number of threads (0 = engine decides; engine defaults to 1 for stability). */
  uint32_t threads;

  /** Precision hint: "fp32" (default), "bf16", "fp16". */
  const char *precision;

  /**
   * Affinity hint ("auto"|"compact"|"scatter").
   * Effective on Linux only and only when CPU affinity is enabled at runtime
   * via the environment variable IE_TP_USE_AFFINITY=1.
   */
  const char *affinity;

  /**
   * Grainsize hint for future block-splitting. Currently informational only;
   * the contiguous partition does not use it.
   */
  uint32_t grainsize;

  /**
   * NUMA mode string for documentation/CLI consistency ("compact", "interleave",
   * or "node:X"). Implementation in this baseline is provided by the external
   * scripts/set_numa.sh helper and not enforced inside the engine.
   */
  const char *numa_mode;
} ie_engine_params_t;

/** @brief Metrics snapshot returned by the engine. */
typedef struct ie_metrics {
  double tps_true;            /**< Approximate true tokens/sec (from p50). */
  double latency_p50_ms;      /**< Per-token latency p50 in milliseconds. */
  double latency_p95_ms;      /**< Per-token latency p95 in milliseconds. */
  size_t rss_peak_mb;         /**< Peak resident set size in megabytes. */
  unsigned long long kv_hits;   /**< Simulated KV cache hits. */
  unsigned long long kv_misses; /**< Simulated KV cache misses. */
} ie_metrics_t;

/**
 * @brief Create an inference engine instance.
 *
 * @param p    Optional parameters; may be NULL for defaults.
 * @param out  Output engine handle. Must not be NULL.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out);

/**
 * @brief Destroy an inference engine instance and release resources.
 *
 * @param h Engine handle (NULL allowed; no-op).
 */
void ie_engine_destroy(ie_engine_t *h);

/**
 * @brief Generate up to @p max_new_tokens tokens from @p prompt.
 *
 * If @p max_new_tokens == 0, @p out_tokens may be NULL and the function
 * returns immediately with @p *out_count set to 0.
 *
 * @param h               Engine handle.
 * @param prompt          UTF-8 input string (may be NULL/empty).
 * @param max_new_tokens  Number of tokens requested.
 * @param out_tokens      Output buffer (length >= max_new_tokens). May be NULL
 *                        iff @p max_new_tokens == 0.
 * @param out_count       Receives number of tokens actually generated.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count);

/**
 * @brief Snapshot performance metrics accumulated by the engine.
 *
 * @param h   Engine handle.
 * @param out Output metrics pointer to fill. Must not be NULL.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out);

#ifdef __cplusplus
}
#endif

#endif /* IE_API_H_ */
