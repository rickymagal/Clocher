/* ========================================================================== */
/* File: engine/include/ie_api.h                                              */
/* ========================================================================== */
#ifndef IE_API_H_
#define IE_API_H_

/**
 * @file ie_api.h
 * @brief Public C API for the inference engine (minimal surface used by CLI/tests).
 *
 * The API is intentionally small. Fields like @ref ie_engine_params_t::precision
 * exist for compatibility with tests; engines may ignore hints they do not need.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Status codes (compat)                                                      */
/* -------------------------------------------------------------------------- */

/** @brief Status/result type. `0` indicates success. */
typedef int ie_status_t;

/** @brief Success (compat for tests that assert IE_OK). */
#ifndef IE_OK
#define IE_OK 0
#endif

/* -------------------------------------------------------------------------- */
/* Opaque handles                                                             */
/* -------------------------------------------------------------------------- */

/** @brief Engine opaque handle. */
typedef struct ie_engine ie_engine_t;

/* -------------------------------------------------------------------------- */
/* Metrics                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Metrics snapshot returned by the engine.
 *
 * Only fields used by CLI/tests are exposed here. The caller must zero-init
 * instances before calling @ref ie_engine_metrics for forward compatibility.
 */
typedef struct ie_metrics {
  double   tps_true;        /**< True tokens/s across the last run (or 0). */
  double   latency_p50_ms;  /**< p50 latency in ms (best-effort). */
  double   latency_p95_ms;  /**< p95 latency in ms (best-effort). */
  size_t   rss_peak_mb;     /**< Peak RSS in MB (best-effort). */
  uint64_t kv_hits;         /**< KV cache hits (if applicable). */
  uint64_t kv_misses;       /**< KV cache misses (if applicable). */
} ie_metrics_t;

/* -------------------------------------------------------------------------- */
/* Engine parameters                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Engine creation parameters (all optional hints).
 *
 * Leave fields zero/NULL to use engine defaults. Unknown hints are ignored.
 * The @ref precision field exists for compatibility with tests that set it.
 */
typedef struct ie_engine_params {
  int        threads;         /**< Requested worker threads; `<=0` means auto. */
  const char *affinity;       /**< "auto" | "compact" | "scatter" (hint). */
  const char *pretranspose;   /**< "none" | "woh" | "wxh" | "all" (hint). */
  const char *prefetch;       /**< "off" | "on" | "auto" | "0|1|2" (hint). */

  /* ---- compatibility field expected by tests (may be ignored by engine) --- */
  const char *precision;      /**< "fp32" | "bf16" | "fp16" (compat; optional). */
} ie_engine_params_t;

/* -------------------------------------------------------------------------- */
/* API                                                                         */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a new engine instance.
 *
 * The engine copies or internally references the provided parameters. Unknown
 * hints are ignored. On success, @p *out is set to a valid handle.
 *
 * @param[in]  p    Optional parameters (may be `NULL` for defaults).
 * @param[out] out  Output engine handle; must be non-`NULL`.
 * @retval IE_OK          on success.
 * @retval non-zero       on failure (resource error, invalid args, etc).
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out);

/**
 * @brief Generate up to @p max_new_tokens from @p prompt.
 *
 * Implementations may output fewer tokens (e.g., due to EOS). The caller owns
 * the @p out_tokens buffer and must supply capacity for @p max_new_tokens.
 *
 * @param[in]  h                Engine handle.
 * @param[in]  prompt           NUL-terminated prompt text.
 * @param[in]  max_new_tokens   Upper bound on tokens to produce.
 * @param[out] out_tokens       Output buffer for token IDs (size >= @p max_new_tokens).
 * @param[out] out_count        Number of tokens actually produced (<= @p max_new_tokens).
 * @retval IE_OK          on success.
 * @retval non-zero       on failure.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               size_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count);

/**
 * @brief Snapshot engine metrics into @p out.
 *
 * The caller must zero-initialize @p out before calling to preserve
 * forward compatibility.
 *
 * @param[in]  h    Engine handle (const).
 * @param[out] out  Output struct; must be non-`NULL` and zero-initialized.
 * @retval IE_OK          on success.
 * @retval non-zero       on failure (invalid args, etc).
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out);

/**
 * @brief Destroy an engine and free its resources.
 *
 * @param[in] h Engine handle (may be `NULL`; no-op).
 */
void ie_engine_destroy(ie_engine_t *h);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_API_H_ */
