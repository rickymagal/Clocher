#ifndef IE_API_H_
#define IE_API_H_

/**
 * @file ie_api.h
 * @brief Public C API for the inference engine (minimal surface used by CLI/tests).
 *
 * The API surface is intentionally small. Fields like ::ie_engine_params_t::precision
 * exist as *soft hints* that backends may honor or ignore.
 *
 * @section precision Precision labels
 * The `precision` hint is a string label accepted by the CLI/tests to describe the
 * desired numeric pathway. In addition to floating-point labels, two weight-only
 * quantization labels are documented:
 *   - "int8w" : weight-only INT8 (activations in fp32/fp16/bf16 as the backend decides)
 *   - "int4w" : weight-only INT4 (packed nibbles, symmetric, zero-point = 0)
 *
 * Engines are free to ignore these hints or to act on them. The CLI never remaps
 * "int4" to "fp32" — when seen, it normalizes to "int4w".
 *
 * @section sparsity Sparsity labels
 * The optional `sparsity` hint describes whether the engine should attempt to use
 * sparse kernels/layouts when available. This is a soft hint:
 *   - "none"       : dense weights (default when unset).
 *   - "block"      : block-sparse weights (e.g., BSR layout).
 *   - "auto"       : let the backend decide based on model metadata.
 *
 * Backends that do not support sparsity should silently ignore this hint.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Status codes                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @typedef ie_status_t
 * @brief Status/result type where `0` indicates success.
 */
typedef int ie_status_t;

/** @brief Success (used by unit tests that assert IE_OK). */
#ifndef IE_OK
#define IE_OK 0
#endif

/* -------------------------------------------------------------------------- */
/* Precision labels (soft hints)                                              */
/* -------------------------------------------------------------------------- */

/** @brief Label for 32-bit floating-point precision hint ("fp32"). */
#define IE_PREC_FP32  "fp32"
/** @brief Label for bfloat16 precision hint ("bf16"). */
#define IE_PREC_BF16  "bf16"
/** @brief Label for 16-bit floating-point precision hint ("fp16"). */
#define IE_PREC_FP16  "fp16"
/** @brief Label for weight-only INT8 precision hint ("int8w"). */
#define IE_PREC_INT8W "int8w"
/** @brief Label for weight-only INT4 precision hint ("int4w"). */
#define IE_PREC_INT4W "int4w"
/** @brief Convenience alias accepted by the CLI; normalized to "int4w". */
#define IE_PREC_INT4  "int4"

/* -------------------------------------------------------------------------- */
/* Opaque handle                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @struct ie_engine
 * @brief Engine opaque handle (incomplete type).
 */
typedef struct ie_engine ie_engine_t;

/* -------------------------------------------------------------------------- */
/* Metrics                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @struct ie_metrics
 * @brief Metrics snapshot returned by the engine.
 *
 * Callers should zero-initialize instances before calling ::ie_engine_metrics
 * to preserve forward compatibility.
 */
typedef struct ie_metrics {
  double   tps_true;        /**< True tokens/s for the last run (or 0). */
  double   latency_p50_ms;  /**< p50 latency in ms (best-effort). */
  double   latency_p95_ms;  /**< p95 latency in ms (best-effort). */
  size_t   rss_peak_mb;     /**< Peak RSS in MB (best-effort). */
  uint64_t kv_hits;         /**< KV cache hits (if applicable). */
  uint64_t kv_misses;       /**< KV cache misses (if applicable). */
} ie_metrics_t;

/* -------------------------------------------------------------------------- */
/* Engine parameters                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @struct ie_engine_params
 * @brief Engine creation parameters (all optional hints).
 *
 * Leave fields zero/NULL to use engine defaults. Unknown hints are ignored.
 * The ::precision field exists for compatibility with tests and CLI.
 *
 * @note Accepted precision labels include:
 *       ::IE_PREC_FP32, ::IE_PREC_BF16, ::IE_PREC_FP16,
 *       ::IE_PREC_INT8W, ::IE_PREC_INT4W (and "int4" as an alias to "int4w").
 *       Weight-only labels are *soft hints* for loaders/backends.
 *
 * @note The optional ::sparsity field is a soft hint for sparse kernels:
 *       - "none"  : dense (default when NULL).
 *       - "block" : block-sparse layout when available.
 *       - "auto"  : backend decides from metadata/model.
 */
typedef struct ie_engine_params {
  int        threads;         /**< Requested worker threads; `<=0` means auto. */
  const char *affinity;       /**< "auto" | "compact" | "scatter" (hint). */
  const char *pretranspose;   /**< "none" | "woh" | "wxh" | "all" (hint). */
  const char *prefetch;       /**< "off" | "on" | "auto" | "0|1|2" (hint). */

  /**
   * @brief Sparsity policy hint for weights.
   *
   * Typical values:
   *  - "none"  : dense layout (default when NULL).
   *  - "block" : block-sparse layout (e.g., BSR) when supported.
   *  - "auto"  : let backend decide based on model metadata.
   *
   * Backends that do not implement sparsity must ignore this field.
   */
  const char *sparsity;

  /* ---- compatibility field consumed by CLI/tests ---- */
  const char *precision;      /**< e.g. "fp32" | "bf16" | "fp16" | "int8w" | "int4w" (or "int4"). */
} ie_engine_params_t;

/* -------------------------------------------------------------------------- */
/* API                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a new engine instance.
 *
 * The engine copies provided parameters by value. Unknown hints are ignored.
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
 * The caller should zero-initialize @p out before calling to preserve
 * forward compatibility.
 *
 * @param[in]  h    Engine handle (const).
 * @param[out] out  Output struct; must be non-`NULL`.
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
