/**
 * @file ie_api.h
 * @brief Public C API for the Inference Engine core (baseline FP32 path).
 *
 * @defgroup IE_API Inference Engine C API
 * @brief Public entry points for engine lifecycle, generation, and metrics.
 * @{
 */
#ifndef IE_API_H
#define IE_API_H

#include <stddef.h>
#include <stdint.h>
#include "ie_metrics.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque engine handle. */
typedef struct ie_engine ie_engine_t;

/**
 * @brief Return codes for API functions.
 */
typedef enum {
  IE_OK = 0,                 /**< Operation completed successfully. */
  IE_ERR_INVALID_ARGUMENT=1, /**< One or more arguments are invalid (NULL, out of range, etc.). */
  IE_ERR_IO = 2,             /**< I/O error while accessing model/vocab files. */
  IE_ERR_UNSUPPORTED = 3,    /**< Requested feature/precision is not supported in the current build. */
  IE_ERR_INTERNAL = 255      /**< Internal error (e.g., allocation failure). */
} ie_status_t;

/**
 * @brief Engine creation parameters.
 *
 * All string pointers may be NULL; sensible defaults will be used in the baseline build.
 */
typedef struct {
  const char *weights_path;     /**< Path to model.ie.bin (IEBIN v1). May be NULL. */
  const char *shape_json_path;  /**< Path to model.ie.json (IEBIN v1). May be NULL. */
  const char *vocab_path;       /**< Path to vocab.json. May be NULL. */
  uint32_t    threads;          /**< 0 = auto (unused in baseline). */
  const char *affinity;         /**< "auto"|"compact"|"scatter"|cpu list (unused in baseline). */
  const char *precision;        /**< "fp32"|"bf16"|"int8w" (baseline uses "fp32"). */
} ie_engine_params_t;

/**
 * @brief Create a new engine instance.
 *
 * @param[in]  p       Optional pointer to initialization parameters (may be NULL).
 * @param[out] out     Output pointer for the created engine handle.
 * @return IE_OK on success; error code otherwise.
 *
 * @note The engine owns all internal resources. Destroy it via ::ie_engine_destroy.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out);

/**
 * @brief Destroy an engine instance and release all associated resources.
 *
 * @param[in,out] e Engine handle to destroy (NULL is allowed and is a no-op).
 */
void ie_engine_destroy(ie_engine_t *e);

/**
 * @brief Generate tokens for a given prompt.
 *
 * Performs a prompt prefill, then decodes up to @p max_new_tokens tokens.
 *
 * @param[in]  e               Engine handle created by ::ie_engine_create.
 * @param[in]  prompt          UTF-8 input string (may be NULL to indicate empty).
 * @param[in]  max_new_tokens  Maximum number of tokens to generate (0 allowed).
 * @param[out] out_tokens      Caller-allocated buffer to receive token IDs
 *                             (must have capacity for @p max_new_tokens).
 * @param[out] out_count       Number of tokens actually generated.
 * @return IE_OK on success; error code otherwise.
 *
 * @warning The caller owns @p out_tokens and must ensure it is large enough.
 */
ie_status_t ie_engine_generate(ie_engine_t *e,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count);

/**
 * @brief Snapshot engine metrics accumulated during the most recent run.
 *
 * @param[in]  e    Engine handle.
 * @param[out] out  Filled with metrics such as p50/p95 latency and TPS estimate.
 * @return IE_OK on success; error code otherwise.
 *
 * @note The harness computes "true TPS" using end-to-end wall time; the engine’s
 *       @ref ie_metrics_t::tps_true field is derived from per-token p50.
 */
ie_status_t ie_engine_metrics(const ie_engine_t *e, ie_metrics_t *out);

#ifdef __cplusplus
}
#endif
/** @} */ /* end of IE_API */
#endif /* IE_API_H */
