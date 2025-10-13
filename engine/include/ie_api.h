/**
 * @file ie_api.h
 * @brief Public C API for the Inference Engine core (baseline).
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

/** @brief Return codes. */
typedef enum {
  IE_OK = 0,
  IE_ERR_INVALID_ARGUMENT = 1,
  IE_ERR_IO = 2,
  IE_ERR_UNSUPPORTED = 3,
  IE_ERR_INTERNAL = 255
} ie_status_t;

/** @brief Engine params (baseline subset). */
typedef struct {
  const char *weights_path;      /**< Path to model.ie.bin (unused baseline). */
  const char *shape_json_path;   /**< Path to model.ie.json (unused baseline). */
  const char *vocab_path;        /**< Path to vocab.json (unused baseline). */
  uint32_t    threads;           /**< 0 = auto (unused baseline). */
  const char *affinity;          /**< "auto"|... (unused baseline). */
  const char *precision;         /**< "fp32"|"bf16"|"int8w" (baseline: "fp32"). */
} ie_engine_params_t;

/** @brief Create/destroy the engine. */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out);
void        ie_engine_destroy(ie_engine_t *e);

/**
 * @brief Generate tokens for a prompt (baseline dummy generator).
 *
 * The baseline produces deterministic fake tokens quickly to validate
 * the harness/metrics pipeline without external dependencies.
 */
ie_status_t ie_engine_generate(ie_engine_t *e,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count);

/** @brief Snapshot current metrics. */
ie_status_t ie_engine_metrics(const ie_engine_t *e, ie_metrics_t *out);

#ifdef __cplusplus
}
#endif
#endif /* IE_API_H */
