/* ============================================================================
 * File: engine/include/ie_api.h
 * ============================================================================
 */
/**
 * @file ie_api.h
 * @brief Public C API for creating an inference engine instance and generating tokens.
 *
 * @details
 * This header intentionally keeps the interface small and stable:
 *  - The CLI can create an engine for a given (device, model_dir) pair.
 *  - Generation emits integer token ids into a caller-provided buffer.
 *
 * The extended generation entrypoint can optionally return timing statistics:
 *  - Total wall time
 *  - Prefill time
 *  - Decode-loop time
 *  - Decode-only TPS (cannot be inflated by prefill)
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes returned by the public API.
 */
typedef enum ie_status {
  /** @brief Success. */
  IE_OK = 0,
  /** @brief Invalid argument. */
  IE_ERR_BADARG = 1,
  /** @brief Allocation failed. */
  IE_ERR_OOM = 2,
  /** @brief Model could not be loaded or accessed. */
  IE_ERR_MODEL = 3,
  /** @brief Unsupported device or configuration. */
  IE_ERR_UNSUPPORTED = 4,
  /** @brief Internal failure. */
  IE_ERR_INTERNAL = 5
} ie_status_t;

/**
 * @brief Engine creation parameters.
 *
 * @details
 * All string fields are expected to be long-lived for the duration of engine use
 * (or copied by the implementation).
 */
typedef struct ie_engine_params {
  /** @brief Precision label (e.g., "fp32", "bf16", "int4w"). */
  const char *precision;

  /** @brief Sparsity policy (e.g., "none", "block", "auto"). */
  const char *sparsity;

  /** @brief CPU affinity policy (e.g., "auto", "compact", "scatter"). */
  const char *affinity;

  /** @brief Pretranspose mode (e.g., "none", "woh", "wxh", "all"). */
  const char *pretranspose;

  /** @brief Prefetch policy (e.g., "auto", "on", "off", or a numeric string). */
  const char *prefetch;

  /** @brief Thread count (0 means "auto"). */
  int threads;

  /**
   * @brief Optional tokenizer file override.
   *
   * @details
   * When NULL, the engine resolves a tokenizer under model_dir.
   */
  const char *tokenizer_path;

  /**
   * @brief Optional weights JSON path override.
   *
   * @details
   * When NULL, the engine uses the default JSON file under model_dir.
   */
  const char *weights_json_path;

  /**
   * @brief Optional weights BIN path override.
   *
   * @details
   * When NULL, the engine uses the default BIN file under model_dir.
   */
  const char *weights_bin_path;
} ie_engine_params_t;

/** @brief Opaque engine type. */
typedef struct ie_engine ie_engine_t;

/**
 * @brief Optional generation timing statistics.
 *
 * @details
 * Times are in seconds.
 *
 * Interpretation:
 *  - wall_time_s: total time inside generation call (includes encode, KV alloc, prefill, decode loop).
 *  - prefill_time_s: time spent in prefill only.
 *  - decode_time_s: time spent in the decode loop only.
 *  - tps_decode: tokens per second computed from decode_time_s only.
 *  - tps_total: tokens per second computed from wall_time_s (can be useful, but not for decode-only comparisons).
 *  - ttft_s: best-effort time-to-first-token (prefill + first decode step).
 */
typedef struct ie_generate_stats {
  /** @brief Total wall time spent in the generation call. */
  double wall_time_s;

  /** @brief Time spent inside the prefill call. */
  double prefill_time_s;

  /** @brief Time spent in the decode loop (sampling + infer_step). */
  double decode_time_s;

  /** @brief Time-to-first-token (best-effort). */
  double ttft_s;

  /** @brief Decode-only throughput (tokens per second). */
  double tps_decode;

  /** @brief Total-call throughput (tokens per second). */
  double tps_total;
} ie_generate_stats_t;

/**
 * @brief Create an engine instance.
 *
 * @param p Engine params (may be NULL for defaults).
 * @param device Device string (e.g., "cpu", "cuda", "auto").
 * @param model_dir Directory that contains model artifacts.
 * @param out Receives the created engine pointer on success.
 * @return Status code.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out);

/**
 * @brief Destroy an engine instance.
 *
 * @param e Engine pointer (may be NULL).
 */
void ie_engine_destroy(ie_engine_t *e);

/**
 * @brief Generate up to @p max_new tokens for the given prompt.
 *
 * @details
 * This is the stable entrypoint that only returns token ids.
 * For timing breakdowns, use @ref ie_engine_generate_ex.
 *
 * @param e Engine pointer (required).
 * @param prompt Prompt string (may be empty, must be non-NULL).
 * @param max_new Maximum number of tokens to generate.
 * @param out_tokens Output buffer (length at least max_new when max_new>0).
 * @param out_n_tokens Receives number of generated tokens.
 * @return Status code.
 */
ie_status_t ie_engine_generate(const ie_engine_t *e,
                               const char *prompt,
                               size_t max_new,
                               int *out_tokens,
                               size_t *out_n_tokens);

/**
 * @brief Generate tokens and optionally return timing statistics.
 *
 * @param e Engine pointer (required).
 * @param prompt Prompt string (may be empty, must be non-NULL).
 * @param max_new Maximum number of tokens to generate.
 * @param out_tokens Output buffer (length at least max_new when max_new>0).
 * @param out_n_tokens Receives number of generated tokens.
 * @param out_stats Optional timing breakdown (may be NULL).
 * @return Status code.
 */
ie_status_t ie_engine_generate_ex(const ie_engine_t *e,
                                  const char *prompt,
                                  size_t max_new,
                                  int *out_tokens,
                                  size_t *out_n_tokens,
                                  ie_generate_stats_t *out_stats);

#ifdef __cplusplus
} /* extern "C" */
#endif
