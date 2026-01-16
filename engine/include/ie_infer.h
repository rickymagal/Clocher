/* ============================================================================
 * File: engine/include/ie_infer.h
 * ============================================================================
 */

#ifndef IE_INFER_H
#define IE_INFER_H

#include <stdint.h>

#include "ie_device.h"
#include "ie_io.h"
#include "ie_kv_cache.h"

/**
 * @file ie_infer.h
 * @brief Inference interface for the GPT-OSS runtime.
 *
 * @details
 *
 * Performance notes:
 * - INT4 runs use Q4_0 weight GEMV kernels (runtime-dispatched to AVX2/FMA when available).
 * - Per-32-column scales may be stored as BF16 (2 bytes) or FP8(E4M3) (1 byte) depending on artifacts.

 * This header defines the stable ABI used by the public API layer (ie_api.c)
 * and by the CLI harness (main_infer.c) to run inference.
 *
 * The runtime implementation is model-specific and lives under engine/src/runtime.
 * The interface is intentionally minimal:
 *  - Create/destroy an inference context bound to a device + weights + hparams.
 *  - Prefill a prompt (multi-token).
 *  - Step one token at a time.
 *
 * The KV cache storage is provided by the caller (one cache per layer) via
 * ie_kv_cache so the runtime can update attention state incrementally.
 */

/**
 * @brief Minimal hyperparameters required by a GPT-OSS style forward pass.
 */
typedef struct ie_gptoss_hparams {
  uint32_t n_layers;    /**< Number of transformer layers. */
  uint32_t n_heads;     /**< Number of attention heads. */
  uint32_t n_kv_heads;  /**< Number of KV heads (GQA/MQA). */
  uint32_t d_model;     /**< Model width. */
  uint32_t d_head;      /**< Per-head dimension. */
  uint32_t d_ff;        /**< MLP hidden dimension. */
  uint32_t vocab_size;  /**< Vocabulary size. */
  uint32_t max_seq;     /**< Maximum sequence length supported by the model. */
} ie_gptoss_hparams_t;

/**
 * @brief Opaque inference context.
 */
typedef struct ie_gptoss_infer ie_gptoss_infer_t;

/**
 * @brief Create an inference context.
 *
 * @param dev Device backend (CPU/CUDA).
 * @param w   Opened weights handle.
 * @param hp  Hyperparameters.
 * @param out_ctx Receives the newly created context on success.
 *
 * @return 0 on success; non-zero on failure.
 */
int ie_gptoss_infer_create(const ie_device_t *dev,
                           const ie_weights_t *w,
                           const ie_gptoss_hparams_t *hp,
                           ie_gptoss_infer_t **out_ctx);

/**
 * @brief Destroy an inference context.
 */
void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx);

/**
 * @brief Prefill: consume a prompt, update the KV cache, and produce logits.
 *
 * @param ctx        Inference context.
 * @param kv         Array of KV caches, one per layer.
 * @param prompt     Prompt token IDs.
 * @param n_prompt   Number of prompt tokens.
 * @param out_logits Output logits for the last position (vocab-sized).
 *
 * @return 0 on success; non-zero on failure.
 */
int ie_gptoss_infer_prefill(ie_gptoss_infer_t *ctx,
                            ie_kv_cache *kv,
                            const uint32_t *prompt,
                            uint32_t n_prompt,
                            float *out_logits);

/**
 * @brief Step: consume one token, update the KV cache, and produce next logits.
 *
 * @param ctx        Inference context.
 * @param kv         Array of KV caches, one per layer.
 * @param token_id   The token to consume.
 * @param out_logits Output logits for the next token (vocab-sized).
 *
 * @return 0 on success; non-zero on failure.
 */
int ie_gptoss_infer_step(ie_gptoss_infer_t *ctx,
                         ie_kv_cache *kv,
                         uint32_t token_id,
                         float *out_logits);

/**
 * @brief Get the current position (number of tokens already consumed).
 *
 * @details
 *
 * Performance notes:
 * - INT4 runs use Q4_0 weight GEMV kernels (runtime-dispatched to AVX2/FMA when available).
 * - Per-32-column scales may be stored as BF16 (2 bytes) or FP8(E4M3) (1 byte) depending on artifacts.

 * The runtime maintains a position counter internally. This is used to index
 * into the KV cache (time dimension) and to drive rotary embeddings.
 */
uint32_t ie_gptoss_infer_get_pos(const ie_gptoss_infer_t *ctx);

/**
 * @brief Set the current position.
 *
 * @details
 *
 * Performance notes:
 * - INT4 runs use Q4_0 weight GEMV kernels (runtime-dispatched to AVX2/FMA when available).
 * - Per-32-column scales may be stored as BF16 (2 bytes) or FP8(E4M3) (1 byte) depending on artifacts.

 * This exists to enable persistent prompt-prefix reuse: if the KV cache already
 * contains valid K/V entries for positions [0, pos-1], the caller can rewind the
 * runtime position to @p pos and continue generation without recomputing the
 * prefix.
 */
void ie_gptoss_infer_set_pos(ie_gptoss_infer_t *ctx, uint32_t pos);

#endif /* IE_INFER_H */
