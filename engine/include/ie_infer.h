/* ============================================================================
 * File: engine/include/ie_infer.h
 * ============================================================================
 */

#ifndef IE_INFER_H
#define IE_INFER_H

#include <stdint.h>

#include "ie_device.h"
#include "ie_kv_cache.h"
#include "ie_io.h"

/*
 * Minimal hyperparameters needed by a GPT-OSS style forward pass.
 *
 * The complete model implementation is intentionally outside the scope of this
 * header. In this repository snapshot the forward pass is not implemented yet,
 * but the types and entrypoints are kept stable so the rest of the codebase
 * (weights IO, harness, tokenization) can be exercised end-to-end.
 */

typedef struct ie_gptoss_hparams {
  uint32_t n_layers;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t d_model;
  uint32_t d_head;
  uint32_t d_ff;
  uint32_t vocab_size;
  uint32_t max_seq;
} ie_gptoss_hparams_t;

/* Opaque inference context. */
typedef struct ie_gptoss_infer ie_gptoss_infer_t;

/* Create/destroy context. */
int ie_gptoss_infer_create(const ie_device_t *dev,
                           const ie_weights_t *w,
                           const ie_gptoss_hparams_t *hp,
                           ie_gptoss_infer_t **out_ctx);

void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx);

/*
 * Prefill: consume the prompt tokens, update KV cache, and (optionally) produce
 * logits for the last position.
 */
int ie_gptoss_infer_prefill(ie_gptoss_infer_t *ctx,
                            ie_kv_cache *kv,
                            const uint32_t *prompt,
                            uint32_t n_prompt,
                            float *out_logits);

/*
 * Step: consume a single token_id, update KV cache, and produce logits for the
 * next token.
 */
int ie_gptoss_infer_step(ie_gptoss_infer_t *ctx,
                         ie_kv_cache *kv,
                         uint32_t token_id,
                         float *out_logits);

#endif /* IE_INFER_H */
