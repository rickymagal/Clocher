/* ============================================================================
 * File: engine/src/runtime/infer_gptoss.c
 * ============================================================================
 *
 * GPT-OSS inference entrypoints (stub snapshot).
 *
 * IMPORTANT:
 * This repository snapshot does not ship a full transformer forward pass.
 * Therefore, prefill() and step() return an error. Higher-level plumbing can
 * still be compiled and exercised (weights IO, tokenizer IO, harness), but any
 * attempt to generate real text must fail unless a real forward path is added.
 */

#include <stdlib.h>

#include "ie_infer.h"

/**
 * @brief Private implementation backing the public ie_gptoss_infer_t handle.
 *
 * The snapshot keeps only references to objects owned elsewhere.
 * A real implementation would allocate per-context scratch buffers, KV cache
 * metadata, and backend-specific resources.
 */
struct ie_gptoss_infer_impl {
  /** @brief Device backend (CPU/CUDA). */
  const ie_device_t *dev;

  /** @brief Loaded model weights (possibly deduplicated / reconstructed). */
  const ie_weights_t *weights;

  /** @brief Model hyperparameters (layers, heads, hidden size, etc.). */
  const ie_gptoss_hparams_t *hparams;
};

int ie_gptoss_infer_create(const ie_device_t *dev,
                           const ie_weights_t *w,
                           const ie_gptoss_hparams_t *hp,
                           ie_gptoss_infer_t **out_ctx) {
  if (!out_ctx) return -1;
  *out_ctx = NULL;

  struct ie_gptoss_infer_impl *impl =
      (struct ie_gptoss_infer_impl *)calloc(1, sizeof(*impl));
  if (!impl) return -1;

  impl->dev = dev;
  impl->weights = w;
  impl->hparams = hp;

  *out_ctx = (ie_gptoss_infer_t *)impl;
  return 0;
}

void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx) {
  if (!ctx) return;
  free(ctx);
}

int ie_gptoss_infer_prefill(ie_gptoss_infer_t *ctx,
                            ie_kv_cache *kv,
                            const uint32_t *prompt,
                            uint32_t n_prompt,
                            float *out_logits) {
  (void)ctx;
  (void)kv;
  (void)prompt;
  (void)n_prompt;
  (void)out_logits;

  /* Snapshot stub: no transformer forward pass is shipped in this repo. */
  return -1;
}

int ie_gptoss_infer_step(ie_gptoss_infer_t *ctx,
                         ie_kv_cache *kv,
                         uint32_t token_id,
                         float *out_logits) {
  (void)ctx;
  (void)kv;
  (void)token_id;
  (void)out_logits;

  /* Snapshot stub: no transformer forward pass is shipped in this repo. */
  return -1;
}
