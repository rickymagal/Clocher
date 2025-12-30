/* ============================================================================
 * File: engine/include/generate_gptoss.h
 * ============================================================================
 */
/**
 * @file generate_gptoss.h
 * @brief High-level GPT-OSS token generation loop (prompt -> token IDs) built on
 *        ie_gptoss_infer_prefill/step() and the sampling utilities.
 *
 * @details
 * This module is the glue between:
 *  - Tokenization (prompt text -> prompt token IDs),
 *  - Transformer forward pass (prefill/step producing logits),
 *  - Sampling (logits -> next token ID),
 *  - Returning a generated token stream to the caller.
 *
 * The actual transformer math must be implemented in:
 *  - ie_gptoss_infer_prefill()
 *  - ie_gptoss_infer_step()
 *
 * Once those are real, this module provides a strict, deterministic and
 * harness-friendly generation path.
 *
 * Design goals:
 *  - No global state.
 *  - Allocation is local to the generate call (correctness-first).
 *  - Minimal assumptions about the surrounding engine; the caller provides
 *    device handle, weights handle, tokenizer handle and model directory.
 */

#ifndef GENERATE_GPTOSS_H
#define GENERATE_GPTOSS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "ie_device.h"
#include "ie_infer.h"
#include "ie_io.h"
#include "ie_sampling.h"
#include "ie_tokenizer_gptoss.h"

/**
 * @brief Status codes returned by the GPT-OSS generation wrapper.
 *
 * @details
 * These are intentionally simple and negative on error, similar to other
 * internal modules in this repository.
 */
typedef enum ie_generate_gptoss_status_e {
  /** @brief Success. */
  IE_GEN_GPTOSS_OK = 0,
  /** @brief Invalid arguments. */
  IE_GEN_GPTOSS_ERR_ARGS = -1,
  /** @brief I/O error reading model config or tokenizer usage failure. */
  IE_GEN_GPTOSS_ERR_IO = -2,
  /** @brief Model files/weights/hparams inconsistent or inference create failed. */
  IE_GEN_GPTOSS_ERR_MODEL = -3,
  /** @brief Out-of-memory. */
  IE_GEN_GPTOSS_ERR_NOMEM = -4,
  /** @brief Feature not implemented in the current snapshot. */
  IE_GEN_GPTOSS_ERR_UNSUPPORTED = -5
} ie_generate_gptoss_status_t;

/**
 * @brief Fill a conservative default sampling configuration.
 *
 * @details
 * Defaults are chosen to be predictable and safe for early bring-up:
 *  - Greedy selection (argmax),
 *  - Temperature=1.0,
 *  - No top-k/top-p filtering.
 *
 * Callers may override any fields after this call.
 *
 * @param out_cfg Output sampling configuration (required).
 */
void ie_generate_gptoss_default_sample_cfg(ie_sample_cfg_t *out_cfg);

/**
 * @brief Generate up to @p max_new tokens for a prompt string using GPT-OSS.
 *
 * @details
 * This function performs:
 *  1) Load minimal hyperparameters from @p model_dir/config.json.
 *  2) Encode @p prompt into token IDs using @p tok.
 *  3) Create an inference context via ie_gptoss_infer_create().
 *  4) Allocate a KV cache sized from hyperparameters.
 *  5) Run prefill over the prompt, then step+sample to generate new tokens.
 *
 * The generation loop is:
 *  - logits = prefill(prompt)
 *  - repeat max_new:
 *      next = sample(logits)
 *      out_tokens[i] = next
 *      logits = step(next)
 *
 * Requirements:
 *  - @p tok must be a valid tokenizer handle capable of encoding the prompt.
 *  - Real inference must exist in ie_gptoss_infer_prefill/step.
 *
 * @param dev Device backend handle (required).
 * @param w Loaded weights handle (required).
 * @param model_dir Model directory containing config.json (required).
 * @param tok Tokenizer handle opened from tokenizer.json (required).
 * @param prompt UTF-8 prompt string (non-NULL; may be empty).
 * @param max_new Maximum number of tokens to generate.
 * @param sample_cfg Sampling configuration (may be NULL for defaults).
 * @param seed RNG seed (0 is remapped to a non-zero value).
 * @param out_tokens Output buffer (length at least @p max_new when @p max_new > 0).
 * @param out_n_tokens Receives number of generated tokens.
 * @return Status code from ::ie_generate_gptoss_status_t.
 */
int ie_generate_gptoss(const ie_device_t *dev,
                       const ie_weights_t *w,
                       const char *model_dir,
                       const ie_tok_gptoss_t *tok,
                       const char *prompt,
                       uint32_t max_new,
                       const ie_sample_cfg_t *sample_cfg,
                       uint64_t seed,
                       int *out_tokens,
                       uint32_t *out_n_tokens);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GENERATE_GPTOSS_H */
