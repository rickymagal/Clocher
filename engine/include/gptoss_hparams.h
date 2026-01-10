/* ============================================================================
 * File: engine/include/gptoss_hparams.h
 * ============================================================================
 */
/**
 * @file gptoss_hparams.h
 * @brief Load GPT-OSS hyperparameters from a HuggingFace-style `config.json`.
 *
 * @details
 * Real inference needs a small set of model hyperparameters (layer/head counts,
 * embedding dimensions, vocabulary size, and maximum sequence length).
 *
 * The engine’s stable inference interface uses ::ie_gptoss_hparams_t (declared
 * in `ie_infer.h`). This module provides a dependency-free loader that reads a
 * HuggingFace `config.json` and fills those fields.
 *
 * The parser is intentionally relaxed:
 *  - no third-party JSON library,
 *  - tolerant of extra fields and whitespace,
 *  - extracts only the keys required by the forward pass.
 *
 * In addition to the core fields, this module can also extract commonly-needed
 * extras (RMSNorm epsilon, RoPE configuration, and tie_word_embeddings) through
 * ::gptoss_hparams_ex_t.
 */

#ifndef GPTOSS_HPARAMS_H
#define GPTOSS_HPARAMS_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "ie_infer.h"
#include "ie_io.h"

/* ============================================================================
 * Public types
 * ========================================================================== */

/**
 * @struct gptoss_hparams_ex_t
 * @brief Extended hyperparameters typically required by a GPT-OSS forward pass.
 *
 * @details
 * The core engine interface uses ::ie_gptoss_hparams_t. A full forward pass
 * also commonly needs:
 *  - RMSNorm epsilon (HF key: `rms_norm_eps`),
 *  - RoPE base theta (HF key: `rope_theta`),
 *  - optional RoPE scaling (HF object: `rope_scaling`),
 *  - whether `lm_head` is tied to the token embedding (`tie_word_embeddings`).
 *
 * The ::ie_gptoss_hparams_t is embedded as the first member so this structure
 * can be safely passed to APIs that expect a pointer to the core type when
 * appropriate.
 */
typedef struct gptoss_hparams_ex_t {
  /** Core hyperparameters (must remain the first field). */
  ie_gptoss_hparams_t core;

  /** RMSNorm epsilon for normalization layers. */
  float rms_norm_eps;

  /** RoPE base theta (typically 10000.0). */
  float rope_theta;

  /**
   * RoPE scaling type (e.g., "linear", "dynamic", "yarn").
   * Empty string means "no scaling".
   */
  char rope_scaling_type[32];

  /** RoPE scaling factor. Defaults to 1.0 when scaling is absent. */
  float rope_scaling_factor;

  /**
   * Original max position embeddings sometimes provided by rope_scaling
   * (common for YARN-style configs). 0 means "absent".
   */
  uint32_t rope_scaling_original_max_position_embeddings;

  /**
   * Non-zero if output embeddings are tied to token embeddings.
   * Defaults to 0 when the key is absent.
   */
  int tie_word_embeddings;
} gptoss_hparams_ex_t;

/* ============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Load core hyperparameters from a model directory.
 *
 * @details
 * This function searches for a HuggingFace `config.json` in:
 *  1) `<model_dir>/config.json`
 *  2) `<model_dir>/hf/original/config.json`
 *
 * It fills ::ie_gptoss_hparams_t with:
 *  - `n_layers` (HF key: `num_hidden_layers`)
 *  - `n_heads` (HF key: `num_attention_heads`)
 *  - `n_kv_heads` (HF key: `num_key_value_heads`, defaults to `n_heads`)
 *  - `d_model` (HF key: `hidden_size`)
 *  - `d_ff` (HF key: `intermediate_size`)
 *  - `vocab_size` (HF key: `vocab_size`)
 *  - `max_seq` (HF key: `max_position_embeddings`)
 *
 * It also derives:
 *  - `d_head = d_model / n_heads` (must divide evenly)
 *
 * @param model_dir Model directory containing HuggingFace artifacts.
 * @param out_hp    Output hyperparameter struct (written on success).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_hparams_load(const char *model_dir, ie_gptoss_hparams_t *out_hp);

/**
 * @brief Load extended hyperparameters from a model directory.
 *
 * @details
 * Same search behavior as ::gptoss_hparams_load, but also extracts:
 *  - RMSNorm epsilon (keys: `rms_norm_eps`, `rms_epsilon`, `layer_norm_eps`)
 *  - RoPE theta (`rope_theta`)
 *  - RoPE scaling object (`rope_scaling.type`, `rope_scaling.factor`)
 *  - tie_word_embeddings (`tie_word_embeddings`)
 *
 * Missing optional fields are filled with conservative defaults:
 *  - `rms_norm_eps = 1e-5f`
 *  - `rope_theta = 10000.0f`
 *  - `rope_scaling_type = ""`, `rope_scaling_factor = 1.0f`
 *  - `rope_scaling_original_max_position_embeddings = 0`
 *  - `tie_word_embeddings = 0`
 *
 * @param model_dir Model directory containing HuggingFace artifacts.
 * @param out_ex    Output extended hyperparameter struct (written on success).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_hparams_load_ex(const char *model_dir, gptoss_hparams_ex_t *out_ex);

/**
 * @brief Load extended hyperparameters from an explicit HuggingFace config path.
 *
 * @param config_json_path Path to a HuggingFace `config.json`.
 * @param out_ex           Output extended hyperparameter struct (written on success).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_hparams_load_ex_from_file(const char *config_json_path,
                                     gptoss_hparams_ex_t *out_ex);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPTOSS_HPARAMS_H */
