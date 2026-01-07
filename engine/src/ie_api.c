/* ============================================================================
 * File: engine/src/ie_api.c
 * ============================================================================
 */
/**
 * @file ie_api.c
 * @brief Stable public API implementation for creating, destroying, and running an engine instance.
 *
 * @details
 * This module implements the functions declared in @ref ie_api.h. It keeps the
 * public surface small and stable while delegating model-specific logic to internal
 * subsystems:
 *  - Device creation/teardown (CPU/CUDA selection)
 *  - Tokenizer open/encode/decode (currently GPT-OSS tokenizer via tokenizer.json)
 *  - Weights open/close (model.ie.json + model.ie.bin)
 *  - Hyperparameter loading
 *  - GPT-OSS runtime creation and token-by-token inference
 *  - Sampling to select the next token
 *
 * Logging policy:
 *  - Stage-level INFO logs during create/generate to quickly locate failures.
 *  - ERROR logs with rc/status codes and key parameters on failure paths.
 *  - Per-token logs are opt-in via environment variables.
 *
 * Environment variables (optional):
 *  - IE_API_LOG_TOKENS:         If set to 1, log prompt token ids (truncated).
 *  - IE_API_LOG_EVERY:          If set to N>0, log every N decode steps (token id).
 *  - IE_API_DEBUG_DECODE:       If set to 1, attempt to decode generated tokens to UTF-8 and log (best-effort).
 *  - IE_API_DEBUG_DECODE_EVERY: If set to N>0, decode/log every N decode steps when IE_API_DEBUG_DECODE=1.
 *
 * Error-handling policy:
 *  - Fail fast with a clear return code (@ref ie_status_t).
 *  - On failure, release partially acquired resources before returning.
 *  - Do not expose partially initialized objects to the caller.
 *
 * Header layout constraint:
 *  - The include directory is flat. All includes must reference headers by filename only.
 */

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#ifndef _XOPEN_SOURCE
#  define _XOPEN_SOURCE 700
#endif

#include "ie_api.h"

#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gptoss_hparams.h"
#include "ie_device.h"
#include "ie_infer.h"
#include "ie_io.h"
#include "ie_kv_cache.h"
#include "ie_sampling.h"
#include "ie_tokenizer_gptoss.h"
#include "tensor_map.h"
#include "util_logging.h"

/* ============================================================================
 * Internal engine object
 * ========================================================================== */

/**
 * @brief Concrete engine implementation backing the opaque @ref ie_engine_t handle.
 *
 * @details
 * The public API exposes @ref ie_engine_t as an opaque struct. This file defines
 * the concrete layout, collecting all long-lived objects needed for generation:
 *  - Device handle
 *  - Weights handle
 *  - Tokenizer handle
 *  - Model hyperparameters
 *  - Inference runtime handle
 *  - Scratch buffers (logits and sampler scratch)
 *  - Sampler configuration and RNG state
 */
struct ie_engine {
  /** @brief Device instance (CPU/CUDA). */
  ie_device_t *dev;

  /** @brief Weights handle (mmap/file-backed depending on io implementation). */
  ie_weights_t w;

  /** @brief GPT-OSS tokenizer handle. */
  ie_tok_gptoss_t *tok;

  /** @brief Model hyperparameters. */
  ie_gptoss_hparams_t hp;

  /** @brief GPT-OSS inference runtime instance. */
  ie_gptoss_infer_t *infer;

  /** @brief Logits buffer of length vocab_size. */
  float *logits;

  /** @brief Number of logits entries (vocab size). */
  size_t logits_len;

  /** @brief Sampler scratch: permutation indices. */
  uint32_t *scratch_idx;

  /** @brief Sampler scratch: probabilities. */
  float *scratch_prob;

  /** @brief Capacity of sampler scratch buffers (in entries). */
  size_t scratch_cap;

  /** @brief Sampling configuration. */
  ie_sample_cfg_t sample_cfg;

  /** @brief RNG state for sampling. */
  ie_rng_t rng;
};

/* ============================================================================
 * Small helpers
 * ========================================================================== */

/**
 * @brief Join two path segments using a single slash when needed.
 *
 * @details
 * This helper is intentionally minimal and ASCII-only:
 *  - Copies @p a into @p dst.
 *  - Adds a slash if @p a is non-empty and does not already end with one.
 *  - Appends @p b.
 *  - Always NUL-terminates @p dst.
 *
 * @param dst Output buffer (must not be NULL).
 * @param cap Output buffer capacity in bytes (must be > 0).
 * @param a   First path segment (may be NULL).
 * @param b   Second path segment (may be NULL).
 */
static void join_path_(char *dst, size_t cap, const char *a, const char *b) {
  if (!dst || cap == 0) return;

  size_t i = 0;
  if (a) {
    for (; a[i] && i + 1 < cap; ++i) dst[i] = a[i];
  }

  const int need_slash = (i > 0 && dst[i - 1] != '/');
  if (need_slash && i + 1 < cap) dst[i++] = '/';

  if (b) {
    size_t j = 0;
    for (; b[j] && i + 1 < cap; ++j, ++i) dst[i] = b[j];
  }

  dst[i < cap ? i : (cap - 1)] = '\0';
}

/**
 * @brief Get an environment flag as 0/1 with a default.
 *
 * @details
 * Accepts typical boolean spellings: 0/1, false/true, no/yes, off/on.
 * Unknown values are treated as "enabled" to avoid silent disabling.
 *
 * @param name Environment variable name (must not be NULL).
 * @param default_value Value returned when unset/empty.
 * @return 0 or 1.
 */
static int env_flag_(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 ||
      strcmp(v, "no") == 0 || strcmp(v, "NO") == 0 || strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0) {
    return 0;
  }
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 ||
      strcmp(v, "yes") == 0 || strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 || strcmp(v, "ON") == 0) {
    return 1;
  }
  return 1;
}

/**
 * @brief Get an environment integer with a default and basic clamping.
 *
 * @param name Environment variable name (must not be NULL).
 * @param default_value Default value when unset/empty/invalid.
 * @param min_value Minimum allowed return value.
 * @param max_value Maximum allowed return value.
 * @return Parsed and clamped integer value.
 */
static int env_int_(const char *name, int default_value, int min_value, int max_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (!end || end == v) return default_value;
  if (x < (long)min_value) x = (long)min_value;
  if (x > (long)max_value) x = (long)max_value;
  return (int)x;
}

/**
 * @brief Convert public status codes to stable strings for logs.
 *
 * @param st Status code.
 * @return A constant string.
 */
static const char *status_str_(ie_status_t st) {
  switch (st) {
    case IE_OK: return "IE_OK";
    case IE_ERR_BADARG: return "IE_ERR_BADARG";
    case IE_ERR_OOM: return "IE_ERR_OOM";
    case IE_ERR_MODEL: return "IE_ERR_MODEL";
    case IE_ERR_UNSUPPORTED: return "IE_ERR_UNSUPPORTED";
    case IE_ERR_INTERNAL: return "IE_ERR_INTERNAL";
    default: return "IE_STATUS_UNKNOWN";
  }
}

/**
 * @brief Emit a one-line summary of loaded hyperparameters.
 *
 * @param hp Hyperparameter struct (may be NULL).
 */
static void log_hparams_(const ie_gptoss_hparams_t *hp) {
  if (!hp) return;
  ie_log_info(
      "hparams: layers=%" PRIu32 " d_model=%" PRIu32 " heads=%" PRIu32 " kv_heads=%" PRIu32
      " head_dim=%" PRIu32 " vocab=%" PRIu32 " max_seq=%" PRIu32,
      hp->n_layers,
      hp->d_model,
      hp->n_heads,
      hp->n_kv_heads,
      hp->d_head,
      hp->vocab_size,
      hp->max_seq);
}

/**
 * @brief Attempt to correct head_dim using the first-layer Q projection tensor shape.
 *
 * @details
 * Some model bundles may ship inconsistent or missing head_dim in config metadata.
 * This routine consults tensor_map.json (if present) and inspects the shape of:
 *   "model.layers.0.self_attn.q_proj.weight"
 *
 * If it can infer a better head_dim, it updates @p hp in-place and logs a warning.
 *
 * Best-effort behavior:
 *  - If tensor_map.json is missing/unreadable, it does nothing.
 *  - If the tensor is missing or ambiguous, it does nothing.
 *
 * @param model_dir Model directory containing tensor_map.json (must not be NULL).
 * @param hp Hyperparameter struct to potentially modify (must not be NULL).
 */
static void maybe_correct_head_dim_(const char *model_dir, ie_gptoss_hparams_t *hp) {
  if (!model_dir || !hp) return;
  if (hp->n_heads == 0) return;

  char path[4096];
  join_path_(path, sizeof(path), model_dir, "tensor_map.json");

  tensor_map_t map;
  if (tensor_map_load(path, &map) != 0) return;

  const tensor_desc_t *q = tensor_map_find(&map, "model.layers.0.self_attn.q_proj.weight");
  if (q && q->ndim >= 2 && q->shape) {
    const uint32_t n_heads = hp->n_heads;
    const uint32_t q0 = q->shape[0];
    const uint32_t q1 = q->shape[1];

    uint32_t new_d_head = hp->d_head;

    /* Prefer the dimension that cleanly divides by n_heads and is not equal to d_model. */
    if (q0 != 0 && (q0 % n_heads) == 0 && q0 != hp->d_model) {
      new_d_head = q0 / n_heads;
    } else if (q1 != 0 && (q1 % n_heads) == 0 && q1 != hp->d_model) {
      new_d_head = q1 / n_heads;
    }

    if (new_d_head != 0 && new_d_head != hp->d_head) {
      ie_log_warn("correcting head_dim from %" PRIu32 " to %" PRIu32 " using %s (shape=[%" PRIu32 ",%" PRIu32 "])",
                  hp->d_head,
                  new_d_head,
                  "model.layers.0.self_attn.q_proj.weight",
                  q0,
                  q1);
      hp->d_head = new_d_head;
    }
  }

  tensor_map_free(&map);
}

/**
 * @brief Convert internal return codes into stable @ref ie_status_t values.
 *
 * @details
 * Internal subsystems sometimes return negative errno-like codes. This helper maps
 * a small known subset to stable public statuses while defaulting to IE_ERR_INTERNAL.
 *
 * Current mapping:
 *  - 0    -> IE_OK
 *  - -12  -> IE_ERR_OOM
 *  - -5   -> IE_ERR_OOM (some subsystems use -5 for allocation failures)
 *  - else -> IE_ERR_INTERNAL
 *
 * @param rc Internal return code.
 * @return Public status code.
 */
static ie_status_t status_from_rc_(int rc) {
  if (rc == 0) return IE_OK;
  if (rc == -12 || rc == -5) return IE_ERR_OOM;
  return IE_ERR_INTERNAL;
}

/**
 * @brief Log a short, safe prompt preview for diagnostics.
 *
 * @details
 * Logs at most 96 bytes, replacing embedded newlines/tabs with spaces.
 *
 * @param prompt Prompt string (may be NULL).
 */
static void log_prompt_preview_(const char *prompt) {
  if (!prompt) {
    ie_log_info("prompt: (null)");
    return;
  }

  char tmp[128];
  size_t n = 0;
  for (; prompt[n] && n + 1 < sizeof(tmp) && n < 96; ++n) {
    char ch = prompt[n];
    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    tmp[n] = ch;
  }
  tmp[n] = '\0';

  ie_log_info("prompt: bytes=%zu preview=\"%s%s\"", strlen(prompt), tmp, (prompt[n] ? "..." : ""));
}

/* ============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Create an inference engine instance.
 *
 * @details
 * The create path performs:
 *  - Device selection via @p device string
 *  - Tokenizer open (tokenizer.json under model_dir)
 *  - Weights open (model.ie.json under model_dir)
 *  - Hyperparameter load
 *  - Head dimension correction heuristic (best-effort)
 *  - GPT-OSS runtime creation
 *  - Scratch buffer allocation (logits and sampler scratch)
 *  - Sampler config initialization and RNG seeding
 *
 * @param p         Optional engine parameters (may be NULL).
 * @param device    Device selector string ("cpu", "cuda"); NULL defaults to "cpu".
 * @param model_dir Path to the model directory containing assets (must not be NULL).
 * @param out       Output pointer receiving the created engine on success (must not be NULL).
 * @return IE_OK on success, otherwise a non-OK status.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out) {
  if (!model_dir || !out) {
    ie_log_error("ie_engine_create: bad args (model_dir=%p out=%p)", (const void *)model_dir, (void *)out);
    return IE_ERR_BADARG;
  }
  *out = NULL;

  const char *dev_str = device ? device : "cpu";
  ie_log_info("ie_engine_create: begin (device=%s model_dir=%s)", dev_str, model_dir);

  if (p) {
    ie_log_info("ie_engine_create: params precision=%s sparsity=%s affinity=%s pretranspose=%s prefetch=%s threads=%d",
                (p->precision ? p->precision : "(null)"),
                (p->sparsity ? p->sparsity : "(null)"),
                (p->affinity ? p->affinity : "(null)"),
                (p->pretranspose ? p->pretranspose : "(null)"),
                (p->prefetch ? p->prefetch : "(null)"),
                p->threads);
  }

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) {
    ie_log_error("ie_engine_create: OOM allocating engine struct");
    return IE_ERR_OOM;
  }

  /* ---- device ---- */
  ie_log_info("ie_engine_create: device_create (device=%s)", dev_str);
  const ie_device_kind_t kind = ie_device_kind_from_str(dev_str);
  const int dev_rc = ie_device_create(kind, &e->dev);
  if (dev_rc != 0 || !e->dev) {
    ie_log_error("ie_engine_create: device_create failed (rc=%d device=%s)", dev_rc, dev_str);
    free(e);
    return IE_ERR_UNSUPPORTED;
  }

  /* ---- tokenizer ---- */
  const char *tok_path = NULL;
  char tok_path_buf[4096];
  if (p && p->tokenizer_path && p->tokenizer_path[0] != '\0') {
    tok_path = p->tokenizer_path;
  } else {
    join_path_(tok_path_buf, sizeof(tok_path_buf), model_dir, "tokenizer.json");
    tok_path = tok_path_buf;
  }
  ie_log_info("ie_engine_create: tokenizer_open (path=%s)", tok_path);

  const int tok_rc = ie_tok_gptoss_open(tok_path, &e->tok);
  if (tok_rc != 0 || !e->tok) {
    ie_log_error("ie_engine_create: tokenizer_open failed (rc=%d path=%s)", tok_rc, tok_path);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  const uint32_t tok_vocab = ie_tok_gptoss_vocab_size(e->tok);
  ie_log_info("ie_engine_create: tokenizer_open ok (vocab=%" PRIu32 ")", tok_vocab);

  /* ---- weights ---- */
  const char *weights_json = NULL;
  char weights_json_buf[4096];
  const char *bin_override = NULL;
  if (p && p->weights_json_path && p->weights_json_path[0] != '\0') {
    weights_json = p->weights_json_path;
  } else {
    join_path_(weights_json_buf, sizeof(weights_json_buf), model_dir, "model.ie.json");
    weights_json = weights_json_buf;
  }
  if (p && p->weights_bin_path && p->weights_bin_path[0] != '\0') {
    bin_override = p->weights_bin_path;
  }
  ie_log_info("ie_engine_create: weights_open (json=%s bin_override=%s)",
              weights_json,
              bin_override ? bin_override : "(null)");

  const int w_rc = ie_weights_open(weights_json, bin_override, &e->w);
  if (w_rc != 0) {
    ie_log_error("ie_engine_create: weights_open failed (rc=%d json=%s)", w_rc, weights_json);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  ie_log_info("ie_engine_create: weights_open ok (dtype=%s bin=%s bin_size_bytes=%zu dedup=%d)",
              e->w.dtype,
              e->w.weights_path,
              e->w.bin_size_bytes,
              e->w.is_dedup ? 1 : 0);

  /* ---- hparams ---- */
  ie_log_info("ie_engine_create: hparams_load (dir=%s)", model_dir);
  const int hp_rc = gptoss_hparams_load(model_dir, &e->hp);
  if (hp_rc != 0) {
    ie_log_error("ie_engine_create: hparams_load failed (rc=%d dir=%s)", hp_rc, model_dir);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  log_hparams_(&e->hp);

  if (tok_vocab != 0 && e->hp.vocab_size != 0 && tok_vocab != e->hp.vocab_size) {
    ie_log_warn("ie_engine_create: tokenizer vocab (%" PRIu32 ") != hparams vocab (%" PRIu32 ")",
                tok_vocab,
                e->hp.vocab_size);
  }

  /* Best-effort correction if asset metadata is inconsistent. */
  maybe_correct_head_dim_(model_dir, &e->hp);

  /* ---- runtime ---- */
  ie_log_info("ie_engine_create: infer_create");
  const int inf_rc = ie_gptoss_infer_create(e->dev, &e->w, &e->hp, &e->infer);
  if (inf_rc != 0 || !e->infer) {
    ie_log_error("ie_engine_create: infer_create failed (rc=%d)", inf_rc);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_INTERNAL;
  }

  /* ---- scratch ---- */
  e->logits_len = (size_t)e->hp.vocab_size;
  if (e->logits_len == 0) {
    ie_log_error("ie_engine_create: invalid vocab_size=0; cannot allocate logits");
    ie_gptoss_infer_destroy(e->infer);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  const size_t bytes_logits = e->logits_len * sizeof(float);
  const size_t bytes_idx = e->logits_len * sizeof(uint32_t);
  const size_t bytes_prob = e->logits_len * sizeof(float);

  ie_log_info("ie_engine_create: allocating scratch (logits=%zu bytes idx=%zu bytes prob=%zu bytes)",
              bytes_logits,
              bytes_idx,
              bytes_prob);

  e->logits = (float *)malloc(bytes_logits);
  e->scratch_idx = (uint32_t *)malloc(bytes_idx);
  e->scratch_prob = (float *)malloc(bytes_prob);
  e->scratch_cap = e->logits_len;

  if (!e->logits || !e->scratch_idx || !e->scratch_prob) {
    ie_log_error("ie_engine_create: OOM allocating scratch (logits=%p idx=%p prob=%p)",
                 (void *)e->logits,
                 (void *)e->scratch_idx,
                 (void *)e->scratch_prob);
    free(e->scratch_prob);
    free(e->scratch_idx);
    free(e->logits);
    ie_gptoss_infer_destroy(e->infer);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_OOM;
  }

  /* ---- default sampling config ---- */
  e->sample_cfg.kind = IE_SAMPLE_TOPP;
  e->sample_cfg.temperature = 0.8f;
  e->sample_cfg.top_p = 0.95f;
  e->sample_cfg.top_k = 0;
  e->sample_cfg.disallow_token0 = 1;

  ie_log_info("ie_engine_create: sampling defaults (kind=%d temp=%.4f top_p=%.4f top_k=%d disallow_token0=%d)",
              (int)e->sample_cfg.kind,
              (double)e->sample_cfg.temperature,
              (double)e->sample_cfg.top_p,
              (int)e->sample_cfg.top_k,
              (int)e->sample_cfg.disallow_token0);

  /* ---- RNG ---- */
  ie_rng_init(&e->rng, 1u);
  ie_log_info("ie_engine_create: RNG seeded (seed=%u)", 1u);

  *out = e;
  ie_log_info("ie_engine_create: success (engine=%p)", (void *)e);
  return IE_OK;
}

/**
 * @brief Destroy an engine instance and release all resources.
 *
 * @details
 * This function is safe to call with NULL. For non-NULL engine objects, it releases
 * resources in a defensive order:
 *  - Scratch buffers
 *  - Runtime
 *  - Weights
 *  - Tokenizer
 *  - Device
 *  - Engine struct memory
 *
 * @param e Engine instance to destroy (may be NULL).
 */
void ie_engine_destroy(ie_engine_t *e) {
  if (!e) return;

  ie_log_info("ie_engine_destroy: begin (engine=%p)", (void *)e);

  free(e->scratch_prob);
  free(e->scratch_idx);
  free(e->logits);

  if (e->infer) {
    ie_log_info("ie_engine_destroy: infer_destroy");
    ie_gptoss_infer_destroy(e->infer);
  }

  ie_log_info("ie_engine_destroy: weights_close");
  ie_weights_close(&e->w);

  if (e->tok) {
    ie_log_info("ie_engine_destroy: tokenizer_close");
    ie_tok_gptoss_close(e->tok);
  }
  if (e->dev) {
    ie_log_info("ie_engine_destroy: device_destroy");
    ie_device_destroy(e->dev);
  }

  free(e);
  ie_log_info("ie_engine_destroy: done");
}

/**
 * @brief Generate tokens from a prompt using GPT-OSS inference and sampling.
 *
 * @details
 * This function:
 *  - Encodes @p prompt into token ids using the GPT-OSS tokenizer.
 *  - Initializes a KV cache for all layers.
 *  - Runs a prefill pass over the prompt to populate KV and compute next-token logits.
 *  - Repeats up to @p max_new times:
 *      - Sample next token from logits
 *      - Run one decode step with that token to update KV and logits
 *
 * Output contract:
 *  - Writes up to @p max_new integers into @p out_tokens.
 *  - Returns the number of written tokens via @p out_n_tokens.
 *
 * @param e            Engine instance (must not be NULL).
 * @param prompt       UTF-8 prompt text (must not be NULL).
 * @param max_new      Maximum number of new tokens to generate.
 * @param out_tokens   Output array of length at least max_new (required if max_new > 0).
 * @param out_n_tokens Output count of generated tokens (must not be NULL).
 * @return IE_OK on success, otherwise a non-OK status.
 */
ie_status_t ie_engine_generate(const ie_engine_t *e,
                               const char *prompt,
                               size_t max_new,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  if (!e || !prompt || !out_n_tokens) {
    ie_log_error("ie_engine_generate: bad args (e=%p prompt=%p out_n_tokens=%p)",
                 (const void *)e,
                 (const void *)prompt,
                 (void *)out_n_tokens);
    return IE_ERR_BADARG;
  }
  if (max_new > 0 && !out_tokens) {
    ie_log_error("ie_engine_generate: bad args (max_new=%zu out_tokens=NULL)", max_new);
    return IE_ERR_BADARG;
  }

  *out_n_tokens = 0;
  if (max_new == 0) {
    ie_log_info("ie_engine_generate: max_new=0; nothing to do");
    return IE_OK;
  }

  ie_log_info("ie_engine_generate: begin (engine=%p max_new=%zu)", (const void *)e, max_new);
  log_prompt_preview_(prompt);

  const int log_tokens = env_flag_("IE_API_LOG_TOKENS", 0);
  const int log_every = env_int_("IE_API_LOG_EVERY", 0, 0, 1000000);
  const int dbg_decode = env_flag_("IE_API_DEBUG_DECODE", 0);
  const int dbg_decode_every = env_int_("IE_API_DEBUG_DECODE_EVERY", 0, 0, 1000000);

  /* ---- encode prompt (size query) ---- */
  uint32_t need = 0;
  int rc = ie_tok_gptoss_encode(e->tok, prompt, NULL, &need);
  if (rc != 0) {
    ie_log_error("ie_engine_generate: tokenizer encode size-query failed (rc=%d)", rc);
    return IE_ERR_MODEL;
  }

  ie_log_info("ie_engine_generate: tokenizer size-query ok (need_tokens=%" PRIu32 ")", need);

  uint32_t n_prompt = (need == 0) ? 1u : need;
  uint32_t *prompt_ids = (uint32_t *)malloc((size_t)n_prompt * sizeof(uint32_t));
  if (!prompt_ids) {
    ie_log_error("ie_engine_generate: OOM allocating prompt_ids (n_prompt=%" PRIu32 ")", n_prompt);
    return IE_ERR_OOM;
  }

  if (need == 0) {
    prompt_ids[0] = 0;
    n_prompt = 1;
    ie_log_warn("ie_engine_generate: prompt encoded to empty; forcing token_id=0");
  } else {
    uint32_t inout = n_prompt;
    rc = ie_tok_gptoss_encode(e->tok, prompt, prompt_ids, &inout);
    if (rc != 0) {
      ie_log_error("ie_engine_generate: tokenizer encode failed (rc=%d capacity=%" PRIu32 ")", rc, n_prompt);
      free(prompt_ids);
      return IE_ERR_MODEL;
    }
    n_prompt = inout;
    if (n_prompt == 0) {
      prompt_ids[0] = 0;
      n_prompt = 1;
      ie_log_warn("ie_engine_generate: encode returned 0 tokens; forcing token_id=0");
    }
  }

  ie_log_info("ie_engine_generate: encoded prompt tokens=%" PRIu32, n_prompt);

  if (log_tokens) {
    const uint32_t cap = (n_prompt < 32u) ? n_prompt : 32u;
    char line[512];
    size_t pos = 0;

    pos += (size_t)snprintf(line + pos, sizeof(line) - pos, "prompt_ids[0..%u):", (unsigned)cap);
    for (uint32_t i = 0; i < cap && pos + 16 < sizeof(line); ++i) {
      pos += (size_t)snprintf(line + pos, sizeof(line) - pos, " %u", (unsigned)prompt_ids[i]);
    }
    if (cap < n_prompt && pos + 8 < sizeof(line)) {
      (void)snprintf(line + pos, sizeof(line) - pos, " ...");
    }
    ie_log_info("%s", line);
  }

  if (e->hp.n_layers == 0) {
    ie_log_error("ie_engine_generate: invalid hparams (n_layers=0)");
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  /* ---- KV cache ---- */
  const size_t n_layers = (size_t)e->hp.n_layers;
  ie_kv_cache *kv_layers = (ie_kv_cache *)calloc(n_layers, sizeof(*kv_layers));
  if (!kv_layers) {
    ie_log_error("ie_engine_generate: OOM allocating kv_layers (n_layers=%zu)", n_layers);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  ie_kv_opts kv_opts;
  memset(&kv_opts, 0, sizeof(kv_opts));
  kv_opts.storage = IE_KV_STORAGE_F32;
  kv_opts.heads = (int32_t)e->hp.n_kv_heads;
  kv_opts.head_dim = (int32_t)e->hp.d_head;

  int32_t want_seq = (int32_t)n_prompt + (int32_t)max_new + 8;
  if (want_seq < 16) want_seq = 16;

  int32_t hp_seq = (int32_t)e->hp.max_seq;
  if (hp_seq <= 0) hp_seq = want_seq;

  kv_opts.max_seq = (want_seq < hp_seq) ? want_seq : hp_seq;

  ie_log_info("ie_engine_generate: kv_opts heads=%" PRIi32 " head_dim=%" PRIi32 " want_seq=%" PRIi32 " hp_seq=%" PRIi32 " max_seq=%" PRIi32,
              kv_opts.heads,
              kv_opts.head_dim,
              want_seq,
              hp_seq,
              kv_opts.max_seq);

  if (kv_opts.heads <= 0 || kv_opts.head_dim <= 0 || kv_opts.max_seq <= 0) {
    ie_log_error("ie_engine_generate: invalid kv_opts (heads=%" PRIi32 " head_dim=%" PRIi32 " max_seq=%" PRIi32 ")",
                 kv_opts.heads,
                 kv_opts.head_dim,
                 kv_opts.max_seq);
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  if (ie_kv_init_layers(kv_layers, (int)e->hp.n_layers, &kv_opts) != 0) {
    ie_log_error("ie_engine_generate: kv_init_layers failed (n_layers=%" PRIu32 ")", e->hp.n_layers);
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  /* ---- prefill ---- */
  ie_log_info("ie_engine_generate: prefill begin (n_prompt=%" PRIu32 " logits_len=%zu)", n_prompt, e->logits_len);

  rc = ie_gptoss_infer_prefill(e->infer, kv_layers, prompt_ids, n_prompt, e->logits);
  if (rc != 0) {
    const ie_status_t st = status_from_rc_(rc);
    ie_log_error("ie_engine_generate: prefill failed (rc=%d -> %s)", rc, status_str_(st));
    ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
    free(kv_layers);
    free(prompt_ids);
    return st;
  }

  ie_log_info("ie_engine_generate: prefill ok");

  /* ---- decode loop ---- */
  size_t produced = 0;
  for (; produced < max_new; ++produced) {
    uint32_t next = 0;

    rc = ie_sample_next(e->logits,
                        e->logits_len,
                        &e->sample_cfg,
                        (ie_rng_t *)&e->rng,
                        e->scratch_idx,
                        e->scratch_prob,
                        e->scratch_cap,
                        &next);
    if (rc != 0) {
      ie_log_error("ie_engine_generate: sample_next failed at step=%zu (rc=%d). Stopping early.", produced, rc);
      break;
    }

    out_tokens[produced] = (int)next;

    if (log_every > 0 && (produced % (size_t)log_every) == 0) {
      ie_log_info("ie_engine_generate: step=%zu next_token=%" PRIu32, produced, next);
    }

    rc = ie_gptoss_infer_step(e->infer, kv_layers, next, e->logits);
    if (rc != 0) {
      const ie_status_t st = status_from_rc_(rc);
      ie_log_error("ie_engine_generate: infer_step failed at step=%zu token=%" PRIu32 " (rc=%d -> %s)",
                   produced,
                   next,
                   rc,
                   status_str_(st));
      ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
      free(kv_layers);
      free(prompt_ids);
      return st;
    }

    if (dbg_decode && dbg_decode_every > 0 && (produced % (size_t)dbg_decode_every) == 0) {
      size_t need_bytes = 0;
      const uint32_t cnt = (uint32_t)(produced + 1);
      const int drc = ie_tok_gptoss_decode(e->tok, (const uint32_t *)out_tokens, cnt, NULL, &need_bytes);
      if (drc == 0 && need_bytes > 0 && need_bytes < (size_t)(1u << 22)) {
        char *txt = (char *)malloc(need_bytes);
        if (txt) {
          size_t inout = need_bytes;
          const int drc2 = ie_tok_gptoss_decode(e->tok, (const uint32_t *)out_tokens, cnt, txt, &inout);
          if (drc2 == 0) {
            ie_log_info("ie_engine_generate: decoded[%u] bytes=%zu preview=\"%s%s\"",
                        cnt,
                        inout,
                        txt,
                        (inout > 120 ? "..." : ""));
          } else {
            ie_log_warn("ie_engine_generate: decode failed (rc=%d) at step=%zu", drc2, produced);
          }
          free(txt);
        }
      } else {
        ie_log_warn("ie_engine_generate: decode size-query failed (rc=%d need_bytes=%zu) at step=%zu",
                    drc,
                    need_bytes,
                    produced);
      }
    }
  }

  *out_n_tokens = produced;

  ie_log_info("ie_engine_generate: done (produced=%zu requested=%zu)", produced, max_new);

  /* ---- cleanup ---- */
  ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
  free(kv_layers);
  free(prompt_ids);

  return IE_OK;
}
