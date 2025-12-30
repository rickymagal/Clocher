/* ============================================================================
 * File: engine/src/runtime/generate_gptoss.c
 * ============================================================================
 */
/**
 * @file generate_gptoss.c
 * @brief GPT-OSS token generation loop built on prefill/step + sampling.
 *
 * @details
 * This module is a correctness-first implementation of the "outer loop" around
 * the transformer:
 *  - Reads minimal hyperparameters from HuggingFace config.json,
 *  - Encodes the prompt to token IDs,
 *  - Creates KV cache and inference context,
 *  - Runs prefill and incremental step calls,
 *  - Samples tokens from logits using ie_sampling.
 *
 * It is intentionally conservative:
 *  - Allocations happen inside the call (bring-up friendly).
 *  - KV cache uses FP32 storage by default (simple, correct).
 *  - Prompt encoding requires a tokenizer handle.
 *
 * Once the forward pass is real, this file is what lets the CLI and API produce
 * actual token IDs (and optionally decoded text in higher layers).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "generate_gptoss.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_kv_cache.h"

/* ----------------------------- JSON scanning ----------------------------- */

/**
 * @brief Skip ASCII whitespace in a JSON buffer.
 *
 * @param p Current cursor.
 * @return Pointer advanced past whitespace.
 */
static const char *json_skip_ws(const char *p) {
  while (p && *p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  return p;
}

/**
 * @brief Find a JSON key occurrence of the form "key".
 *
 * @details
 * This is a strict, minimal scanner for HuggingFace config.json. It looks for
 * the literal `"key"` substring. It does not attempt to parse arbitrary JSON.
 *
 * @param json JSON buffer.
 * @param key  Key name (without quotes).
 * @return Pointer to the character after the closing quote of the match, or NULL.
 */
static const char *json_find_key(const char *json, const char *key) {
  if (!json || !key || !*key) return NULL;

  const size_t klen = strlen(key);
  const size_t needle_len = klen + 2u; /* quotes */
  char *needle = (char *)malloc(needle_len + 1u);
  if (!needle) return NULL;

  needle[0] = '"';
  memcpy(needle + 1, key, klen);
  needle[1 + klen] = '"';
  needle[needle_len] = '\0';

  const char *p = strstr(json, needle);
  free(needle);
  if (!p) return NULL;

  return p + needle_len;
}

/**
 * @brief Parse an unsigned integer value after a key.
 *
 * @param json JSON buffer.
 * @param key  Key name.
 * @param out  Output value.
 * @return 0 on success, non-zero on failure.
 */
static int json_get_u32(const char *json, const char *key, uint32_t *out) {
  if (!json || !key || !out) return -1;

  const char *p = json_find_key(json, key);
  if (!p) return -1;

  p = json_skip_ws(p);
  if (*p != ':') return -1;
  p = json_skip_ws(p + 1);

  /* Optional sign not allowed for u32. */
  if (!isdigit((unsigned char)*p)) return -1;

  unsigned long v = 0ul;
  while (isdigit((unsigned char)*p)) {
    v = v * 10ul + (unsigned long)(*p - '0');
    ++p;
  }

  if (v > 0xFFFFFFFFul) return -1;
  *out = (uint32_t)v;
  return 0;
}

/**
 * @brief Read an entire file into memory (NUL-terminated).
 *
 * @param path File path.
 * @param out_buf Receives allocated buffer (caller frees).
 * @param out_len Receives length excluding NUL.
 * @return 0 on success, non-zero on error.
 */
static int read_entire_file(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf || !out_len) return -1;
  *out_buf = NULL;
  *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -1;

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return -1;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }

  size_t sz = (size_t)n;
  char *buf = (char *)malloc(sz + 1u);
  if (!buf) {
    fclose(f);
    return -1;
  }

  size_t rd = fread(buf, 1, sz, f);
  fclose(f);
  if (rd != sz) {
    free(buf);
    return -1;
  }

  buf[sz] = '\0';
  *out_buf = buf;
  *out_len = sz;
  return 0;
}

/**
 * @brief Load minimal GPT-OSS hyperparameters from HF config.json.
 *
 * @details
 * Keys commonly present in HuggingFace GPT-style configs:
 *  - num_hidden_layers
 *  - num_attention_heads
 *  - num_key_value_heads (optional; defaults to num_attention_heads)
 *  - hidden_size
 *  - intermediate_size
 *  - vocab_size
 *  - max_position_embeddings
 *
 * This function tolerates extra fields and whitespace but is otherwise strict.
 *
 * @param model_dir Model directory containing config.json.
 * @param out_hp Output hyperparameters.
 * @return 0 on success, non-zero on error.
 */
static int load_hparams_from_config(const char *model_dir, ie_gptoss_hparams_t *out_hp) {
  if (!model_dir || !out_hp) return -1;

  char path[2048];
  int n = snprintf(path, sizeof(path), "%s/config.json", model_dir);
  if (n <= 0 || (size_t)n >= sizeof(path)) return -1;

  char *json = NULL;
  size_t json_len = 0;
  if (read_entire_file(path, &json, &json_len) != 0 || !json || json_len == 0) {
    free(json);
    return -1;
  }

  uint32_t n_layers = 0, n_heads = 0, n_kv_heads = 0;
  uint32_t d_model = 0, d_ff = 0, vocab = 0, max_seq = 0;

  /* Required fields (for this snapshot). */
  if (json_get_u32(json, "num_hidden_layers", &n_layers) != 0) goto fail;
  if (json_get_u32(json, "num_attention_heads", &n_heads) != 0) goto fail;
  if (json_get_u32(json, "hidden_size", &d_model) != 0) goto fail;
  if (json_get_u32(json, "intermediate_size", &d_ff) != 0) goto fail;
  if (json_get_u32(json, "vocab_size", &vocab) != 0) goto fail;

  /* Optional fields with fallbacks. */
  if (json_get_u32(json, "num_key_value_heads", &n_kv_heads) != 0) {
    n_kv_heads = n_heads;
  }
  if (json_get_u32(json, "max_position_embeddings", &max_seq) != 0) {
    /* Conservative fallback; callers should provide a real config.json. */
    max_seq = 2048u;
  }

  if (n_layers == 0u || n_heads == 0u || d_model == 0u || d_ff == 0u || vocab == 0u) goto fail;
  if ((d_model % n_heads) != 0u) goto fail;
  if (n_kv_heads == 0u || n_kv_heads > n_heads) goto fail;
  if ((n_heads % n_kv_heads) != 0u) goto fail;

  out_hp->n_layers = n_layers;
  out_hp->n_heads = n_heads;
  out_hp->n_kv_heads = n_kv_heads;
  out_hp->d_model = d_model;
  out_hp->d_head = d_model / n_heads;
  out_hp->d_ff = d_ff;
  out_hp->vocab_size = vocab;
  out_hp->max_seq = max_seq;

  free(json);
  return 0;

fail:
  free(json);
  return -1;
}

/* ---------------------------- Public functions --------------------------- */

void ie_generate_gptoss_default_sample_cfg(ie_sample_cfg_t *out_cfg) {
  if (!out_cfg) return;
  memset(out_cfg, 0, sizeof(*out_cfg));
  out_cfg->temperature = 1.0f;
  out_cfg->top_k = 0;
  out_cfg->top_p = 0.0f;
  out_cfg->greedy = 1;
  out_cfg->disallow_token0 = 0;
}

int ie_generate_gptoss(const ie_device_t *dev,
                       const ie_weights_t *w,
                       const char *model_dir,
                       const ie_tok_gptoss_t *tok,
                       const char *prompt,
                       uint32_t max_new,
                       const ie_sample_cfg_t *sample_cfg,
                       uint64_t seed,
                       int *out_tokens,
                       uint32_t *out_n_tokens) {
  if (!dev || !w || !model_dir || !tok || !prompt || (!out_tokens && max_new > 0u) || !out_n_tokens) {
    return IE_GEN_GPTOSS_ERR_ARGS;
  }

  *out_n_tokens = 0;
  if (max_new == 0u) return IE_GEN_GPTOSS_OK;

  ie_gptoss_hparams_t hp;
  memset(&hp, 0, sizeof(hp));
  if (load_hparams_from_config(model_dir, &hp) != 0) {
    return IE_GEN_GPTOSS_ERR_IO;
  }

  /* Prepare sampling config and RNG. */
  ie_sample_cfg_t cfg_local;
  if (!sample_cfg) {
    ie_generate_gptoss_default_sample_cfg(&cfg_local);
    sample_cfg = &cfg_local;
  }

  ie_rng_t rng;
  ie_rng_init(&rng, seed);

  /* Encode prompt. */
  size_t prompt_bytes = strlen(prompt);
  uint32_t cap_ids = (prompt_bytes > 0u) ? (uint32_t)(prompt_bytes + 8u) : 8u;
  if (cap_ids < 8u) cap_ids = 8u;
  if (cap_ids > 1u << 20) cap_ids = 1u << 20; /* hard safety cap */

  uint32_t *prompt_ids = (uint32_t *)malloc((size_t)cap_ids * sizeof(uint32_t));
  if (!prompt_ids) return IE_GEN_GPTOSS_ERR_NOMEM;

  uint32_t n_prompt = cap_ids;
  int tok_rc = ie_tok_gptoss_encode(tok, prompt, prompt_ids, &n_prompt);
  if (tok_rc != IE_TOK_GPTOSS_OK) {
    free(prompt_ids);
    return IE_GEN_GPTOSS_ERR_IO;
  }

  /* Allocate logits and sampling scratch. */
  float *logits = (float *)malloc((size_t)hp.vocab_size * sizeof(float));
  uint32_t *idx_scratch = (uint32_t *)malloc((size_t)hp.vocab_size * sizeof(uint32_t));
  float *prob_scratch = (float *)malloc((size_t)hp.vocab_size * sizeof(float));
  if (!logits || !idx_scratch || !prob_scratch) {
    free(prob_scratch);
    free(idx_scratch);
    free(logits);
    free(prompt_ids);
    return IE_GEN_GPTOSS_ERR_NOMEM;
  }

  /* Create inference context. */
  ie_gptoss_infer_t *infer = NULL;
  if (ie_gptoss_infer_create(dev, w, &hp, &infer) != 0 || !infer) {
    free(prob_scratch);
    free(idx_scratch);
    free(logits);
    free(prompt_ids);
    return IE_GEN_GPTOSS_ERR_UNSUPPORTED;
  }

  /* Create KV cache (FP32, correctness-first). */
  ie_kv_cache kv;
  memset(&kv, 0, sizeof(kv));
  if (ie_kv_init(&kv,
                 (int)hp.n_kv_heads,
                 (int)hp.d_head,
                 (int)hp.max_seq,
                 IE_KV_STORAGE_F32,
                 (size_t)0,
                 1,
                 IE_FP8_E4M3) != 0) {
    ie_gptoss_infer_destroy(infer);
    free(prob_scratch);
    free(idx_scratch);
    free(logits);
    free(prompt_ids);
    return IE_GEN_GPTOSS_ERR_NOMEM;
  }

  /* Prefill or bootstrap. */
  int rc = 0;
  if (n_prompt > 0u) {
    rc = ie_gptoss_infer_prefill(infer, &kv, prompt_ids, n_prompt, logits);
  } else {
    /* Empty prompt: bootstrap with token 0 as a minimal, deterministic start. */
    rc = ie_gptoss_infer_step(infer, &kv, 0u, logits);
  }

  if (rc != 0) {
    ie_kv_free(&kv);
    ie_gptoss_infer_destroy(infer);
    free(prob_scratch);
    free(idx_scratch);
    free(logits);
    free(prompt_ids);
    return IE_GEN_GPTOSS_ERR_UNSUPPORTED;
  }

  /* Generate tokens. */
  for (uint32_t i = 0; i < max_new; ++i) {
    uint32_t next_id = 0u;
    const int s_rc = ie_sample_next(logits,
                                   (size_t)hp.vocab_size,
                                   sample_cfg,
                                   &rng,
                                   idx_scratch,
                                   prob_scratch,
                                   (size_t)hp.vocab_size,
                                   &next_id);
    if (s_rc != 0) {
      break;
    }

    out_tokens[i] = (int)next_id;
    *out_n_tokens = i + 1u;

    /* Compute logits for the following step unless this was the last token. */
    if (i + 1u < max_new) {
      if (ie_gptoss_infer_step(infer, &kv, next_id, logits) != 0) {
        break;
      }
    }
  }

  ie_kv_free(&kv);
  ie_gptoss_infer_destroy(infer);
  free(prob_scratch);
  free(idx_scratch);
  free(logits);
  free(prompt_ids);

  return IE_GEN_GPTOSS_OK;
}
