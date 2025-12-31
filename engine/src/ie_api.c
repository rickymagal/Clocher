/**
 * @file ie_api.c
 * @brief Small, stable public API wrapper around the internal engine.
 */

#include "ie_api.h"

#include <inttypes.h>
#include <stdint.h>
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

struct ie_engine {
  ie_device_t *dev;

  ie_weights_t w;

  ie_tok_gptoss_t *tok;

  ie_gptoss_hparams_t hp;
  ie_gptoss_infer_t *infer;

  float *logits;
  size_t logits_len;

  uint32_t *scratch_idx;
  float *scratch_prob;
  size_t scratch_cap;

  ie_sample_cfg_t sample_cfg;
  ie_rng_t rng;
};

static void join_path_(char *dst, size_t cap, const char *a, const char *b) {
  if (!dst || cap == 0) return;

  size_t i = 0;
  if (a) {
    for (; a[i] && i + 1 < cap; ++i) dst[i] = a[i];
  }
  int need_slash = (i > 0 && dst[i - 1] != '/');
  if (need_slash && i + 1 < cap) dst[i++] = '/';

  if (b) {
    size_t j = 0;
    for (; b[j] && i + 1 < cap; ++j, ++i) dst[i] = b[j];
  }
  dst[i < cap ? i : (cap - 1)] = '\0';
}

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
    if (q0 != 0 && (q0 % n_heads) == 0 && q0 != hp->d_model) {
      new_d_head = q0 / n_heads;
    } else if (q1 != 0 && (q1 % n_heads) == 0 && q1 != hp->d_model) {
      new_d_head = q1 / n_heads;
    }

    if (new_d_head != 0 && new_d_head != hp->d_head) {
      fprintf(stderr,
              "warn: correcting head_dim from %" PRIu32 " to %" PRIu32 " using %s\n",
              hp->d_head,
              new_d_head,
              "model.layers.0.self_attn.q_proj.weight");
      hp->d_head = new_d_head;
    }
  }

  tensor_map_free(&map);
}

static ie_status_t status_from_rc_(int rc) {
  if (rc == 0) return IE_OK;
  if (rc == -12) return IE_ERR_OOM;
  return IE_ERR_INTERNAL;
}

ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out) {
  (void)p;

  if (!model_dir || !out) return IE_ERR_BADARG;
  *out = NULL;

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_OOM;

  const char *dev_str = device ? device : "cpu";
  const ie_device_kind_t kind = ie_device_kind_from_str(dev_str);

  if (ie_device_create(kind, &e->dev) != 0 || !e->dev) {
    free(e);
    return IE_ERR_UNSUPPORTED;
  }

  char tok_path[4096];
  join_path_(tok_path, sizeof(tok_path), model_dir, "tokenizer.json");
  if (ie_tok_gptoss_open(tok_path, &e->tok) != 0 || !e->tok) {
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  char weights_json[4096];
  join_path_(weights_json, sizeof(weights_json), model_dir, "model.ie.json");
  if (ie_weights_open(weights_json, NULL, &e->w) != 0) {
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  if (gptoss_hparams_load(model_dir, &e->hp) != 0) {
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  maybe_correct_head_dim_(model_dir, &e->hp);

  if (ie_gptoss_infer_create(e->dev, &e->w, &e->hp, &e->infer) != 0 || !e->infer) {
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_INTERNAL;
  }

  e->logits_len = (size_t)e->hp.vocab_size;
  if (e->logits_len == 0) {
    ie_gptoss_infer_destroy(e->infer);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  e->logits = (float *)malloc(e->logits_len * sizeof(float));
  e->scratch_idx = (uint32_t *)malloc(e->logits_len * sizeof(uint32_t));
  e->scratch_prob = (float *)malloc(e->logits_len * sizeof(float));
  e->scratch_cap = e->logits_len;

  if (!e->logits || !e->scratch_idx || !e->scratch_prob) {
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

  e->sample_cfg.kind = IE_SAMPLE_TOPP;
  e->sample_cfg.temperature = 0.8f;
  e->sample_cfg.top_p = 0.95f;
  e->sample_cfg.top_k = 0;
  e->sample_cfg.disallow_token0 = 1;

  ie_rng_init(&e->rng, 1u);

  *out = e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *e) {
  if (!e) return;

  free(e->scratch_prob);
  free(e->scratch_idx);
  free(e->logits);

  if (e->infer) ie_gptoss_infer_destroy(e->infer);
  ie_weights_close(&e->w);
  if (e->tok) ie_tok_gptoss_close(e->tok);
  if (e->dev) ie_device_destroy(e->dev);

  free(e);
}

ie_status_t ie_engine_generate(const ie_engine_t *e,
                               const char *prompt,
                               size_t max_new,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  if (!e || !prompt || !out_n_tokens) return IE_ERR_BADARG;
  if (max_new > 0 && !out_tokens) return IE_ERR_BADARG;

  *out_n_tokens = 0;
  if (max_new == 0) return IE_OK;

  uint32_t need = 0;
  int rc = ie_tok_gptoss_encode(e->tok, prompt, NULL, &need);
  if (rc != 0) return IE_ERR_MODEL;

  uint32_t n_prompt = need;
  if (n_prompt == 0) n_prompt = 1;

  uint32_t *prompt_ids = (uint32_t *)malloc((size_t)n_prompt * sizeof(uint32_t));
  if (!prompt_ids) return IE_ERR_OOM;

  if (need == 0) {
    prompt_ids[0] = 0;
    n_prompt = 1;
  } else {
    uint32_t inout = n_prompt;
    rc = ie_tok_gptoss_encode(e->tok, prompt, prompt_ids, &inout);
    if (rc != 0) {
      free(prompt_ids);
      return IE_ERR_MODEL;
    }
    n_prompt = inout;
    if (n_prompt == 0) {
      prompt_ids[0] = 0;
      n_prompt = 1;
    }
  }

  if (e->hp.n_layers == 0) {
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  const size_t n_layers = (size_t)e->hp.n_layers;
  ie_kv_cache *kv_layers = (ie_kv_cache *)calloc(n_layers, sizeof(*kv_layers));
  if (!kv_layers) {
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  ie_kv_opts kv_opts;
  memset(&kv_opts, 0, sizeof(kv_opts));
  kv_opts.storage = IE_KV_STORAGE_F32;
  kv_opts.heads = (int32_t)e->hp.n_kv_heads;
  kv_opts.head_dim = (int32_t)e->hp.d_head;
  kv_opts.max_seq = (int32_t)e->hp.max_seq;

  if (kv_opts.heads <= 0 || kv_opts.head_dim <= 0 || kv_opts.max_seq <= 0) {
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  if (ie_kv_init_layers(kv_layers, (int)e->hp.n_layers, &kv_opts) != 0) {
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  rc = ie_gptoss_infer_prefill(e->infer, kv_layers, prompt_ids, n_prompt, e->logits);
  if (rc != 0) {
    ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
    free(kv_layers);
    free(prompt_ids);
    return status_from_rc_(rc);
  }

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
    if (rc != 0) break;

    out_tokens[produced] = (int)next;

    rc = ie_gptoss_infer_step(e->infer, kv_layers, next, e->logits);
    if (rc != 0) {
      produced += 1;
      break;
    }
  }

  *out_n_tokens = produced;

  ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
  free(kv_layers);
  free(prompt_ids);

  return IE_OK;
}
