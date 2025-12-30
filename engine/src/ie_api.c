/* ============================================================================
 * File: engine/src/ie_api.c
 * ============================================================================
 */
/**
 * @file ie_api.c
 * @brief Public engine API implementation (model loading + generation entrypoint).
 *
 * @details
 * Responsibilities:
 *  - Create/destroy device backends (CPU/CUDA).
 *  - Load weights (including dedup reconstruction through weights_dedup.c).
 *  - Load tokenizer.json when available for prompt encoding.
 *  - Derive GPT-OSS hyperparameters from model_dir/config.json (with hf/original fallback).
 *  - Run the real decode loop by calling:
 *      - ie_gptoss_infer_prefill() to consume the prompt
 *      - ie_gptoss_infer_step() for each generated token
 *
 * Token decoding into human-readable text is performed by the harness/CLI.
 *
 * Fallback behavior:
 *  - If IE_ALLOW_FAKE_TOKENS=1, the engine can emit deterministic fake token IDs
 *    for plumbing tests when inference is unavailable or fails early.
 */

#include "ie_api.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "ie_device.h"
#include "ie_io.h"
#include "ie_infer.h"
#include "ie_kv_cache.h"
#include "ie_sampling.h"
#include "ie_tokenizer_gptoss.h"

/* ----------------------------- small env helpers ---------------------------- */

static int env_truthy(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !v[0]) return default_value;
  if (strcmp(v, "1") == 0) return 1;
  if (strcmp(v, "true") == 0) return 1;
  if (strcmp(v, "yes") == 0) return 1;
  if (strcmp(v, "on") == 0) return 1;
  if (strcmp(v, "0") == 0) return 0;
  if (strcmp(v, "false") == 0) return 0;
  if (strcmp(v, "no") == 0) return 0;
  if (strcmp(v, "off") == 0) return 0;
  return default_value;
}

static uint32_t env_u32(const char *name, uint32_t default_value) {
  const char *v = getenv(name);
  if (!v || !v[0]) return default_value;
  errno = 0;
  char *end = NULL;
  unsigned long x = strtoul(v, &end, 10);
  if (errno != 0 || end == v || (end && *end != '\0')) return default_value;
  if (x > 0xfffffffful) return default_value;
  return (uint32_t)x;
}

static float env_f32(const char *name, float default_value) {
  const char *v = getenv(name);
  if (!v || !v[0]) return default_value;
  errno = 0;
  char *end = NULL;
  float x = strtof(v, &end);
  if (errno != 0 || end == v || (end && *end != '\0')) return default_value;
  return x;
}

/* ----------------------------- deterministic fake --------------------------- */

static uint32_t fnv1a32_bytes(const void *data, size_t n) {
  const uint8_t *p = (const uint8_t *)data;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;
  }
  return h;
}

static uint32_t fnv1a32_cstr(const char *s) {
  return fnv1a32_bytes(s ? s : "", s ? strlen(s) : 0u);
}

static uint32_t xorshift32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

/* ----------------------------- config.json parse ---------------------------- */

static int read_text_file(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf || !out_len) return -1;
  *out_buf = NULL;
  *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -2;

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -2;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return -2;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -2;
  }

  size_t n = (size_t)sz;
  char *buf = (char *)malloc(n + 1u);
  if (!buf) {
    fclose(f);
    return -3;
  }

  size_t rd = fread(buf, 1, n, f);
  fclose(f);
  if (rd != n) {
    free(buf);
    return -2;
  }

  buf[n] = '\0';
  *out_buf = buf;
  *out_len = n;
  return 0;
}

static int json_find_i64(const char *json, const char *key, int64_t *out) {
  if (!json || !key || !out) return -1;

  char pat[128];
  int n = snprintf(pat, sizeof(pat), "\"%s\"", key);
  if (n <= 0 || (size_t)n >= sizeof(pat)) return 0;

  const char *p = strstr(json, pat);
  if (!p) return 0;

  p += strlen(pat);
  while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  if (*p != ':') return 0;
  ++p;
  while (*p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;

  errno = 0;
  char *end = NULL;
  long long v = strtoll(p, &end, 10);
  if (errno != 0 || end == p) return 0;

  *out = (int64_t)v;
  return 1;
}

static int try_read_config_json(const char *model_dir, char **out_buf, size_t *out_len, char *used_path, size_t used_path_n) {
  if (!model_dir || !out_buf || !out_len) return -1;

  char p0[2048];
  char p1[2048];

  int n0 = snprintf(p0, sizeof(p0), "%s/config.json", model_dir);
  int n1 = snprintf(p1, sizeof(p1), "%s/hf/original/config.json", model_dir);

  if (n0 > 0 && (size_t)n0 < sizeof(p0)) {
    if (read_text_file(p0, out_buf, out_len) == 0) {
      if (used_path && used_path_n) (void)snprintf(used_path, used_path_n, "%s", p0);
      return 0;
    }
  }

  if (n1 > 0 && (size_t)n1 < sizeof(p1)) {
    if (read_text_file(p1, out_buf, out_len) == 0) {
      if (used_path && used_path_n) (void)snprintf(used_path, used_path_n, "%s", p1);
      return 0;
    }
  }

  return -2;
}

static int gptoss_hparams_from_config_json(const char *model_dir,
                                          uint32_t vocab_size_from_tok,
                                          ie_gptoss_hparams_t *out_hp) {
  if (!model_dir || !out_hp) return -1;
  memset(out_hp, 0, sizeof(*out_hp));

  char *buf = NULL;
  size_t len = 0;
  char used_path[2048];
  used_path[0] = '\0';

  if (try_read_config_json(model_dir, &buf, &len, used_path, sizeof(used_path)) != 0) {
    (void)fprintf(stderr, "ie_engine_create: could not read config.json at %s/config.json or %s/hf/original/config.json\n",
                  model_dir, model_dir);
    return -2;
  }

  int64_t v = 0;

  if (json_find_i64(buf, "num_hidden_layers", &v) == 1) out_hp->n_layers = (uint32_t)v;
  else if (json_find_i64(buf, "n_layer", &v) == 1) out_hp->n_layers = (uint32_t)v;

  if (json_find_i64(buf, "num_attention_heads", &v) == 1) out_hp->n_heads = (uint32_t)v;
  else if (json_find_i64(buf, "n_head", &v) == 1) out_hp->n_heads = (uint32_t)v;

  if (json_find_i64(buf, "num_key_value_heads", &v) == 1) out_hp->n_kv_heads = (uint32_t)v;
  else out_hp->n_kv_heads = out_hp->n_heads;

  if (json_find_i64(buf, "hidden_size", &v) == 1) out_hp->d_model = (uint32_t)v;
  else if (json_find_i64(buf, "n_embd", &v) == 1) out_hp->d_model = (uint32_t)v;

  if (json_find_i64(buf, "intermediate_size", &v) == 1) out_hp->d_ff = (uint32_t)v;
  else if (json_find_i64(buf, "n_inner", &v) == 1) out_hp->d_ff = (uint32_t)v;
  else if (out_hp->d_model) out_hp->d_ff = out_hp->d_model * 4u;

  if (vocab_size_from_tok != 0u) out_hp->vocab_size = vocab_size_from_tok;
  else if (json_find_i64(buf, "vocab_size", &v) == 1) out_hp->vocab_size = (uint32_t)v;

  if (json_find_i64(buf, "max_position_embeddings", &v) == 1) out_hp->max_seq = (uint32_t)v;
  else if (json_find_i64(buf, "n_positions", &v) == 1) out_hp->max_seq = (uint32_t)v;
  else out_hp->max_seq = 2048u;

  free(buf);

  if (out_hp->n_layers == 0u || out_hp->n_heads == 0u || out_hp->d_model == 0u ||
      out_hp->vocab_size == 0u || out_hp->max_seq == 0u) {
    (void)fprintf(stderr, "ie_engine_create: invalid/missing hparams in config.json (model_dir=%s)\n", model_dir);
    return -3;
  }

  out_hp->d_head = out_hp->d_model / out_hp->n_heads;
  if (out_hp->d_head == 0u) return -3;

  return 0;
}

static int try_open_tokenizer_json(const char *model_dir, ie_tok_gptoss_t **out_tok, uint32_t *out_vocab) {
  if (!model_dir || !out_tok || !out_vocab) return -1;
  *out_tok = NULL;
  *out_vocab = 0u;

  char p0[2048];
  char p1[2048];

  int n0 = snprintf(p0, sizeof(p0), "%s/tokenizer.json", model_dir);
  int n1 = snprintf(p1, sizeof(p1), "%s/hf/original/tokenizer.json", model_dir);

  if (n0 > 0 && (size_t)n0 < sizeof(p0)) {
    ie_tok_gptoss_t *tok = NULL;
    if (ie_tok_gptoss_open(p0, &tok) == 0 && tok) {
      *out_tok = tok;
      *out_vocab = ie_tok_gptoss_vocab_size(tok);
      return 0;
    }
  }

  if (n1 > 0 && (size_t)n1 < sizeof(p1)) {
    ie_tok_gptoss_t *tok = NULL;
    if (ie_tok_gptoss_open(p1, &tok) == 0 && tok) {
      *out_tok = tok;
      *out_vocab = ie_tok_gptoss_vocab_size(tok);
      return 0;
    }
  }

  return -2;
}

/* ------------------------------ engine object ------------------------------ */

struct ie_engine {
  ie_engine_params_t params;
  char model_dir[1024];
  ie_device_caps_t caps;
  ie_device_t *dev;
  ie_weights_t weights;
  int weights_open;
  ie_tok_gptoss_t *tok;
  uint32_t vocab_size;
  ie_gptoss_hparams_t hp;
  ie_gptoss_infer_t *infer;
};

static void params_set_defaults(ie_engine_params_t *p) {
  if (!p) return;
  p->precision = "fp32";
  p->sparsity = "none";
  p->affinity = "auto";
  p->pretranspose = "none";
  p->prefetch = "auto";
  p->threads = 0;
}

static void params_merge(ie_engine_params_t *dst, const ie_engine_params_t *src) {
  if (!dst || !src) return;
  if (src->precision && src->precision[0] != '\0') dst->precision = src->precision;
  if (src->sparsity && src->sparsity[0] != '\0') dst->sparsity = src->sparsity;
  if (src->affinity && src->affinity[0] != '\0') dst->affinity = src->affinity;
  if (src->pretranspose && src->pretranspose[0] != '\0') dst->pretranspose = src->pretranspose;
  if (src->prefetch && src->prefetch[0] != '\0') dst->prefetch = src->prefetch;
  if (src->threads > 0) dst->threads = src->threads;
}

/* ------------------------------- public API -------------------------------- */

ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out_engine) {
  if (!out_engine || !device || !model_dir) return IE_ERR_BADARG;
  *out_engine = NULL;

  struct ie_engine *e = (struct ie_engine *)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_OOM;

  params_set_defaults(&e->params);
  if (p) params_merge(&e->params, p);

  (void)snprintf(e->model_dir, sizeof(e->model_dir), "%s", model_dir);

  const ie_device_kind_t kind = ie_device_kind_from_str(device);
  if (ie_device_create(kind, &e->dev) != 0 || !e->dev) {
    (void)fprintf(stderr, "ie_engine_create: ie_device_create failed (device=%s)\n", device);
    free(e);
    return IE_ERR_UNSUPPORTED;
  }
  (void)ie_device_caps(e->dev, &e->caps);

  {
    char json_path[2048];
    int n = snprintf(json_path, sizeof(json_path), "%s/model.ie.json", model_dir);
    if (n <= 0 || (size_t)n >= sizeof(json_path)) {
      (void)fprintf(stderr, "ie_engine_create: model.ie.json path overflow (model_dir=%s)\n", model_dir);
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_MODEL;
    }
    int io_rc = ie_weights_open(json_path, NULL, &e->weights);
    if (io_rc != 0) {
      (void)fprintf(stderr, "ie_engine_create: ie_weights_open failed (rc=%d, json=%s)\n", io_rc, json_path);
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_MODEL;
    }
    e->weights_open = 1;
  }

  e->tok = NULL;
  e->vocab_size = 0;
  {
    ie_tok_gptoss_t *tok = NULL;
    uint32_t vocab = 0;
    if (try_open_tokenizer_json(model_dir, &tok, &vocab) == 0 && tok) {
      e->tok = tok;
      e->vocab_size = vocab;
      (void)fprintf(stderr, "ie_engine_create: tokenizer loaded (vocab=%u)\n", (unsigned)e->vocab_size);
    } else {
      (void)fprintf(stderr, "ie_engine_create: tokenizer.json not found; falling back to stub vocab\n");
    }
  }

  if (e->vocab_size == 0u) {
    e->vocab_size = 50257u;
  }

  {
    int hp_rc = gptoss_hparams_from_config_json(model_dir, e->vocab_size, &e->hp);
    if (hp_rc != 0) {
      if (e->tok) {
        ie_tok_gptoss_close(e->tok);
        e->tok = NULL;
      }
      if (e->weights_open) {
        ie_weights_close(&e->weights);
        e->weights_open = 0;
      }
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_MODEL;
    }
  }

  e->infer = NULL;
  {
    int inf_rc = ie_gptoss_infer_create(e->dev, &e->weights, &e->hp, &e->infer);
    if (inf_rc != 0 || !e->infer) {
      (void)fprintf(stderr, "ie_engine_create: ie_gptoss_infer_create failed (rc=%d)\n", inf_rc);
      if (e->tok) {
        ie_tok_gptoss_close(e->tok);
        e->tok = NULL;
      }
      if (e->weights_open) {
        ie_weights_close(&e->weights);
        e->weights_open = 0;
      }
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_UNSUPPORTED;
    }
  }

  *out_engine = (ie_engine_t *)e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *engine) {
  struct ie_engine *e = (struct ie_engine *)engine;
  if (!e) return;

  if (e->infer) {
    ie_gptoss_infer_destroy(e->infer);
    e->infer = NULL;
  }

  if (e->tok) {
    ie_tok_gptoss_close(e->tok);
    e->tok = NULL;
  }

  if (e->weights_open) {
    ie_weights_close(&e->weights);
    e->weights_open = 0;
  }

  if (e->dev) {
    ie_device_destroy(e->dev);
    e->dev = NULL;
  }

  free(e);
}

/* ------------------------------ KV + sampling ------------------------------ */

static int kv_array_create(const ie_gptoss_hparams_t *hp,
                           uint32_t need_seq,
                           ie_kv_cache **out_kv) {
  if (!hp || !out_kv) return -1;
  *out_kv = NULL;

  const uint32_t n_layers = hp->n_layers;
  if (n_layers == 0u) return -1;

  uint32_t max_seq = hp->max_seq;
  if (max_seq < need_seq) max_seq = need_seq;

  ie_kv_cache *kv = (ie_kv_cache *)calloc((size_t)n_layers, sizeof(ie_kv_cache));
  if (!kv) return -2;

  const char *storage_s = getenv("IE_KV_STORAGE");
  ie_kv_storage_type storage = IE_KV_STORAGE_F32;
  if (storage_s && storage_s[0]) {
    if (strcmp(storage_s, "f32") == 0) storage = IE_KV_STORAGE_F32;
    else if (strcmp(storage_s, "int8") == 0) storage = IE_KV_STORAGE_INT8;
    else if (strcmp(storage_s, "fp8") == 0) storage = IE_KV_STORAGE_FP8;
  }

  const size_t group_size = (size_t)env_u32("IE_KV_GROUP_SIZE", 64u);
  const int symmetric = env_truthy("IE_KV_SYMMETRIC", 1);
  const ie_fp8_format fp8_fmt = (env_u32("IE_KV_FP8_FMT", 0u) == 1u) ? IE_FP8_E5M2 : IE_FP8_E4M3;

  const uint32_t kv_heads = (hp->n_kv_heads != 0u) ? hp->n_kv_heads : hp->n_heads;

  for (uint32_t l = 0; l < n_layers; ++l) {
    ie_kv_opts opts;
    opts.heads = (int)kv_heads;
    opts.head_dim = (int)hp->d_head;
    opts.max_seq = (int)max_seq;
    opts.storage = storage;
    opts.group_size = group_size;
    opts.symmetric = symmetric;
    opts.fp8_format = fp8_fmt;

    if (ie_kv_init(&kv[l], &opts) != 0) {
      for (uint32_t j = 0; j < l; ++j) ie_kv_free(&kv[j]);
      free(kv);
      return -3;
    }
  }

  *out_kv = kv;
  return 0;
}

static void kv_array_free(const ie_gptoss_hparams_t *hp, ie_kv_cache *kv) {
  if (!hp || !kv) return;
  for (uint32_t l = 0; l < hp->n_layers; ++l) ie_kv_free(&kv[l]);
  free(kv);
}

static void sample_cfg_from_env(ie_sample_cfg_t *out_cfg) {
  if (!out_cfg) return;
  memset(out_cfg, 0, sizeof(*out_cfg));

  const char *k = getenv("IE_SAMPLE");
  if (!k || !k[0] || strcmp(k, "greedy") == 0) out_cfg->kind = IE_SAMPLE_GREEDY;
  else if (strcmp(k, "topk") == 0) out_cfg->kind = IE_SAMPLE_TOPK;
  else if (strcmp(k, "topp") == 0) out_cfg->kind = IE_SAMPLE_TOPP;
  else out_cfg->kind = IE_SAMPLE_GREEDY;

  out_cfg->temperature = env_f32("IE_TEMP", 1.0f);
  if (out_cfg->temperature <= 0.0f) out_cfg->temperature = 1.0f;

  out_cfg->top_k = env_u32("IE_TOPK", 50u);
  out_cfg->top_p = env_f32("IE_TOPP", 0.9f);
  if (out_cfg->top_p <= 0.0f) out_cfg->top_p = 0.9f;
  if (out_cfg->top_p > 1.0f) out_cfg->top_p = 1.0f;

  out_cfg->disallow_token0 = env_truthy("IE_DISALLOW_TOKEN0", 0);
}

static ie_status_t generate_fake_tokens(const struct ie_engine *e,
                                        const char *prompt,
                                        size_t max_new_tokens,
                                        int *out_tokens,
                                        size_t *out_n_tokens) {
  const uint32_t vocab = (e->vocab_size != 0u) ? e->vocab_size : 1u;
  uint32_t seed = fnv1a32_cstr(prompt);
  seed ^= 0x9e3779b9u;

  for (size_t i = 0; i < max_new_tokens; ++i) {
    uint32_t x = seed + (uint32_t)(i * 0x9e3779b9u);
    x = xorshift32(x);
    out_tokens[i] = (int)(x % vocab);
  }

  *out_n_tokens = max_new_tokens;
  return IE_OK;
}

ie_status_t ie_engine_generate(const ie_engine_t *engine,
                               const char *prompt,
                               size_t max_new_tokens,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  const struct ie_engine *e = (const struct ie_engine *)engine;
  if (!e || !prompt || !out_tokens || !out_n_tokens) return IE_ERR_BADARG;

  *out_n_tokens = 0;
  if (max_new_tokens == 0) return IE_OK;

  if (!e->infer) {
    if (env_truthy("IE_ALLOW_FAKE_TOKENS", 0)) {
      return generate_fake_tokens(e, prompt, max_new_tokens, out_tokens, out_n_tokens);
    }
    (void)fprintf(stderr,
                  "ERROR: Inference context is unavailable.\n"
                  "ERROR: Implement and enable ie_gptoss_infer_prefill/step.\n");
    return IE_ERR_UNSUPPORTED;
  }

  uint32_t *prompt_ids = NULL;
  uint32_t n_prompt = 0;
  if (e->tok) {
    uint32_t cap = 0;
    if (ie_tok_gptoss_encode(e->tok, prompt, NULL, &cap) != 0 || cap == 0u) {
      cap = 1u;
    }
    prompt_ids = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    if (!prompt_ids) return IE_ERR_OOM;

    n_prompt = cap;
    if (ie_tok_gptoss_encode(e->tok, prompt, prompt_ids, &n_prompt) != 0) {
      free(prompt_ids);
      return IE_ERR_INTERNAL;
    }
    if (n_prompt == 0u) {
      n_prompt = 1u;
      prompt_ids[0] = 0u;
    }
  } else {
    prompt_ids = (uint32_t *)malloc(sizeof(uint32_t));
    if (!prompt_ids) return IE_ERR_OOM;
    n_prompt = 1u;
    prompt_ids[0] = 0u;
  }

  ie_kv_cache *kv = NULL;
  uint32_t need_seq = n_prompt + (uint32_t)max_new_tokens;
  if (kv_array_create(&e->hp, need_seq, &kv) != 0) {
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  const size_t vocab = (size_t)e->hp.vocab_size;
  float *logits = (float *)malloc(vocab * sizeof(float));
  if (!logits) {
    kv_array_free(&e->hp, kv);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  ie_sample_cfg_t cfg;
  sample_cfg_from_env(&cfg);

  uint32_t *idx_scratch = NULL;
  float *prob_scratch = NULL;
  size_t scratch_cap = 0;
  if (cfg.kind != IE_SAMPLE_GREEDY) {
    scratch_cap = vocab;
    idx_scratch = (uint32_t *)malloc(vocab * sizeof(uint32_t));
    prob_scratch = (float *)malloc(vocab * sizeof(float));
    if (!idx_scratch || !prob_scratch) {
      free(prob_scratch);
      free(idx_scratch);
      free(logits);
      kv_array_free(&e->hp, kv);
      free(prompt_ids);
      return IE_ERR_OOM;
    }
  }

  ie_rng_t rng;
  uint64_t seed = (uint64_t)env_u32("IE_SEED", 0u);
  if (seed == 0u) {
    seed = ((uint64_t)fnv1a32_cstr(prompt) << 32) ^ (uint64_t)fnv1a32_cstr(e->model_dir);
  }
  ie_rng_init(&rng, seed);

  int rc = ie_gptoss_infer_prefill(e->infer, kv, prompt_ids, n_prompt, logits);
  if (rc != 0) {
    const int allow_fake = env_truthy("IE_ALLOW_FAKE_TOKENS", 0);
    free(prob_scratch);
    free(idx_scratch);
    free(logits);
    kv_array_free(&e->hp, kv);
    free(prompt_ids);

    if (allow_fake) {
      return generate_fake_tokens(e, prompt, max_new_tokens, out_tokens, out_n_tokens);
    }

    (void)fprintf(stderr,
                  "ERROR: Transformer prefill failed (rc=%d).\n"
                  "ERROR: Implement ie_gptoss_infer_prefill/step for real generation.\n",
                  rc);
    return IE_ERR_UNSUPPORTED;
  }

  for (size_t i = 0; i < max_new_tokens; ++i) {
    uint32_t next_id = 0;
    if (ie_sample_next(logits, vocab, &cfg, &rng,
                       idx_scratch, prob_scratch, scratch_cap, &next_id) != 0) {
      *out_n_tokens = i;
      break;
    }

    out_tokens[i] = (int)next_id;
    *out_n_tokens = i + 1;

    rc = ie_gptoss_infer_step(e->infer, kv, next_id, logits);
    if (rc != 0) break;
  }

  free(prob_scratch);
  free(idx_scratch);
  free(logits);
  kv_array_free(&e->hp, kv);
  free(prompt_ids);

  return IE_OK;
}
