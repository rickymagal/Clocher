/* ============================================================================
 * File: engine/src/runtime/generate_gptoss.c
 * ============================================================================
 */
/**
 * @file generate_gptoss.c
 * @brief Convenience token generation wrapper for the GPT-OSS forward pass.
 */

#include "generate_gptoss.h"

#include <stdlib.h>
#include <string.h>

#include "gptoss_hparams.h"
#include "ie_kv_cache.h"
#include "ie_sampling.h"
#include "ie_tokenizer_gptoss.h"

ie_sample_cfg_t ie_default_sample_cfg(void) {
  ie_sample_cfg_t c;
  memset(&c, 0, sizeof(c));
  c.kind = IE_SAMPLE_TOPP;
  c.temperature = 1.0f;
  c.top_k = 40u;
  c.top_p = 0.95f;
  c.disallow_token0 = 0;
  return c;
}

static void ie_free_ptr(void **p) {
  if (p && *p) {
    free(*p);
    *p = NULL;
  }
}

int ie_generate_gptoss(const ie_device_t *dev,
                       const ie_weights_t *weights,
                       const char *model_dir,
                       const ie_tok_gptoss_t *tok,
                       const char *prompt,
                       uint32_t max_new,
                       const ie_sample_cfg_t *cfg,
                       uint64_t seed,
                       int *out_ids,
                       uint32_t *out_count)
{
  if (out_count) *out_count = 0;
  if (!dev || !weights || !model_dir || !tok || !prompt || !out_ids || !out_count) return -1;
  if (max_new == 0) return 0;

  int rc = 0;

  ie_gptoss_hparams_t hp;
  memset(&hp, 0, sizeof(hp));
  if (gptoss_hparams_load(model_dir, &hp) != 0) return -2;

  /* Encode the prompt. */
  uint32_t prompt_n = 0;
  if (ie_tok_gptoss_encode(tok, prompt, NULL, &prompt_n) != 0) return -3;
  if (prompt_n == 0) return -4;

  uint32_t *prompt_ids = (uint32_t *)malloc((size_t)prompt_n * sizeof(uint32_t));
  if (!prompt_ids) return -5;

  {
    uint32_t tmp = prompt_n;
    if (ie_tok_gptoss_encode(tok, prompt, prompt_ids, &tmp) != 0) {
      ie_free_ptr((void **)&prompt_ids);
      return -6;
    }
    prompt_n = tmp;
    if (prompt_n == 0) {
      ie_free_ptr((void **)&prompt_ids);
      return -7;
    }
  }

  /* Create inference context. */
  ie_gptoss_infer_t *inf = NULL;
  if (ie_gptoss_infer_create(dev, weights, &hp, &inf) != 0 || !inf) {
    ie_free_ptr((void **)&prompt_ids);
    return -8;
  }

  /* KV cache: one layer object per transformer layer. */
  const uint32_t n_layers = (hp.n_layers > 0) ? (uint32_t)hp.n_layers : 0u;
  if (n_layers == 0) {
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -9;
  }

  ie_kv_cache *kv = (ie_kv_cache *)calloc((size_t)n_layers, sizeof(ie_kv_cache));
  if (!kv) {
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -10;
  }

  ie_kv_opts kv_opts;
  memset(&kv_opts, 0, sizeof(kv_opts));
  kv_opts.max_seq = (uint32_t)((hp.max_seq > 0) ? hp.max_seq : 1);
  kv_opts.heads = (uint32_t)((hp.n_kv_heads > 0) ? hp.n_kv_heads : hp.n_heads);
  kv_opts.head_dim = (uint32_t)((hp.d_head > 0) ? hp.d_head : 1);
  kv_opts.storage = IE_KV_STORAGE_F32;
  kv_opts.group_size = 0;
  kv_opts.symmetric = 0;
  kv_opts.fp8_format = IE_FP8_E4M3;

  if (ie_kv_init_layers(kv, n_layers, &kv_opts) != 0) {
    ie_kv_free_layers(kv, n_layers);
    ie_free_ptr((void **)&kv);
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -11;
  }

  const size_t vocab = (size_t)((hp.vocab_size > 0) ? hp.vocab_size : 0);
  if (vocab == 0) {
    ie_kv_free_layers(kv, n_layers);
    ie_free_ptr((void **)&kv);
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -12;
  }

  float *logits = (float *)malloc(vocab * sizeof(float));
  if (!logits) {
    ie_kv_free_layers(kv, n_layers);
    ie_free_ptr((void **)&kv);
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -13;
  }

  if (ie_gptoss_infer_prefill(inf, kv, prompt_ids, (size_t)prompt_n, logits) != 0) {
    ie_free_ptr((void **)&logits);
    ie_kv_free_layers(kv, n_layers);
    ie_free_ptr((void **)&kv);
    ie_gptoss_infer_destroy(inf);
    ie_free_ptr((void **)&prompt_ids);
    return -14;
  }

  ie_rng_t rng;
  ie_rng_init(&rng, (seed == 0) ? 1u : seed);

  ie_sample_cfg_t local_cfg;
  if (cfg) {
    local_cfg = *cfg;
  } else {
    local_cfg = ie_default_sample_cfg();
  }

  uint32_t *idx_scratch = NULL;
  float *prob_scratch = NULL;

  if (local_cfg.kind != IE_SAMPLE_GREEDY) {
    idx_scratch = (uint32_t *)malloc(vocab * sizeof(uint32_t));
    prob_scratch = (float *)malloc(vocab * sizeof(float));
    if (!idx_scratch || !prob_scratch) {
      ie_free_ptr((void **)&prob_scratch);
      ie_free_ptr((void **)&idx_scratch);
      ie_free_ptr((void **)&logits);
      ie_kv_free_layers(kv, n_layers);
      ie_free_ptr((void **)&kv);
      ie_gptoss_infer_destroy(inf);
      ie_free_ptr((void **)&prompt_ids);
      return -15;
    }
  }

  uint32_t produced = 0;
  for (uint32_t i = 0; i < max_new; ++i) {
    uint32_t next_id = 0;
    if (ie_sample_next(logits,
                       vocab,
                       &local_cfg,
                       &rng,
                       idx_scratch,
                       prob_scratch,
                       vocab,
                       &next_id) != 0)
    {
      rc = -16;
      break;
    }

    out_ids[produced] = (int)next_id;
    produced++;

    if (ie_gptoss_infer_step(inf, kv, next_id, logits) != 0) {
      rc = -17;
      break;
    }
  }

  *out_count = produced;

  ie_free_ptr((void **)&prob_scratch);
  ie_free_ptr((void **)&idx_scratch);
  ie_free_ptr((void **)&logits);

  ie_kv_free_layers(kv, n_layers);
  ie_free_ptr((void **)&kv);

  ie_gptoss_infer_destroy(inf);
  ie_free_ptr((void **)&prompt_ids);

  return rc;
}
