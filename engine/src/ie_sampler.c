/* File: engine/src/ie_sampler.c */

/**
 * @file ie_sampler.c
 * @brief Implementation of logits sampling utilities (greedy, temperature, top-k, top-p).
 *
 * Notes:
 *  - This module performs softmax in float and may allocate O(vocab) scratch.
 *  - For large vocabularies, this is not optimal; once the model forward pass is
 *    wired, you may prefer partial selection without full softmax.
 *  - The goal here is correctness and debuggability to unblock coherent text generation.
 */

#include "ie_sampler.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Work item used for sorting probabilities.
 */
typedef struct ie_prob_item {
  uint32_t id;
  float p;
} ie_prob_item_t;

struct ie_sampler {
  size_t vocab_size;
  ie_sampler_cfg_t cfg;
  uint64_t rng;

  float *probs;            /* length vocab_size */
  ie_prob_item_t *items;   /* length vocab_size */
};

/**
 * @brief Xorshift64* PRNG step.
 *
 * @param[in,out] state RNG state (must be nonzero for maximal period).
 * @return Next random 64-bit value.
 */
static uint64_t rng_next_u64(uint64_t *state) {
  uint64_t x = *state;
  if (x == 0) x = 0x9e3779b97f4a7c15ULL;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 2685821657736338717ULL;
}

/**
 * @brief Convert RNG output to uniform float in [0,1).
 *
 * @param[in,out] state RNG state.
 * @return Uniform float in [0,1).
 */
static float rng_next_f01(uint64_t *state) {
  uint64_t r = rng_next_u64(state);
  uint32_t hi = (uint32_t)(r >> 40); /* 24 bits */
  return (float)hi / (float)(1u << 24);
}

/**
 * @brief Comparator for descending probability sort.
 */
static int prob_item_cmp_desc(const void *a, const void *b) {
  const ie_prob_item_t *x = (const ie_prob_item_t *)a;
  const ie_prob_item_t *y = (const ie_prob_item_t *)b;
  if (x->p > y->p) return -1;
  if (x->p < y->p) return 1;
  if (x->id < y->id) return -1;
  if (x->id > y->id) return 1;
  return 0;
}

/**
 * @brief Compute argmax of logits, treating NaN as -inf if allowed.
 */
static int argmax_logits(const float *logits, size_t n, int allow_nan, uint32_t *out_id) {
  if (!logits || n == 0 || !out_id) return 0;

  float best = -FLT_MAX;
  uint32_t best_id = 0;

  for (size_t i = 0; i < n; ++i) {
    float v = logits[i];
    if (isnan(v)) {
      if (!allow_nan) return 0;
      continue;
    }
    if (v > best) {
      best = v;
      best_id = (uint32_t)i;
    }
  }

  *out_id = best_id;
  return 1;
}

/**
 * @brief Compute softmax probabilities from logits with temperature scaling.
 *
 * @param[in]  logits     Input logits.
 * @param[in]  n          Length.
 * @param[in]  temperature Temperature (must be > 0).
 * @param[in]  allow_nan  If nonzero, NaNs are treated as -inf.
 * @param[out] probs      Output probabilities (length n).
 * @return 1 on success, 0 on failure.
 */
static int softmax_probs(const float *logits,
                         size_t n,
                         float temperature,
                         int allow_nan,
                         float *probs) {
  if (!logits || !probs || n == 0) return 0;
  if (!(temperature > 0.0f)) return 0;

  float maxv = -FLT_MAX;
  for (size_t i = 0; i < n; ++i) {
    float v = logits[i];
    if (isnan(v)) {
      if (!allow_nan) return 0;
      continue;
    }
    v = v / temperature;
    if (v > maxv) maxv = v;
  }

  if (maxv == -FLT_MAX) {
    for (size_t i = 0; i < n; ++i) probs[i] = 0.0f;
    probs[0] = 1.0f;
    return 1;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float v = logits[i];
    if (isnan(v)) {
      probs[i] = 0.0f;
      continue;
    }
    v = v / temperature;
    float e = expf(v - maxv);
    probs[i] = e;
    sum += e;
  }

  if (!(sum > 0.0f) || isnan(sum) || isinf(sum)) return 0;

  float inv = 1.0f / sum;
  for (size_t i = 0; i < n; ++i) probs[i] *= inv;
  return 1;
}

void ie_sampler_cfg_default(ie_sampler_cfg_t *cfg) {
  if (!cfg) return;
  cfg->temperature = 1.0f;
  cfg->top_p = 1.0f;
  cfg->top_k = 0;
  cfg->allow_nan_logits = 0;
}

ie_sampler_status_t ie_sampler_create(ie_sampler_t **out,
                                      size_t vocab_size,
                                      const ie_sampler_cfg_t *cfg,
                                      uint64_t seed) {
  if (!out || vocab_size == 0) return IE_SAMPLER_EINVAL;
  *out = NULL;

  ie_sampler_t *s = (ie_sampler_t *)calloc(1, sizeof(*s));
  if (!s) return IE_SAMPLER_ENOMEM;

  s->vocab_size = vocab_size;
  ie_sampler_cfg_default(&s->cfg);
  if (cfg) s->cfg = *cfg;

  s->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;

  s->probs = (float *)malloc(sizeof(float) * vocab_size);
  s->items = (ie_prob_item_t *)malloc(sizeof(ie_prob_item_t) * vocab_size);
  if (!s->probs || !s->items) {
    free(s->probs);
    free(s->items);
    free(s);
    return IE_SAMPLER_ENOMEM;
  }

  *out = s;
  return IE_SAMPLER_OK;
}

void ie_sampler_destroy(ie_sampler_t **ps) {
  if (!ps || !*ps) return;
  ie_sampler_t *s = *ps;

  free(s->probs);
  free(s->items);
  free(s);

  *ps = NULL;
}

ie_sampler_status_t ie_sampler_set_cfg(ie_sampler_t *s, const ie_sampler_cfg_t *cfg) {
  if (!s || !cfg) return IE_SAMPLER_EINVAL;
  s->cfg = *cfg;
  return IE_SAMPLER_OK;
}

ie_sampler_status_t ie_sampler_set_seed(ie_sampler_t *s, uint64_t seed) {
  if (!s) return IE_SAMPLER_EINVAL;
  s->rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
  return IE_SAMPLER_OK;
}

ie_sampler_status_t ie_sampler_sample(ie_sampler_t *s,
                                      const float *logits,
                                      size_t logits_len,
                                      uint32_t *out_id,
                                      float *out_prob) {
  if (!s || !logits || !out_id) return IE_SAMPLER_EINVAL;
  if (logits_len != s->vocab_size) return IE_SAMPLER_EINVAL;

  const float temperature = s->cfg.temperature;
  const uint32_t top_k = s->cfg.top_k;
  const float top_p = s->cfg.top_p;
  const int allow_nan = s->cfg.allow_nan_logits;

  if (!(temperature > 0.0f)) {
    uint32_t id = 0;
    if (!argmax_logits(logits, logits_len, allow_nan, &id)) return IE_SAMPLER_EINVAL;
    *out_id = id;
    if (out_prob) *out_prob = 1.0f;
    return IE_SAMPLER_OK;
  }

  if (!softmax_probs(logits, logits_len, temperature, allow_nan, s->probs)) {
    uint32_t id = 0;
    if (!argmax_logits(logits, logits_len, allow_nan, &id)) return IE_SAMPLER_EINVAL;
    *out_id = id;
    if (out_prob) *out_prob = 1.0f;
    return IE_SAMPLER_OK;
  }

  for (size_t i = 0; i < logits_len; ++i) {
    s->items[i].id = (uint32_t)i;
    s->items[i].p = s->probs[i];
  }

  qsort(s->items, logits_len, sizeof(s->items[0]), prob_item_cmp_desc);

  size_t kept = logits_len;

  if (top_k != 0 && top_k < kept) kept = (size_t)top_k;

  if (top_p < 1.0f) {
    float cum = 0.0f;
    size_t k = 0;
    while (k < kept) {
      cum += s->items[k].p;
      ++k;
      if (cum >= top_p) break;
    }
    if (k == 0) k = 1;
    kept = k;
  }

  float sum = 0.0f;
  for (size_t i = 0; i < kept; ++i) sum += s->items[i].p;

  if (!(sum > 0.0f) || isnan(sum) || isinf(sum)) {
    uint32_t id = s->items[0].id;
    *out_id = id;
    if (out_prob) *out_prob = 1.0f;
    return IE_SAMPLER_OK;
  }

  float r = rng_next_f01(&s->rng);
  float acc = 0.0f;
  uint32_t chosen = s->items[kept - 1].id;
  float chosen_p = s->items[kept - 1].p / sum;

  for (size_t i = 0; i < kept; ++i) {
    float p = s->items[i].p / sum;
    acc += p;
    if (r < acc) {
      chosen = s->items[i].id;
      chosen_p = p;
      break;
    }
  }

  *out_id = chosen;
  if (out_prob) *out_prob = chosen_p;
  return IE_SAMPLER_OK;
}

const char *ie_sampler_status_str(ie_sampler_status_t st) {
  switch (st) {
    case IE_SAMPLER_OK: return "IE_SAMPLER_OK";
    case IE_SAMPLER_EINVAL: return "IE_SAMPLER_EINVAL";
    case IE_SAMPLER_ENOMEM: return "IE_SAMPLER_ENOMEM";
    case IE_SAMPLER_EINTERNAL: return "IE_SAMPLER_EINTERNAL";
    default: return "IE_SAMPLER_UNKNOWN";
  }
}
