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
 *
 * Debugging:
 *  - IE_DEBUG_TOPK:          If set to K>0, dump the top-K logits (and decoded pieces if available).
 *  - IE_DEBUG_TOPK_EVERY:    If set to N>0, dump every N calls to ie_sampler_sample (default: 1).
 *  - IE_DEBUG_TOPK_LIMIT:    If set to M>0, stop dumping after M dumps (default: unlimited).
 *
 * Decoding:
 *  - If the engine provides a token decoder as a weak symbol:
 *      int ie_debug_decode_token(uint32_t id, char *dst, size_t cap);
 *    then the dump will include a best-effort decoded preview string.
 *  - If not provided, the dump prints "<no_decoder>".
 */

#include "ie_sampler.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Optional external decoder hook (weak).
 *
 * @details
 * If linked, it should write a NUL-terminated UTF-8 (or best-effort) token piece
 * preview into dst. Return nonzero on success.
 */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak)) int ie_debug_decode_token(uint32_t id, char *dst, size_t cap);
#endif

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

/* ============================================================================
 * Debug helpers
 * ========================================================================== */

static int env_int_(const char *name, int def, int minv, int maxv) {
  const char *v = getenv(name);
  if (!v || !*v) return def;

  errno = 0;
  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (errno != 0 || !end || end == v) return def;

  if (x < (long)minv) x = (long)minv;
  if (x > (long)maxv) x = (long)maxv;
  return (int)x;
}

static int debug_topk_k_(void) {
  static int inited = 0;
  static int k = 0;
  if (!inited) {
    k = env_int_("IE_DEBUG_TOPK", 0, 0, 128);
    inited = 1;
  }
  return k;
}

static int debug_topk_every_(void) {
  static int inited = 0;
  static int every = 1;
  if (!inited) {
    every = env_int_("IE_DEBUG_TOPK_EVERY", 1, 1, 1000000000);
    inited = 1;
  }
  return every;
}

static uint64_t debug_topk_limit_(void) {
  static int inited = 0;
  static uint64_t lim = 0;
  if (!inited) {
    const char *v = getenv("IE_DEBUG_TOPK_LIMIT");
    if (v && *v) {
      errno = 0;
      char *end = NULL;
      unsigned long long x = strtoull(v, &end, 10);
      if (errno == 0 && end && end != v) lim = (uint64_t)x;
    }
    inited = 1;
  }
  return lim;
}

static int debug_topk_should_dump_(uint64_t call_idx) {
  const int k = debug_topk_k_();
  if (k <= 0) return 0;

  const int every = debug_topk_every_();
  if (every <= 0) return 0;
  if ((call_idx % (uint64_t)every) != 0) return 0;

  const uint64_t lim = debug_topk_limit_();
  if (lim == 0) return 1;

  /* call_idx is 1-based; approximate dump-count as call_idx/every rounded up. */
  const uint64_t dump_no = (call_idx + (uint64_t)every - 1) / (uint64_t)every;
  return dump_no <= lim;
}

static void decode_preview_(uint32_t id, char *dst, size_t cap) {
  if (!dst || cap == 0) return;
  dst[0] = '\0';

#if defined(__GNUC__) || defined(__clang__)
  if (ie_debug_decode_token) {
    if (ie_debug_decode_token(id, dst, cap)) {
      dst[cap - 1] = '\0';
      for (size_t i = 0; dst[i]; ++i) {
        unsigned char c = (unsigned char)dst[i];
        if (c < 0x20 || c == 0x7f) dst[i] = ' ';
      }
      return;
    }
  }
#endif

  (void)snprintf(dst, cap, "<no_decoder>");
  dst[cap - 1] = '\0';
}

static void debug_dump_topk_logits_(const float *logits,
                                   const float *probs,
                                   size_t n,
                                   float temperature,
                                   int allow_nan,
                                   int k,
                                   uint64_t call_idx) {
  if (!logits || n == 0 || k <= 0) return;
  if (k > 64) k = 64;
  if ((size_t)k > n) k = (int)n;

  uint32_t top_id[64];
  float top_l[64];

  for (int i = 0; i < k; ++i) {
    top_id[i] = 0;
    top_l[i] = -INFINITY;
  }

  for (size_t i = 0; i < n; ++i) {
    float v = logits[i];
    if (isnan(v)) {
      if (!allow_nan) continue;
      v = -INFINITY;
    }
    if (v <= top_l[k - 1]) continue;

    int pos = k - 1;
    while (pos > 0 && v > top_l[pos - 1]) {
      top_l[pos] = top_l[pos - 1];
      top_id[pos] = top_id[pos - 1];
      --pos;
    }
    top_l[pos] = v;
    top_id[pos] = (uint32_t)i;
  }

  fprintf(stderr,
          "[sampler][topk_logits] call=%" PRIu64 " k=%d temp=%.6g\n",
          (uint64_t)call_idx,
          k,
          (double)temperature);

  for (int r = 0; r < k; ++r) {
    const uint32_t id = top_id[r];
    const float logit_raw = top_l[r];
    const float logit_scaled = (temperature > 0.0f) ? (logit_raw / temperature) : logit_raw;
    const float p = (probs ? probs[id] : -1.0f);

    char piece[160];
    decode_preview_(id, piece, sizeof(piece));

    fprintf(stderr,
            "[sampler][topk_logits] rank=%d id=%" PRIu32 " logit=%.9g scaled=%.9g prob=%.9g piece=\"%s\"\n",
            r,
            id,
            (double)logit_raw,
            (double)logit_scaled,
            (double)p,
            piece);
  }
}

/* ============================================================================
 * RNG (xorshift64*)
 * ========================================================================== */

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

  static uint64_t g_call_idx = 0;
  g_call_idx++;

  const float temperature = s->cfg.temperature;
  const uint32_t top_k = s->cfg.top_k;
  const float top_p = s->cfg.top_p;
  const int allow_nan = s->cfg.allow_nan_logits;

  if (!(temperature > 0.0f)) {
    uint32_t id = 0;
    if (!argmax_logits(logits, logits_len, allow_nan, &id)) return IE_SAMPLER_EINVAL;
    *out_id = id;
    if (out_prob) *out_prob = 1.0f;

    if (debug_topk_should_dump_(g_call_idx)) {
      const int k = debug_topk_k_();
      debug_dump_topk_logits_(logits, NULL, logits_len, temperature, allow_nan, k, g_call_idx);
      fprintf(stderr, "[sampler][chosen] call=%" PRIu64 " id=%" PRIu32 " prob=1\n",
              (uint64_t)g_call_idx, id);
    }
    return IE_SAMPLER_OK;
  }

  if (!softmax_probs(logits, logits_len, temperature, allow_nan, s->probs)) {
    uint32_t id = 0;
    if (!argmax_logits(logits, logits_len, allow_nan, &id)) return IE_SAMPLER_EINVAL;
    *out_id = id;
    if (out_prob) *out_prob = 1.0f;

    if (debug_topk_should_dump_(g_call_idx)) {
      const int k = debug_topk_k_();
      debug_dump_topk_logits_(logits, NULL, logits_len, temperature, allow_nan, k, g_call_idx);
      fprintf(stderr, "[sampler][chosen] call=%" PRIu64 " id=%" PRIu32 " prob=1 (softmax_failed)\n",
              (uint64_t)g_call_idx, id);
    }
    return IE_SAMPLER_OK;
  }

  if (debug_topk_should_dump_(g_call_idx)) {
    const int k = debug_topk_k_();
    debug_dump_topk_logits_(logits, s->probs, logits_len, temperature, allow_nan, k, g_call_idx);
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

    if (debug_topk_should_dump_(g_call_idx)) {
      fprintf(stderr, "[sampler][chosen] call=%" PRIu64 " id=%" PRIu32 " prob=1 (degenerate_sum)\n",
              (uint64_t)g_call_idx, id);
    }
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

  if (debug_topk_should_dump_(g_call_idx)) {
    char piece[160];
    decode_preview_(chosen, piece, sizeof(piece));
    fprintf(stderr,
            "[sampler][chosen] call=%" PRIu64 " id=%" PRIu32 " prob=%.9g piece=\"%s\"\n",
            (uint64_t)g_call_idx,
            chosen,
            (double)chosen_p,
            piece);
  }

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
