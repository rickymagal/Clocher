/* engine/src/runtime/sampling.c */
/**
 * @file sampling.c
 * @brief Token sampling utilities for next-token selection from logits.
 *
 * Implementation details:
 *  - Greedy: argmax.
 *  - Top-k: select k best logits via simple O(V*K) insertion, softmax over K.
 *  - Top-p: softmax over all logits (temperature), sort by probability desc,
 *           take smallest prefix with cumprob >= p, renormalize and sample.
 *
 * This is correctness-first; optimize later if needed.
 */

#include "ie_sampling.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ============================================================================
 * RNG (xorshift64*)
 * ========================================================================== */

/**
 * @brief Initialize RNG with a seed (0 is remapped to a fixed non-zero value).
 *
 * @param rng  RNG state.
 * @param seed Seed value.
 */
void ie_rng_init(ie_rng_t *rng, uint64_t seed) {
  if (!rng) return;
  rng->s = (seed == 0) ? 0x9E3779B97F4A7C15ull : seed;
}

/**
 * @brief xorshift64* core step.
 *
 * @param s RNG state pointer.
 * @return Next 64-bit value.
 */
static uint64_t xorshift64star(uint64_t *s) {
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 2685821657736338717ull;
}

/**
 * @brief Return a random uint32.
 *
 * @param rng RNG state.
 * @return Random number.
 */
uint32_t ie_rng_u32(ie_rng_t *rng) {
  if (!rng) return 0;
  return (uint32_t)(xorshift64star(&rng->s) >> 32);
}

/**
 * @brief Return a random float in [0, 1).
 *
 * @param rng RNG state.
 * @return Random float.
 */
float ie_rng_f32(ie_rng_t *rng) {
  uint32_t x = ie_rng_u32(rng);
  uint32_t m = x >> 8; /* 24 bits */
  return (float)m / (float)(1u << 24);
}

/* ============================================================================
 * Helpers
 * ========================================================================== */

/**
 * @brief Sanitize user-provided sampling config into a safe internal config.
 *
 * @param cfg Input config.
 * @param out Output sanitized config.
 * @return 0 on success, negative on error.
 */
static int cfg_sanitize(const ie_sample_cfg_t *cfg, ie_sample_cfg_t *out) {
  if (!cfg || !out) return -1;
  *out = *cfg;

  if (out->temperature <= 0.0f) out->temperature = 1.0f;

  if (out->kind == IE_SAMPLE_TOPK) {
    if (out->top_k == 0) out->top_k = 1;
  } else if (out->kind == IE_SAMPLE_TOPP) {
    if (out->top_p <= 0.0f) out->top_p = 0.9f;
    if (out->top_p > 1.0f) out->top_p = 1.0f;
  }
  return 0;
}

/**
 * @brief Return argmax index of logits.
 *
 * @param logits     Logits vector.
 * @param n          Length.
 * @param disallow0  If nonzero, skip token 0.
 * @return Index of maximum logit.
 */
static uint32_t argmax_logits(const float *logits, size_t n, int disallow0) {
  uint32_t best_i = 0;
  float best_v = -INFINITY;

  size_t start = 0;
  if (disallow0 && n > 0) start = 1;
  if (start >= n) return 0;

  best_i = (uint32_t)start;
  best_v = logits[start];

  for (size_t i = start + 1; i < n; ++i) {
    float v = logits[i];
    if (v > best_v) {
      best_v = v;
      best_i = (uint32_t)i;
    }
  }
  return best_i;
}

/**
 * @brief Numerically safe exp wrapper.
 *
 * @param x Input value.
 * @return expf(x).
 */
static float safe_exp(float x) {
  return expf(x);
}

/**
 * @brief Normalize an array of positive values to sum to 1 (in-place).
 *
 * @param p Array of unnormalized probabilities.
 * @param n Length.
 */
static void softmax_inplace(float *p, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) sum += p[i];
  if (sum <= 0.0f) {
    float inv = (n ? (1.0f / (float)n) : 0.0f);
    for (size_t i = 0; i < n; ++i) p[i] = inv;
    return;
  }
  float inv = 1.0f / sum;
  for (size_t i = 0; i < n; ++i) p[i] *= inv;
}

/**
 * @brief Sample an index from a probability vector.
 *
 * @param p   Probability array (sum approximately 1).
 * @param idx Optional index mapping (if non-NULL, returned id is idx[i]).
 * @param n   Length.
 * @param rng RNG state.
 * @return Sampled token id (mapped or direct index).
 */
static uint32_t sample_from_probs(const float *p,
                                  const uint32_t *idx,
                                  size_t n,
                                  ie_rng_t *rng) {
  float r = ie_rng_f32(rng);
  float c = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    c += p[i];
    if (r < c) return idx ? idx[i] : (uint32_t)i;
  }
  return idx ? idx[n - 1] : (uint32_t)(n - 1);
}

/**
 * @brief Sort an index array by descending probability (prob[idx[i]]).
 *
 * @param idx  Index array to sort.
 * @param prob Probability array (full vocab).
 * @param n    Length of idx.
 */
static void sort_desc_by_prob(uint32_t *idx, const float *prob, size_t n) {
  for (size_t i = 1; i < n; ++i) {
    uint32_t key = idx[i];
    float pv = prob[key];
    size_t j = i;
    while (j > 0) {
      uint32_t prev = idx[j - 1];
      if (prob[prev] >= pv) break;
      idx[j] = prev;
      --j;
    }
    idx[j] = key;
  }
}

/* ============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Choose next token id from logits given a sampling configuration.
 *
 * @param logits       Logits array.
 * @param vocab_size   Vocabulary size.
 * @param cfg_in       Sampling config.
 * @param rng          RNG state (required for stochastic sampling).
 * @param idx_scratch  Scratch indices (cap >= vocab_size for non-greedy).
 * @param prob_scratch Scratch probabilities (cap >= vocab_size for non-greedy).
 * @param scratch_cap  Scratch capacity (shared).
 * @param out_id       Output token id.
 * @return 0 on success, negative on error.
 */
int ie_sample_next(const float *logits,
                   size_t vocab_size,
                   const ie_sample_cfg_t *cfg_in,
                   ie_rng_t *rng,
                   uint32_t *idx_scratch,
                   float *prob_scratch,
                   size_t scratch_cap,
                   uint32_t *out_id) {
  if (!logits || vocab_size == 0 || !cfg_in || !out_id) return -1;

  ie_sample_cfg_t cfg;
  if (cfg_sanitize(cfg_in, &cfg) != 0) return -1;

  if (cfg.kind == IE_SAMPLE_GREEDY) {
    *out_id = argmax_logits(logits, vocab_size, cfg.disallow_token0);
    return 0;
  }

  if (!idx_scratch || !prob_scratch || scratch_cap < vocab_size) return -1;
  if (!rng) return -1;

  if (cfg.kind == IE_SAMPLE_TOPK) {
    uint32_t k = cfg.top_k;
    if (k > (uint32_t)vocab_size) k = (uint32_t)vocab_size;

    uint32_t sel_n = 0;
    for (size_t i = 0; i < vocab_size && sel_n < k; ++i) {
      if (cfg.disallow_token0 && i == 0) continue;
      idx_scratch[sel_n++] = (uint32_t)i;
    }
    if (sel_n == 0) {
      *out_id = 0;
      return 0;
    }

    for (uint32_t i = 1; i < sel_n; ++i) {
      uint32_t key = idx_scratch[i];
      float v = logits[key];
      uint32_t j = i;
      while (j > 0) {
        uint32_t prev = idx_scratch[j - 1];
        if (logits[prev] <= v) break;
        idx_scratch[j] = prev;
        --j;
      }
      idx_scratch[j] = key;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
      if (cfg.disallow_token0 && i == 0) continue;
      float v = logits[i];
      uint32_t worst = idx_scratch[0];
      if (v <= logits[worst]) continue;

      idx_scratch[0] = (uint32_t)i;
      for (uint32_t j = 1; j < sel_n; ++j) {
        uint32_t a = idx_scratch[j - 1];
        uint32_t b = idx_scratch[j];
        if (logits[a] <= logits[b]) continue;
        idx_scratch[j - 1] = b;
        idx_scratch[j] = a;
      }
    }

    float maxl = -INFINITY;
    for (uint32_t i = 0; i < sel_n; ++i) {
      float v = logits[idx_scratch[i]] / cfg.temperature;
      if (v > maxl) maxl = v;
    }
    for (uint32_t i = 0; i < sel_n; ++i) {
      float v = (logits[idx_scratch[i]] / cfg.temperature) - maxl;
      prob_scratch[i] = safe_exp(v);
    }
    softmax_inplace(prob_scratch, sel_n);
    *out_id = sample_from_probs(prob_scratch, idx_scratch, sel_n, rng);
    return 0;
  }

  if (cfg.kind == IE_SAMPLE_TOPP) {
    float maxl = -INFINITY;
    for (size_t i = 0; i < vocab_size; ++i) {
      if (cfg.disallow_token0 && i == 0) continue;
      float v = logits[i] / cfg.temperature;
      if (v > maxl) maxl = v;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
      if (cfg.disallow_token0 && i == 0) {
        prob_scratch[i] = 0.0f;
        continue;
      }
      float v = (logits[i] / cfg.temperature) - maxl;
      float e = safe_exp(v);
      prob_scratch[i] = e;
      sum += e;
    }
    if (sum <= 0.0f) {
      *out_id = argmax_logits(logits, vocab_size, cfg.disallow_token0);
      return 0;
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < vocab_size; ++i) prob_scratch[i] *= inv;

    size_t nidx = 0;
    for (size_t i = 0; i < vocab_size; ++i) {
      if (cfg.disallow_token0 && i == 0) continue;
      idx_scratch[nidx++] = (uint32_t)i;
    }
    sort_desc_by_prob(idx_scratch, prob_scratch, nidx);

    float cum = 0.0f;
    size_t cut = 0;
    for (; cut < nidx; ++cut) {
      cum += prob_scratch[idx_scratch[cut]];
      if (cum >= cfg.top_p) {
        ++cut;
        break;
      }
    }
    if (cut == 0) cut = 1;
    if (cut > nidx) cut = nidx;

    float local_sum = 0.0f;
    for (size_t i = 0; i < cut; ++i) local_sum += prob_scratch[idx_scratch[i]];
    if (local_sum <= 0.0f) {
      *out_id = idx_scratch[0];
      return 0;
    }
    float linv = 1.0f / local_sum;
    for (size_t i = 0; i < cut; ++i) {
      prob_scratch[i] = prob_scratch[idx_scratch[i]] * linv;
    }

    *out_id = sample_from_probs(prob_scratch, idx_scratch, cut, rng);
    return 0;
  }

  return -1;
}
