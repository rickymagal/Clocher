/* ============================================================================
 * File: engine/src/runtime/sampling.c
 * ============================================================================
 */
/**
 * @file sampling.c
 * @brief Token sampling utilities for next-token selection from logits.
 *
 * @details
 * This module provides a small, self-contained sampler used by the runtime to
 * select the next token id from a vector of logits.
 *
 * Implemented sampling modes:
 *  - Greedy (argmax): selects the maximum-logit token.
 *  - Top-k: keeps the best k logits via an insertion-style selection, applies a
 *           numerically-stable softmax over the selected subset, and samples.
 *  - Top-p (nucleus): applies temperature softmax over the full vocabulary,
 *           sorts by probability descending, takes the smallest prefix whose
 *           cumulative probability is >= p, renormalizes, and samples.
 *
 * Correctness policy:
 *  - This is correctness-first. Expensive operations (full sort in top-p) can be
 *    optimized later if needed.
 *
 * Tracing:
 *  - Set IE_TRACE_SAMPLING=1 to trace sampling decisions.
 *  - Optional: IE_TRACE_SAMPLING_N limits logs to the first N calls (0 = unlimited).
 */

#include "ie_sampling.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Tracing helpers
 * ========================================================================== */

/**
 * @brief Check whether sampling tracing is enabled via environment variable.
 *
 * @details
 * Tracing is enabled if IE_TRACE_SAMPLING is set and not equal to "0".
 * The value is cached after the first call.
 *
 * @return 1 if tracing is enabled; 0 otherwise.
 */
static int sampling_trace_enabled(void) {
  static int inited = 0;
  static int enabled = 0;
  if (!inited) {
    const char *s = getenv("IE_TRACE_SAMPLING");
    enabled = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
    inited = 1;
  }
  return enabled;
}

/**
 * @brief Return the trace call limit set by IE_TRACE_SAMPLING_N.
 *
 * @details
 * If IE_TRACE_SAMPLING_N is unset or parses to 0, tracing is unlimited.
 * The value is cached after the first call.
 *
 * @return Maximum number of trace events to emit; 0 means unlimited.
 */
static uint64_t sampling_trace_limit(void) {
  static int inited = 0;
  static uint64_t lim = 0;
  if (!inited) {
    const char *s = getenv("IE_TRACE_SAMPLING_N");
    if (s && s[0]) {
      errno = 0;
      char *end = NULL;
      unsigned long long v = strtoull(s, &end, 10);
      if (end != s && errno == 0) lim = (uint64_t)v;
    }
    inited = 1;
  }
  return lim;
}

/**
 * @brief Decide whether the current call should emit trace output.
 *
 * @details
 * This function enforces IE_TRACE_SAMPLING_N if present. Calls are counted
 * from the first invocation of this function in the process.
 *
 * @return 1 if tracing should be emitted for this call; 0 otherwise.
 */
static int sampling_trace_allow(void) {
  if (!sampling_trace_enabled()) return 0;
  static uint64_t calls = 0;
  calls++;
  const uint64_t lim = sampling_trace_limit();
  if (lim == 0) return 1;
  return calls <= lim;
}

/**
 * @brief Convert sampler kind enum into a human-readable string.
 *
 * @param k Sampling kind.
 * @return Static string for logging.
 */
static const char *kind_name(ie_sample_kind_t k) {
  switch (k) {
    case IE_SAMPLE_GREEDY: return "greedy";
    case IE_SAMPLE_TOPK:   return "topk";
    case IE_SAMPLE_TOPP:   return "topp";
    default: return "unknown";
  }
}

/* ============================================================================
 * RNG (xorshift64*)
 * ========================================================================== */

/**
 * @brief Initialize RNG state.
 *
 * @details
 * Uses xorshift64* for cheap non-cryptographic randomness.
 * If seed is 0, a fixed non-zero seed is used to avoid the zero lock state.
 *
 * @param rng  RNG state (output).
 * @param seed Seed value.
 */
void ie_rng_init(ie_rng_t *rng, uint64_t seed) {
  if (!rng) return;
  rng->s = (seed == 0) ? 0x9E3779B97F4A7C15ull : seed;
}

/**
 * @brief xorshift64* core step.
 *
 * @param s Pointer to 64-bit RNG state.
 * @return Next 64-bit random value.
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
 * @brief Generate a 32-bit random integer.
 *
 * @param rng RNG state.
 * @return Random 32-bit value (0 on NULL rng).
 */
uint32_t ie_rng_u32(ie_rng_t *rng) {
  if (!rng) return 0;
  return (uint32_t)(xorshift64star(&rng->s) >> 32);
}

/**
 * @brief Generate a float in [0, 1).
 *
 * @details
 * Produces 24 bits of mantissa using the high bits of a 32-bit random value.
 *
 * @param rng RNG state.
 * @return Random float in [0,1). If rng is NULL, returns 0.
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
 * @brief Validate and sanitize a sampling configuration.
 *
 * @details
 * Ensures temperature is positive, clamps top-p into (0,1], and normalizes
 * top-k to at least 1.
 *
 * @param cfg Input config.
 * @param out Output config.
 * @return 0 on success; non-zero on invalid args.
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
 * @brief Return the argmax index for a logits array.
 *
 * @param logits Logits array.
 * @param n      Number of logits.
 * @param disallow0 If non-zero, token id 0 is excluded from consideration.
 * @return Index of the maximum logit (or 0 if no valid index exists).
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
 * @brief Wrapper around expf for clarity and potential later clamping.
 *
 * @param x Input.
 * @return expf(x).
 */
static float safe_exp(float x) {
  return expf(x);
}

/**
 * @brief Normalize an array of non-negative values in-place so they sum to 1.
 *
 * @details
 * If the sum is non-positive, the distribution is replaced by uniform.
 *
 * @param p Array of weights/probabilities.
 * @param n Length of array.
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
 * @brief Sample an index from a probability distribution.
 *
 * @details
 * Uses a linear scan over the CDF.
 *
 * @param p   Probability values (must sum to ~1).
 * @param idx Optional index remap array. If non-NULL, returns idx[i].
 * @param n   Number of probabilities.
 * @param rng RNG state.
 * @return Sampled id (or last element if numerical drift leaves r >= cdf).
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
 * @brief Sort indices in descending order by probability.
 *
 * @details
 * Stable insertion sort using prob[key] as the key.
 *
 * @param idx  Indices to sort in-place.
 * @param prob Probability array indexed by token id.
 * @param n    Number of indices.
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

  if (sampling_trace_allow()) {
    fprintf(stderr,
            "[sampling] kind=%s temp=%.6g top_k=%u top_p=%.6g disallow0=%d vocab=%zu\n",
            kind_name(cfg.kind),
            (double)cfg.temperature,
            (unsigned)cfg.top_k,
            (double)cfg.top_p,
            (int)cfg.disallow_token0,
            vocab_size);
  }

  if (cfg.kind == IE_SAMPLE_GREEDY) {
    const uint32_t id = argmax_logits(logits, vocab_size, cfg.disallow_token0);
    *out_id = id;
    if (sampling_trace_allow()) {
      fprintf(stderr, "[sampling] greedy -> id=%u logit=%.6g\n", (unsigned)id, (double)logits[id]);
    }
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
      if (sampling_trace_allow()) fprintf(stderr, "[sampling] topk -> empty selection, id=0\n");
      return 0;
    }

    /* Keep selection sorted ascending by logit so idx_scratch[0] is the current worst. */
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

    const uint32_t id = sample_from_probs(prob_scratch, idx_scratch, sel_n, rng);
    *out_id = id;

    if (sampling_trace_allow()) {
      fprintf(stderr, "[sampling] topk(k=%u sel=%u) -> id=%u logit=%.6g\n",
              (unsigned)k, (unsigned)sel_n, (unsigned)id, (double)logits[id]);
    }
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
      const uint32_t id = argmax_logits(logits, vocab_size, cfg.disallow_token0);
      *out_id = id;
      if (sampling_trace_allow()) {
        fprintf(stderr, "[sampling] topp -> degenerate sum, fallback greedy id=%u\n", (unsigned)id);
      }
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
      if (sampling_trace_allow()) fprintf(stderr, "[sampling] topp -> degenerate local_sum, id=%u\n", (unsigned)*out_id);
      return 0;
    }
    float linv = 1.0f / local_sum;
    for (size_t i = 0; i < cut; ++i) {
      prob_scratch[i] = prob_scratch[idx_scratch[i]] * linv;
    }

    const uint32_t id = sample_from_probs(prob_scratch, idx_scratch, cut, rng);
    *out_id = id;

    if (sampling_trace_allow()) {
      fprintf(stderr, "[sampling] topp(p=%.6g cut=%zu cum=%.6g) -> id=%u logit=%.6g\n",
              (double)cfg.top_p, cut, (double)cum, (unsigned)id, (double)logits[id]);
    }
    return 0;
  }

  return -1;
}
