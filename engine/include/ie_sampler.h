/* File: engine/include/ie_sampler.h */

/**
 * @file ie_sampler.h
 * @brief Logits sampling utilities (greedy, temperature, top-k, top-p).
 *
 * This module converts a logits vector into a next-token id using common
 * decoding strategies. It is designed to be:
 *  - deterministic given a seed and identical logits
 *  - independent of tokenizer and model I/O
 *  - safe under -Wall/-Wextra/-Werror/-pedantic
 *
 * Typical usage:
 *  1) Create a sampler for a fixed vocab size:
 *       ie_sampler_cfg_t cfg;
 *       ie_sampler_cfg_default(&cfg);
 *       cfg.temperature = 0.8f;
 *       cfg.top_p = 0.95f;
 *
 *       ie_sampler_t *s = NULL;
 *       ie_sampler_create(&s, vocab_size, &cfg, 1234);
 *
 *  2) For each step:
 *       uint32_t next_id = 0;
 *       ie_sampler_sample(s, logits, vocab_size, &next_id, NULL);
 *
 *  3) Destroy when done:
 *       ie_sampler_destroy(&s);
 */

#ifndef IE_SAMPLER_H
#define IE_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sampler status codes.
 */
typedef enum ie_sampler_status {
  /** Success. */
  IE_SAMPLER_OK = 0,
  /** Invalid argument. */
  IE_SAMPLER_EINVAL = -1,
  /** Allocation failure. */
  IE_SAMPLER_ENOMEM = -2,
  /** Internal error (unexpected state). */
  IE_SAMPLER_EINTERNAL = -3
} ie_sampler_status_t;

/**
 * @brief Sampling configuration.
 *
 * Interpretation:
 *  - If temperature <= 0, sampling defaults to greedy argmax.
 *  - top_k == 0 disables top-k filtering.
 *  - top_p >= 1 disables nucleus (top-p) filtering.
 *
 * If both top_k and top_p are enabled, filtering applies as:
 *  - compute probabilities (softmax)
 *  - sort descending by probability
 *  - apply top_k truncation (keep first K)
 *  - apply top_p truncation (keep smallest prefix with cumulative >= top_p)
 *  - renormalize and sample
 */
typedef struct ie_sampler_cfg {
  /** Temperature scaling. 1.0 = unchanged, <1 sharper, >1 flatter. */
  float temperature;
  /** Nucleus sampling probability threshold (0,1]. >=1 disables. */
  float top_p;
  /** Top-k sampling cutoff. 0 disables. */
  uint32_t top_k;
  /** If nonzero, treat NaN logits as -inf and continue; if zero, return error. */
  int allow_nan_logits;
} ie_sampler_cfg_t;

/**
 * @brief Opaque sampler handle.
 *
 * The sampler owns scratch buffers sized for the vocab size.
 */
typedef struct ie_sampler ie_sampler_t;

/**
 * @brief Fill a configuration struct with sane defaults.
 *
 * Defaults:
 *  - temperature = 1.0
 *  - top_p = 1.0 (disabled)
 *  - top_k = 0 (disabled)
 *  - allow_nan_logits = 0
 *
 * @param[out] cfg Configuration to initialize.
 */
void ie_sampler_cfg_default(ie_sampler_cfg_t *cfg);

/**
 * @brief Create a sampler for a fixed vocabulary size.
 *
 * @param[out] out        Receives newly allocated sampler on success.
 * @param[in]  vocab_size Vocabulary size (logits length).
 * @param[in]  cfg        Sampling configuration (may be NULL for defaults).
 * @param[in]  seed       Initial RNG seed (deterministic).
 * @return IE_SAMPLER_OK on success, negative on error.
 */
ie_sampler_status_t ie_sampler_create(ie_sampler_t **out,
                                      size_t vocab_size,
                                      const ie_sampler_cfg_t *cfg,
                                      uint64_t seed);

/**
 * @brief Destroy a sampler and release all resources.
 *
 * Safe to call with NULL or *ps == NULL.
 *
 * @param[in,out] ps Address of sampler handle; set to NULL on return.
 */
void ie_sampler_destroy(ie_sampler_t **ps);

/**
 * @brief Update sampler configuration.
 *
 * @param[in,out] s   Sampler handle.
 * @param[in]     cfg New configuration (must be non-NULL).
 * @return IE_SAMPLER_OK on success, negative on error.
 */
ie_sampler_status_t ie_sampler_set_cfg(ie_sampler_t *s, const ie_sampler_cfg_t *cfg);

/**
 * @brief Reset sampler RNG seed.
 *
 * @param[in,out] s    Sampler handle.
 * @param[in]     seed New seed value.
 * @return IE_SAMPLER_OK on success, negative on error.
 */
ie_sampler_status_t ie_sampler_set_seed(ie_sampler_t *s, uint64_t seed);

/**
 * @brief Sample a token id from logits.
 *
 * If sampling degenerates to greedy (temperature <= 0, or all mass collapses),
 * this returns argmax.
 *
 * @param[in,out] s         Sampler handle.
 * @param[in]     logits    Pointer to logits array (length = logits_len).
 * @param[in]     logits_len Length of logits array (should equal vocab_size).
 * @param[out]    out_id    Receives sampled token id.
 * @param[out]    out_prob  Optional. Receives probability of selected token under the filtered distribution.
 * @return IE_SAMPLER_OK on success, negative on error.
 */
ie_sampler_status_t ie_sampler_sample(ie_sampler_t *s,
                                      const float *logits,
                                      size_t logits_len,
                                      uint32_t *out_id,
                                      float *out_prob);

/**
 * @brief Convert status to string.
 *
 * @param[in] st Status code.
 * @return Constant string describing the status.
 */
const char *ie_sampler_status_str(ie_sampler_status_t st);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_SAMPLER_H */
