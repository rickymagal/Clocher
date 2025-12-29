/* engine/include/ie_sampling.h */
/**
 * @file ie_sampling.h
 * @brief Token sampling utilities for next-token selection from logits.
 *
 * This module provides small, allocation-free (caller-provided scratch) helpers
 * to choose the next token id from a logits vector.
 *
 * Supported strategies:
 *  - Greedy (argmax)
 *  - Top-k (sample from k highest logits)
 *  - Top-p / nucleus (sample from smallest set with cumulative probability >= p)
 *
 * Notes:
 *  - Logits are assumed to be unnormalized. Softmax is computed over the
 *    filtered set only (or full vocab for top-p before truncation).
 *  - For determinism, provide an explicit RNG seed and update it per step.
 *  - No dependency on libc RNG; uses xorshift64*.
 *
 * Thread-safety:
 *  - All functions are pure with respect to inputs, except RNG state which the
 *    caller owns and mutates.
 */

#ifndef IE_SAMPLING_H_
#define IE_SAMPLING_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** Sampling strategy selector. */
typedef enum ie_sample_kind_e {
  IE_SAMPLE_GREEDY = 0,
  IE_SAMPLE_TOPK = 1,
  IE_SAMPLE_TOPP = 2
} ie_sample_kind_t;

/** Sampling configuration. */
typedef struct ie_sample_cfg_s {
  /** Strategy kind. */
  ie_sample_kind_t kind;

  /** Temperature. 1.0f is neutral. Values < 1 sharpen; > 1 soften. */
  float temperature;

  /** Top-k parameter (used if kind == IE_SAMPLE_TOPK). */
  uint32_t top_k;

  /** Top-p parameter in (0, 1] (used if kind == IE_SAMPLE_TOPP). */
  float top_p;

  /** If nonzero, suppress token id 0 (sometimes reserved). */
  int disallow_token0;
} ie_sample_cfg_t;

/**
 * @brief Simple RNG state (xorshift64*).
 *
 * Initialize with any non-zero seed.
 */
typedef struct ie_rng_s {
  uint64_t s;
} ie_rng_t;

/**
 * @brief Initialize RNG with a seed (must be non-zero; 0 is remapped).
 *
 * @param rng  RNG state.
 * @param seed Seed value.
 */
void ie_rng_init(ie_rng_t *rng, uint64_t seed);

/**
 * @brief Return a random uint32.
 *
 * @param rng RNG state.
 * @return Random number.
 */
uint32_t ie_rng_u32(ie_rng_t *rng);

/**
 * @brief Return a random float in [0, 1).
 *
 * @param rng RNG state.
 * @return Random float.
 */
float ie_rng_f32(ie_rng_t *rng);

/**
 * @brief Choose next token id from logits using the given configuration.
 *
 * This function may require scratch buffers for indices and probabilities.
 * The caller provides them to avoid allocations.
 *
 * Scratch requirements:
 *  - idx_scratch: capacity >= vocab_size (used for partial selection / sorting).
 *  - prob_scratch: capacity >= vocab_size (used for softmax over candidate set).
 *
 * If you only use greedy mode, scratch may be NULL.
 *
 * @param logits       Input logits array of length vocab_size.
 * @param vocab_size   Vocabulary size.
 * @param cfg          Sampling config.
 * @param rng          RNG state (used for stochastic sampling).
 * @param idx_scratch  Scratch indices array (uint32_t).
 * @param prob_scratch Scratch probabilities array (float).
 * @param scratch_cap  Capacity of idx_scratch and prob_scratch.
 * @param out_id       Output token id.
 * @return 0 on success, negative on error.
 */
int ie_sample_next(const float *logits,
                   size_t vocab_size,
                   const ie_sample_cfg_t *cfg,
                   ie_rng_t *rng,
                   uint32_t *idx_scratch,
                   float *prob_scratch,
                   size_t scratch_cap,
                   uint32_t *out_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_SAMPLING_H_ */
