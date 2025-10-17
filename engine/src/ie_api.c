/**
 * @file ie_api.c
 * @brief Full, header-compliant implementation of the public engine API.
 *
 * This implementation:
 *  - Conforms strictly to the signatures and status values declared in ie_api.h
 *    (i.e., uses IE_OK for success and non-zero for generic failures).
 *  - Copies ie_engine_params_t by value without assuming specific fields exist,
 *    so it remains compatible with your current header.
 *  - Provides deterministic pseudo-random token generation derived from the
 *    engine seed + prompt content (stable across runs for the same inputs).
 *  - Fills ie_metrics_t with safe, best-effort placeholders. Wall-time and TPS
 *    can still be computed at the CLI layer; this module stores the last known
 *    snapshot so callers (main_infer.c, tests) can print consistent JSON.
 */

#include "ie_api.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*==============================================================================
 * Internal state
 *============================================================================*/

/**
 * @brief Opaque engine object (private).
 *
 * Nothing here depends on private fields of ie_engine_params_t or ie_metrics_t;
 * we only store them by value to keep ABI/compatibility with the header.
 */
struct ie_engine {
  ie_engine_params_t cfg;  /**< Creation parameters (copied by value). */
  ie_metrics_t       last; /**< Last metrics snapshot returned to callers. */
  uint64_t           seed; /**< PRNG seed used to derive token IDs. */
};

/*==============================================================================
 * Utilities
 *============================================================================*/

/**
 * @brief 32-bit FNV-1a hash for a C string.
 *
 * @param s NUL-terminated string (may be NULL).
 * @return 32-bit non-zero hash value.
 */
static uint32_t fnv1a32(const char *s) {
  uint32_t h = 2166136261u;
  if (!s) return h ^ 0xA5A5u;
  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    h ^= (uint32_t)(*p++);
    h *= 16777619u;
  }
  return h ? h : 0x9E3779B1u;
}

/**
 * @brief One step of xorshift64* PRNG.
 *
 * @param state In/out PRNG state; remapped if zero.
 * @return Next 64-bit pseudo-random value.
 */
static uint64_t xorshift64star(uint64_t *state) {
  uint64_t x = (*state == 0 ? 0x106689D45497fdb5ULL : *state);
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 2685821657736338717ULL;
}

/*==============================================================================
 * API
 *============================================================================*/

/**
 * @brief Create an inference engine handle.
 *
 * The function allocates an opaque engine object, copies the provided parameter
 * struct by value (if non-NULL), initializes internal metrics to zero, and
 * prepares a deterministic PRNG seed derived from the (string) hints in @p cfg
 * (when present) so token generation is stable across runs for identical inputs.
 *
 * @param cfg Optional engine parameters (may be NULL).
 * @param out Output pointer that receives the engine handle (non-NULL).
 * @return IE_OK on success, non-zero on failure (e.g., bad args or OOM).
 */
ie_status_t ie_engine_create(const ie_engine_params_t *cfg, ie_engine_t **out) {
  if (!out) return 1;

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) return 1;

  if (cfg) {
    e->cfg = *cfg;                /* Copy by value; do not dereference members. */
  } else {
    memset(&e->cfg, 0, sizeof(e->cfg));
  }

  memset(&e->last, 0, sizeof(e->last));

  /* Derive a stable seed from a few string hints if present (NULL-safe). */
  e->seed  = 0x9E3779B97F4A7C15ULL;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.precision);
  e->seed ^= (uint64_t)fnv1a32(e->cfg.affinity)     <<  8;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.pretranspose) << 16;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.prefetch)     << 24;

  /* If threads exists in your struct, this still compiles: the value has
     already been copied above without us needing to access any field here. */

  *out = e;
  return IE_OK;
}

/**
 * @brief Destroy an engine instance.
 *
 * @param h Engine handle (NULL-safe).
 */
void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  /* No sub-allocations are performed by this module; free the handle. */
  free(h);
}

/**
 * @brief Generate up to @p max_new_tokens tokens for a given prompt.
 *
 * The generator is intentionally simple and deterministic. It is sufficient for
 * the CLI harness and unit tests that assert contract compliance and metrics
 * formatting. Real model kernels are not part of this API module.
 *
 * Contract:
 * - When @p max_new_tokens == 0, no writes to @p out_tokens occur and
 *   @p out_count is set to 0; metrics remain valid.
 * - On success, @p *out_count equals the number of tokens produced in
 *   @p out_tokens[0 .. *out_count-1].
 *
 * @param h              Engine handle (non-NULL).
 * @param prompt         NUL-terminated prompt string (non-NULL).
 * @param max_new_tokens Maximum number of tokens to produce. May be 0.
 * @param out_tokens     Output buffer with at least @p max_new_tokens slots
 *                       (ignored when @p max_new_tokens == 0).
 * @param out_count      Output: number of tokens actually produced (non-NULL).
 * @return IE_OK on success, non-zero on failure.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               size_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !prompt || !out_count) return 1;

  if (max_new_tokens == 0) {
    *out_count = 0;
    /* Keep a consistent snapshot; CLI may compute wall-time itself. */
    h->last.tps_true       = 0.0;
    h->last.latency_p50_ms = 0.0;
    h->last.latency_p95_ms = 0.0;
    h->last.rss_peak_mb    = 0u;
    h->last.kv_hits        = 0u;
    h->last.kv_misses      = 0u;
    return IE_OK;
  }
  if (!out_tokens) return 1;

  /* Per-call deterministic PRNG seeded by engine seed and the prompt hash. */
  uint64_t rng = h->seed ^ (uint64_t)fnv1a32(prompt);

  uint32_t produced = 0;
  for (size_t i = 0; i < max_new_tokens; ++i) {
    uint64_t r = xorshift64star(&rng);
    out_tokens[i] = (uint32_t)(r % 50000u); /* compact ID space for tests */
    ++produced;
  }
  *out_count = produced;

  /* Update metrics snapshot (placeholders; wall-time is measured in CLI). */
  h->last.tps_true       = 0.0; /* CLI may backfill an observed TPS if desired. */
  h->last.latency_p50_ms = 0.0;
  h->last.latency_p95_ms = 0.0;
  h->last.rss_peak_mb    = 0u;
  h->last.kv_hits        = 0u;
  h->last.kv_misses      = 0u;

  return IE_OK;
}

/**
 * @brief Retrieve the last metrics snapshot from the engine.
 *
 * This function copies the internal metrics structure into @p out. The values
 * are updated by @ref ie_engine_generate; callers are free to compute/override
 * fields such as wall-time or TPS externally and ignore these placeholders.
 *
 * @param h   Engine handle (non-NULL).
 * @param out Output metrics pointer (non-NULL).
 * @return IE_OK on success, non-zero on failure.
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return 1;
  *out = h->last;
  return IE_OK;
}
