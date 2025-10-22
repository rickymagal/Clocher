/* ========================================================================== */
/* File: engine/src/ie_api.c                                                  */
/* ========================================================================== */
/**
 * @file ie_api.c
 * @brief Header-compliant, CI-friendly implementation of the public engine API.
 *
 * ## Design goals
 * - **Strict compatibility** with @ref ie_api.h (types, signatures, status codes).
 * - **Deterministic behavior**: pseudo-random token IDs derived from a per-engine
 *   seed and the input prompt hash — stable across runs for identical inputs.
 * - **CI-friendly**: never depends on external model files here; the “real model
 *   required” gating is performed by the CLI layer (@ref main_infer.c) via
 *   `IE_REQUIRE_MODEL`.
 * - **Safe metrics**: always returns a valid @ref ie_metrics_t snapshot; callers
 *   (CLI/tests) compute wall-time/TPS and may override fields as needed.
 *
 * This module has no dependencies beyond the C standard library and the public
 * headers. It is intended to compile cleanly with `-std=c11 -Wall -Wextra
 * -Werror -pedantic`.
 */

#include "ie_api.h"   /* Public types and prototypes: ie_engine_*, ie_metrics_t */

#include <inttypes.h> /* PRIu64 etc. */
#include <stdint.h>
#include <stdlib.h>   /* calloc, free */
#include <string.h>   /* memset */

/* =============================================================================
 * Internal opaque type
 * ============================================================================= */
/**
 * @brief Opaque engine object (private to this translation unit).
 *
 * We keep a by-value copy of the creation parameters to remain ABI-safe if the
 * header evolves. We **never** dereference unknown pointers (e.g., hint strings).
 */
struct ie_engine {
  ie_engine_params_t cfg;   /**< Creation parameters (by value). */
  ie_metrics_t       last;  /**< Last metrics snapshot returned to callers. */
  uint64_t           seed;  /**< PRNG seed used to derive token IDs. */
};

/* =============================================================================
 * Internal utilities
 * ============================================================================= */
/**
 * @brief 32-bit FNV-1a hash for C strings (NULL-safe).
 *
 * @param s NUL-terminated string; `NULL` is accepted and hashed to a fixed value.
 * @return Non-zero 32-bit hash value (remapped if the result would be 0).
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
 * @brief One step of a xorshift64* PRNG (period ≈ 2^64-1).
 *
 * @param state In/out PRNG state. If zero, it is remapped to a fixed non-zero seed.
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

/* =============================================================================
 * Public API implementation
 * ============================================================================= */

/**
 * @copydoc ie_engine_create
 *
 * @details
 * CI-friendly behavior:
 * - Does **not** open or validate model files.
 * - Fails only on allocation/invalid-argument errors.
 * - “Model required” is enforced by the CLI when `IE_REQUIRE_MODEL=1`
 *   (outside of this module).
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return 1;

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) return 1;

  if (p) {
    /* Copy by value to preserve ABI compatibility if the header evolves. */
    e->cfg = *p;
  } else {
    memset(&e->cfg, 0, sizeof(e->cfg));
  }

  /* Initialize metrics snapshot to safe defaults. */
  memset(&e->last, 0, sizeof(e->last));

  /* Derive a stable seed from a few string hints (all are NULL-safe). */
  e->seed  = 0x9E3779B97F4A7C15ULL;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.precision);
  e->seed ^= (uint64_t)fnv1a32(e->cfg.affinity)     <<  8;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.pretranspose) << 16;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.prefetch)     << 24;

  *out = e;
  return IE_OK;
}

/**
 * @copydoc ie_engine_destroy
 */
void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  /* No nested allocations beyond the handle itself. */
  free(h);
}

/**
 * @copydoc ie_engine_generate
 *
 * @par Write contract
 * - If `max_new_tokens == 0`, nothing is written to `out_tokens` and
 *   `*out_count` is set to 0; returns `IE_OK`.
 * - On success, `*out_count` equals the number of tokens produced (≤ max).
 * - Output is deterministic for identical `(engine seed, prompt)`.
 *
 * @par Metrics note
 * This module does **not** measure time; callers (CLI/harness) compute wall-time
 * and TPS. Here we zero `tps_true/latency_*` and leave `rss_peak_mb/kv_*` at 0,
 * allowing upper layers to populate those fields.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               size_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_count || !prompt) return 1;

  if (max_new_tokens == 0) {
    *out_count = 0;
    /* Keep a coherent metrics snapshot for callers. */
    h->last.tps_true       = 0.0;
    h->last.latency_p50_ms = 0.0;
    h->last.latency_p95_ms = 0.0;
    h->last.rss_peak_mb    = 0u;
    h->last.kv_hits        = 0u;
    h->last.kv_misses      = 0u;
    return IE_OK;
  }
  if (!out_tokens) return 1;

  /* Per-call deterministic PRNG seeded by engine seed XOR prompt hash. */
  uint64_t rng = h->seed ^ (uint64_t)fnv1a32(prompt);

  uint32_t produced = 0;
  for (size_t i = 0; i < max_new_tokens; ++i) {
    const uint64_t r = xorshift64star(&rng);
    out_tokens[i] = (uint32_t)(r % 50000u); /* Compact ID space for tests. */
    ++produced;
  }
  *out_count = produced;

  /* Metrics snapshot placeholders (harness fills real values). */
  h->last.tps_true       = 0.0;
  h->last.latency_p50_ms = 0.0;
  h->last.latency_p95_ms = 0.0;
  h->last.rss_peak_mb    = 0u;
  h->last.kv_hits        = 0u;
  h->last.kv_misses      = 0u;

  return IE_OK;
}

/**
 * @copydoc ie_engine_metrics
 *
 * @note Callers should **zero-initialize** `*out` before calling to preserve
 *       forward compatibility (as documented in the header).
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return 1;
  *out = h->last;
  return IE_OK;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
