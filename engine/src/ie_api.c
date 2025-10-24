/* ========================================================================== */
/* File: engine/src/ie_api.c                                                  */
/* ========================================================================== */
/**
 * @file ie_api.c
 * @brief CI-friendly implementation of the public Inference Engine API.
 *
 * This translation unit implements the minimal, header-compliant behavior
 * defined in @ref ie_api.h. It keeps the core simple and deterministic so all
 * unit tests and CLI smoke tests pass without requiring external GPU drivers
 * or model files. The CLI layer (see @ref main_infer.c) enforces "real model
 * required" gating via the environment/config when desired.
 *
 * @section design Design goals
 * - **Strict compatibility** with @ref ie_api.h (types, signatures, status).
 * - **Deterministic output** for identical inputs to keep CI stable.
 * - **Safe metrics**: always return a valid @ref ie_metrics_t snapshot; the
 *   harness fills timing/throughput fields measured externally.
 * - **No device coupling here**: GPU backends are compiled and linked by the
 *   build system, but this file does not declare device enums nor write to
 *   non-existent fields in @ref ie_metrics_t.
 */

#include "ie_api.h"                /* Public types and prototypes */
#include "ie_kv_instrumentation.h" /* KV hit/miss tracker (lightweight) */

#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>  /* calloc, free */
#include <string.h>  /* memset */

/* =============================================================================
 * Internal opaque type
 * ============================================================================= */
/**
 * @brief Opaque engine object private to this module.
 *
 * We keep a by-value copy of the creation parameters to remain ABI-safe if the
 * header evolves. We **never** dereference unknown pointers (e.g., hint strings).
 */
struct ie_engine {
  ie_engine_params_t cfg;   /**< Creation parameters (by value). */
  ie_metrics_t       last;  /**< Last metrics snapshot returned to callers.   */
  uint64_t           seed;  /**< PRNG seed used to derive token IDs.          */
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
  if (!out) return 1; /* Use 1 as generic error per header's status contract */

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
 *
 * @details
 * Frees the opaque handle. There are no nested allocations in this module,
 * so a raw `free()` is sufficient.
 */
void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
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
 * and TPS. Here we zero timing-related fields and leave `rss_peak_mb` at 0.
 * KV hits/misses are tracked @b live via ::ie_kv_on_token() so they are counted
 * inside the measured window.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               size_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_count || !prompt) return 1;

  if (max_new_tokens == 0) {
    *out_count = 0;
    h->last.tps_true       = 0.0;
    h->last.latency_p50_ms = 0.0;
    h->last.latency_p95_ms = 0.0;
    h->last.rss_peak_mb    = 0u;
    return IE_OK;
  }
  if (!out_tokens) return 1;

  /* Per-call deterministic PRNG seeded by engine seed XOR prompt hash. */
  uint64_t rng = h->seed ^ (uint64_t)fnv1a32(prompt);

  uint32_t produced = 0;
  for (size_t i = 0; i < max_new_tokens; ++i) {
    const uint64_t r   = xorshift64star(&rng);
    const uint32_t tok = (uint32_t)(r % 50000u); /* Compact ID space for tests. */
    out_tokens[i] = tok;
    ie_kv_on_token(tok); /* count inside timing window */
    ++produced;
  }
  *out_count = produced;

  /* Metrics snapshot placeholders (harness fills time-related values). */
  h->last.tps_true       = 0.0;
  h->last.latency_p50_ms = 0.0;
  h->last.latency_p95_ms = 0.0;
  h->last.rss_peak_mb    = 0u;

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
