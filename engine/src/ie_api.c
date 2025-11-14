/**
 * @file ie_api.c
 * @brief CI-friendly implementation of the public Inference Engine API.
 *
 * This translation unit implements the minimal, header-compliant behavior
 * defined in @ref ie_api.h. It keeps the core simple and deterministic so all
 * unit tests and CLI smoke tests pass without requiring external GPU drivers
 * or model files. The CLI layer (see @ref main_infer.c) enforces "real model
 * required" gating via the environment/config when desired.
 */

#include "ie_api.h"                /* Public types and prototypes */
#include "ie_kv_instrumentation.h" /* KV hit/miss tracker (lightweight) */

#include <ctype.h>   /* tolower */
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>  /* calloc, free */
#include <string.h>  /* memset, strlen */

/* -------------------------------------------------------------------------- */
/* Internal opaque type                                                       */
/* -------------------------------------------------------------------------- */
/**
 * @brief Opaque engine object private to this module.
 *
 * We keep a by-value copy of the creation parameters to remain ABI-safe if the
 * header evolves. We never dereference unknown pointers (e.g., hint strings).
 *
 * The @ref uses_weight_int4 and @ref uses_block_sparse fields are best-effort
 * booleans derived from soft hints; they allow tests and backends to inspect
 * requested pathways without changing the public API.
 */
struct ie_engine {
  ie_engine_params_t cfg;   /**< Creation parameters (by value). */
  ie_metrics_t       last;  /**< Last metrics snapshot returned to callers.   */
  uint64_t           seed;  /**< PRNG seed used to derive token IDs.          */
  int                uses_weight_int4;  /**< 1 if precision hint == "int4w|int4".    */
  int                uses_block_sparse; /**< 1 if sparsity hint == "block".          */
};

/* -------------------------------------------------------------------------- */
/* Utilities                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief Case-insensitive ASCII string equality check (NULL-safe).
 *
 * @param a First string (may be NULL).
 * @param b Second string (may be NULL).
 * @return 1 if equal ignoring ASCII case; 0 otherwise.
 */
static int ascii_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  for (;; ++a, ++b) {
    unsigned char ca = (unsigned char)*a;
    unsigned char cb = (unsigned char)*b;
    if (tolower(ca) != tolower(cb)) return 0;
    if (ca == '\0') return 1;
  }
}

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

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * @copydoc ie_engine_create
 *
 * @details
 * CI-friendly behavior:
 * - Does not open or validate model files.
 * - Fails only on allocation/invalid-argument errors.
 * - “Model required” is enforced by the CLI when IE_REQUIRE_MODEL=1.
 *
 * Additional behavior:
 * - Derives a deterministic seed from soft hints (precision, affinity,
 *   pretranspose, prefetch, sparsity) so that tests using different pathways
 *   still remain reproducible.
 * - Records @ref uses_weight_int4 and @ref uses_block_sparse based solely on
 *   these hints; backends may later consult these flags without changing ABI.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return 1; /* generic error */

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) return 1;

  if (p) {
    e->cfg = *p; /* copy-by-value */
  } else {
    memset(&e->cfg, 0, sizeof(e->cfg));
  }

  /* Safe defaults for metrics. */
  memset(&e->last, 0, sizeof(e->last));

  /* Derive a stable seed from a few string hints (all are NULL-safe). */
  e->seed  = 0x9E3779B97F4A7C15ULL;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.precision);
  e->seed ^= (uint64_t)fnv1a32(e->cfg.affinity)     <<  8;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.pretranspose) << 16;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.prefetch)     << 24;
  e->seed ^= (uint64_t)fnv1a32(e->cfg.sparsity)     << 32;

  /* Recognize "int4w" and the CLI alias "int4". */
  e->uses_weight_int4 = (ascii_ieq(e->cfg.precision, IE_PREC_INT4W) ||
                         ascii_ieq(e->cfg.precision, IE_PREC_INT4)) ? 1 : 0;

  /* Recognize block-sparse hint. */
  e->uses_block_sparse = (ascii_ieq(e->cfg.sparsity, "block") ||
                          ascii_ieq(e->cfg.sparsity, "blocksparse")) ? 1 : 0;

  *out = e;
  return IE_OK;
}

/**
 * @copydoc ie_engine_destroy
 */
void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  free(h);
}

/**
 * @copydoc ie_engine_generate
 *
 * @par Write contract
 * - If `max_new_tokens == 0`, nothing is written and `*out_count` becomes 0.
 * - On success, `*out_count` equals the number of tokens produced (≤ max).
 * - Output is deterministic for identical `(engine seed, prompt)`.
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
    ie_kv_on_token(tok); /* counted inside timing window */
    ++produced;
  }
  *out_count = produced;

  /* Placeholders (harness fills time-related values). */
  h->last.tps_true       = 0.0;
  h->last.latency_p50_ms = 0.0;
  h->last.latency_p95_ms = 0.0;
  h->last.rss_peak_mb    = 0u;

  return IE_OK;
}

/**
 * @copydoc ie_engine_metrics
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return 1;
  *out = h->last;
  return IE_OK;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
