/* ========================================================================== */
/* File: engine/src/ie_kv_instrumentation.c                                   */
/* ========================================================================== */
/**
 * @file ie_kv_instrumentation.c
 * @brief Lightweight KV cache reuse counters (process-local, single-threaded).
 *
 * See @ref ie_kv_instrumentation.h for the high-level description.
 */

#include "ie_kv_instrumentation.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

/* ----------------------------------------------------------------------------
 * Internal state (process-local, single-threaded)
 * --------------------------------------------------------------------------*/
typedef struct {
  uint64_t hits;
  uint64_t misses;
  int      ready; /* 1 after begin_round(), 0 after finish_round() */
} ie_kv_state_t;

static ie_kv_state_t g_kv;

/* ----------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------*/
int ie_kv_begin_round(void) {
  g_kv.hits = 0;
  g_kv.misses = 0;
  g_kv.ready = 1;
  return 0;
}

void ie_kv_add_hits(uint64_t n) {
  if (!g_kv.ready) return;
  g_kv.hits += n;
}

void ie_kv_add_misses(uint64_t n) {
  if (!g_kv.ready) return;
  g_kv.misses += n;
}

void ie_kv_finish_round(uint64_t total_tokens,
                        uint64_t *out_hits,
                        uint64_t *out_misses) {
  assert(out_hits != NULL);
  assert(out_misses != NULL);

  if (!g_kv.ready) {
    /* Not initialized: conservative fallback. */
    *out_hits = 0;
    *out_misses = total_tokens;
    return;
  }

  uint64_t hits = g_kv.hits;
  uint64_t misses = g_kv.misses;

  /* Fallback if no events were recorded but caller claims progress. */
  if ((hits + misses) == 0 && total_tokens > 0) {
    misses = total_tokens;
  }

  *out_hits = hits;
  *out_misses = misses;

  g_kv.hits = 0;
  g_kv.misses = 0;
  g_kv.ready = 0;
}

/* ========================================================================== */
void ie_kv_on_token(uint32_t token) {
  (void)token;
}

/* End of file                                                                */
/* ========================================================================== */
