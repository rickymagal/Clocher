/* ========================================================================== */
/* File: engine/src/ie_kv_instrumentation.c                                   */
/* ========================================================================== */
/**
 * @file ie_kv_instrumentation.c
 * @brief FIFO-set based KV cache hit/miss counters (single-threaded).
 *
 * See @ref ie_kv_instrumentation.h for the high-level description.
 */

#include "ie_kv_instrumentation.h"

#include <assert.h>
#include <stddef.h>  /* NULL */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------------------------------------------------------
 * Internal state (process-local, single-threaded)
 * --------------------------------------------------------------------------*/
typedef struct {
  uint32_t *slots;     /* FIFO ring buffer of token IDs */
  size_t    capacity;  /* max number of elements in the set */
  size_t    count;     /* current number of valid elements (<= capacity) */
  size_t    head;      /* index of the oldest element for eviction (0..capacity-1) */
  uint64_t  hits;      /* accumulated hits in the round */
  uint64_t  misses;    /* accumulated misses in the round */
  int       ready;     /* 1 after begin_round(), 0 after finish_round() */
} ie_kv_state_t;

static ie_kv_state_t g_kv;

/* ----------------------------------------------------------------------------
 * Utilities
 * --------------------------------------------------------------------------*/
/**
 * @brief Strict-decimal environment reader with fallback.
 *
 * @param name Env var name.
 * @param defv Default value when unset/invalid.
 * @return Parsed non-negative value.
 */
static size_t env_size(const char *name, size_t defv) {
  const char *s = getenv(name);
  if (!s || !*s) return defv;
  char *end = NULL;
  long long v = strtoll(s, &end, 10);
  if (end == s || (end && *end)) return defv;
  if (v <= 0) return defv;
  return (size_t)v;
}

/**
 * @brief Linear membership test in the FIFO set.
 *
 * @param token Token to check.
 * @return 1 if found, 0 otherwise.
 */
static int kv_contains(uint32_t token) {
  for (size_t i = 0; i < g_kv.count; ++i) {
    if (g_kv.slots[i] == token) return 1;
  }
  return 0;
}

/**
 * @brief Insert token into the FIFO set, evicting the oldest if full.
 *
 * @param token Token to insert.
 */
static void kv_insert_fifo(uint32_t token) {
  if (g_kv.capacity == 0) return;
  if (g_kv.count < g_kv.capacity) {
    g_kv.slots[g_kv.count++] = token;
  } else {
    /* Evict oldest at head, write new token, advance head */
    g_kv.slots[g_kv.head] = token;
    g_kv.head = (g_kv.head + 1) % g_kv.capacity;
  }
}

/* ----------------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------------*/
int ie_kv_begin_round(void) {
  /* Reset counters and (re)allocate set as needed. */
  const size_t cap = env_size("IE_KV_CAP", 4096); /* default: 4K tokens */
  if (g_kv.slots && g_kv.capacity != cap) {
    free(g_kv.slots);
    g_kv.slots = NULL;
  }
  if (!g_kv.slots) {
    g_kv.slots = (uint32_t *)malloc(sizeof(uint32_t) * cap);
    if (!g_kv.slots) {
      g_kv.capacity = 0;
      g_kv.count = g_kv.head = 0;
      g_kv.hits = g_kv.misses = 0;
      g_kv.ready = 0;
      return -1;
    }
  }
  g_kv.capacity = cap;
  g_kv.count = 0;
  g_kv.head = 0;
  g_kv.hits = 0;
  g_kv.misses = 0;
  g_kv.ready = 1;
  return 0;
}

void ie_kv_on_token(uint32_t token) {
  if (!g_kv.ready || g_kv.capacity == 0) return;
  if (kv_contains(token)) {
    g_kv.hits++;
    return;
  }
  g_kv.misses++;
  kv_insert_fifo(token);
}

void ie_kv_finish_round(uint64_t total_tokens,
                        uint64_t *out_hits,
                        uint64_t *out_misses) {
  assert(out_hits   != NULL);
  assert(out_misses != NULL);
  if (!g_kv.ready) {
    /* Not initialized: conservative fallback. */
    *out_hits = 0;
    *out_misses = total_tokens;
    return;
  }
  /* Fallback if no tokens were observed but caller claims otherwise. */
  uint64_t hits   = g_kv.hits;
  uint64_t misses = g_kv.misses;
  if ((hits + misses) == 0 && total_tokens > 0) {
    misses = total_tokens;
  }

  *out_hits   = hits;
  *out_misses = misses;

  /* Free the set to keep memory footprint low across short-lived runs. */
  free(g_kv.slots);
  g_kv.slots = NULL;
  g_kv.capacity = g_kv.count = g_kv.head = 0;
  g_kv.hits = g_kv.misses = 0;
  g_kv.ready = 0;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
