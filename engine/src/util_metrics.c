/**
 * @file util_metrics.c
 * @brief Implementation of runtime metrics (KV counters + RSS sampling).
 *
 * @details
 * This module backs the public API in ie_metrics.h. It provides:
 *  - Process-wide accumulators for KV cache hits/misses using C11 atomics.
 *  - A best-effort function to sample peak RSS for reporting purposes.
 */

#include "ie_metrics.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#if defined(__linux__)
  #include <sys/resource.h>
  #include <unistd.h>
#elif defined(__APPLE__)
  #include <sys/resource.h>
#else
  /* Other platforms: leave only very conservative fallbacks. */
  #include <sys/resource.h>
#endif

/* -------------------------------------------------------------------------- */
/* Global accumulators (process-wide).                                        */
/* -------------------------------------------------------------------------- */

static _Atomic uint64_t g_kv_hits = 0;
static _Atomic uint64_t g_kv_miss = 0;

/* -------------------------------------------------------------------------- */
/* KV counters API                                                             */
/* -------------------------------------------------------------------------- */

void ie_metrics_reset(void) {
  atomic_store_explicit(&g_kv_hits, 0, memory_order_relaxed);
  atomic_store_explicit(&g_kv_miss, 0, memory_order_relaxed);
}

void ie_metrics_add_kv_hit(uint64_t n) {
  if (!n) return;
  (void)atomic_fetch_add_explicit(&g_kv_hits, n, memory_order_relaxed);
}

void ie_metrics_add_kv_miss(uint64_t n) {
  if (!n) return;
  (void)atomic_fetch_add_explicit(&g_kv_miss, n, memory_order_relaxed);
}

void ie_metrics_add_kv(uint64_t hits, uint64_t misses) {
  if (hits)   (void)atomic_fetch_add_explicit(&g_kv_hits, hits,  memory_order_relaxed);
  if (misses) (void)atomic_fetch_add_explicit(&g_kv_miss, misses, memory_order_relaxed);
}

void ie_metrics_snapshot(ie_metrics_t* out, int reset_after) {
  if (!out) return;
  const uint64_t hits = atomic_load_explicit(&g_kv_hits, memory_order_relaxed);
  const uint64_t miss = atomic_load_explicit(&g_kv_miss, memory_order_relaxed);

  out->kv_hits   = hits;
  out->kv_misses = miss;
  out->rsvd0     = 0;

  if (reset_after) {
    atomic_store_explicit(&g_kv_hits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_kv_miss, 0, memory_order_relaxed);
  }
}

/* -------------------------------------------------------------------------- */
/* RSS peak sampling                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Parse integer value (kB) from a "Key:  value kB" style line.
 * @param line NUL-terminated text line.
 * @param key  Key prefix to match, e.g., "VmHWM:".
 * @param out_kb Receives the parsed value in kilobytes if successful.
 * @return 1 on success, 0 otherwise.
 */
static int parse_status_kb_line(const char* line, const char* key, uint64_t* out_kb) {
  if (!line || !key || !out_kb) return 0;
  size_t klen = strlen(key);
  if (strncmp(line, key, klen) != 0) return 0;

  /* Expected format: "Key:\t  12345 kB" (spaces/tabs vary) */
  const char* p = line + klen;
  while (*p == ' ' || *p == '\t') ++p;

  /* Read number */
  uint64_t val = 0;
  int matched = 0;
  while (*p >= '0' && *p <= '9') {
    matched = 1;
    val = val * 10 + (uint64_t)(*p - '0');
    ++p;
  }
  if (!matched) return 0;

  /* Skip spaces/tabs */
  while (*p == ' ' || *p == '\t') ++p;

  /* Optional unit "kB" */
  if ((p[0] == 'k' || p[0] == 'K') && (p[1] == 'B' || p[1] == 'b')) {
    /* ok */
  }

  *out_kb = val;
  return 1;
}

uint32_t ie_metrics_sample_rss_peak(void) {
  /* --- Linux: try /proc/self/status VmHWM (peak resident) ----------------- */
#if defined(__linux__)
  {
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
      char buf[512];
      while (fgets(buf, sizeof(buf), f)) {
        uint64_t kb = 0;
        if (parse_status_kb_line(buf, "VmHWM:", &kb)) {
          fclose(f);
          return (uint32_t)(kb / 1024u); /* kB -> MB */
        }
      }
      fclose(f);
    }
  }
  /* Fallback: getrusage ru_maxrss (on Linux it's in kilobytes). */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t kb = (uint64_t)ru.ru_maxrss;
      return (uint32_t)(kb / 1024u); /* kB -> MB */
    }
  }
#elif defined(__APPLE__)
  /* On macOS, getrusage.ru_maxrss is in bytes. */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t bytes = (uint64_t)ru.ru_maxrss;
      return (uint32_t)(bytes / (1024u * 1024u));
    }
  }
#else
  /* Other platforms: best-effort via getrusage; units vary by OS, conservatively treat as kB. */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t kb = (uint64_t)ru.ru_maxrss;
      return (uint32_t)(kb / 1024u);
    }
  }
#endif

  /* Unknown platform or failure. */
  (void)errno; /* silence -Wunused-parameter if errno isn't used above */
  return 0u;
}
