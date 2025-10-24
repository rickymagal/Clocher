/* ========================================================================== */
/* File: engine/src/util_metrics.c                                            */
/* ========================================================================== */
/**
 * @file util_metrics.c
 * @brief Runtime metrics utilities: KV-cache counters and peak RSS sampling.
 *
 * This module implements the public API declared in @ref ie_metrics.h:
 *  - Process-wide accumulators for KV-cache hits/misses using C11 atomics.
 *  - A best-effort routine to sample **peak** process RSS and return it as
 *    MiB (rounded up) for stable reporting in benchmarks.
 *
 * ## Platform notes (units and data sources)
 * - **Linux**
 *   - `getrusage(RUSAGE_SELF).ru_maxrss` → **KiB**, *peak*.
 *   - `/proc/self/status` → `VmHWM:` → **kB**, *peak* (alternative source).
 *   - `/proc/self/statm` → resident pages × page size → **bytes**, *current*
 *     (last-resort; not a peak, only used if other sources fail).
 * - **macOS**
 *   - `getrusage(...).ru_maxrss` → **bytes**, *peak*.
 * - **Other POSIX-like systems**
 *   - `getrusage(...).ru_maxrss` units are OS-dependent; this module
 *     conservatively interprets them as KiB.
 *
 * All conversions normalize to **MiB** and use **ceil** rounding to avoid
 * reporting “0 MB” for small-but-non-zero peaks.
 */

#include "ie_metrics.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__linux__)
  #include <sys/resource.h>
  #include <unistd.h>
#elif defined(__APPLE__) || defined(__MACH__)
  #include <sys/resource.h>
#else
  #include <sys/resource.h>
#endif

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Convert KiB to MiB, rounding up.
 * @param kib Value in kibibytes.
 * @return Value in mebibytes (ceil).
 */
static inline unsigned kib_to_mib_ceil(uint64_t kib) {
  return (unsigned)((kib + 1023u) / 1024u);
}

/**
 * @brief Convert bytes to MiB, rounding up.
 * @param bytes Value in bytes.
 * @return Value in mebibytes (ceil).
 */
static inline unsigned bytes_to_mib_ceil(uint64_t bytes) {
  const uint64_t mask = (1u << 20) - 1u; /* 1 MiB - 1 */
  return (unsigned)((bytes + mask) >> 20);
}

/**
 * @brief Parse integer value (kB) from a `/proc/self/status` style line.
 *
 * Expected formats (whitespace varies):
 * - `"VmHWM:\t  123456 kB"`
 * - `"VmRSS:\t  654321 kB"`
 *
 * @param line   NUL-terminated input line.
 * @param key    Key prefix to match (e.g., `"VmHWM:"`).
 * @param out_kb Receives the parsed value in kB on success.
 * @return 1 on success, 0 otherwise.
 */
static int parse_status_kb_line(const char *line, const char *key, uint64_t *out_kb) {
  if (!line || !key || !out_kb) return 0;

  const size_t klen = strlen(key);
  if (strncmp(line, key, klen) != 0) return 0;

  /* Skip whitespace after the key */
  const char *p = line + klen;
  while (*p == ' ' || *p == '\t') ++p;

  /* Parse number */
  uint64_t val = 0;
  int matched = 0;
  while (*p >= '0' && *p <= '9') {
    matched = 1;
    val = val * 10 + (uint64_t)(*p - '0');
    ++p;
  }
  if (!matched) return 0;

  /* Optional whitespace then optional "kB" unit (ignored) */
  *out_kb = val;
  return 1;
}

#if defined(__linux__)
/**
 * @brief Linux-only: obtain peak RSS in KiB via multiple strategies.
 *
 * Preference order:
 *  1. `getrusage(...).ru_maxrss` (KiB, **peak**).
 *  2. `/proc/self/status` → `VmHWM:` (kB, **peak**).
 *  3. `/proc/self/statm` (current RSS only; last resort).
 *
 * @param out_kib Receives the peak RSS in KiB on success.
 * @return 1 on success, 0 otherwise.
 */
static int linux_rss_peak_kib(uint64_t *out_kib) {
  if (!out_kib) return 0;

  /* 1) getrusage: ru_maxrss in KiB on Linux */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t kib = (uint64_t)ru.ru_maxrss; /* KiB */
      if (kib > 0) { *out_kib = kib; return 1; }
    }
  }

  /* 2) /proc/self/status → VmHWM (kB, peak) */
  {
    FILE *f = fopen("/proc/self/status", "r");
    if (f) {
      char buf[512];
      while (fgets(buf, sizeof(buf), f)) {
        uint64_t kb = 0;
        if (parse_status_kb_line(buf, "VmHWM:", &kb)) {
          fclose(f);
          *out_kib = kb; /* treat kB ≈ KiB for our purposes */
          return 1;
        }
      }
      fclose(f);
    }
  }

  /* 3) /proc/self/statm (CURRENT RSS; not peak) as last resort */
  {
    FILE *f = fopen("/proc/self/statm", "r");
    if (f) {
      unsigned long pages_total = 0, pages_resident = 0;
      if (fscanf(f, "%lu %lu", &pages_total, &pages_resident) == 2) {
        long page_sz = sysconf(_SC_PAGESIZE);
        fclose(f);
        if (page_sz > 0) {
          uint64_t bytes = (uint64_t)pages_resident * (uint64_t)page_sz;
          *out_kib = bytes >> 10; /* bytes → KiB */
          return 1;
        }
      } else {
        fclose(f);
      }
    }
  }

  return 0;
}
#endif /* __linux__ */

/* -------------------------------------------------------------------------- */
/* Global accumulators (process-wide)                                         */
/* -------------------------------------------------------------------------- */

/** @brief Process-wide KV hit counter. */
static _Atomic uint64_t g_kv_hits = 0;
/** @brief Process-wide KV miss counter. */
static _Atomic uint64_t g_kv_miss = 0;

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Reset the process-wide KV hit/miss counters to zero.
 *
 * Thread-safety: uses relaxed C11 atomics.
 */
void ie_metrics_reset(void) {
  atomic_store_explicit(&g_kv_hits, 0, memory_order_relaxed);
  atomic_store_explicit(&g_kv_miss, 0, memory_order_relaxed);
}

/**
 * @brief Add @p n to the global KV **hit** counter.
 * @param n Increment amount; no-op if zero.
 */
void ie_metrics_add_kv_hit(uint64_t n) {
  if (!n) return;
  (void)atomic_fetch_add_explicit(&g_kv_hits, n, memory_order_relaxed);
}

/**
 * @brief Add @p n to the global KV **miss** counter.
 * @param n Increment amount; no-op if zero.
 */
void ie_metrics_add_kv_miss(uint64_t n) {
  if (!n) return;
  (void)atomic_fetch_add_explicit(&g_kv_miss, n, memory_order_relaxed);
}

/**
 * @brief Add to both KV counters in a single call.
 * @param hits   Increment for the hit counter (may be 0).
 * @param misses Increment for the miss counter (may be 0).
 */
void ie_metrics_add_kv(uint64_t hits, uint64_t misses) {
  if (hits)   (void)atomic_fetch_add_explicit(&g_kv_hits,  hits,   memory_order_relaxed);
  if (misses) (void)atomic_fetch_add_explicit(&g_kv_miss, misses, memory_order_relaxed);
}

/**
 * @brief Snapshot the current KV counters into @p out, optionally resetting.
 *
 * On return:
 *  - `out->kv_hits`   receives the current hit count.
 *  - `out->kv_misses` receives the current miss count.
 *  - `out->rsvd0`     is set to 0.
 *
 * @param out         Output structure (must not be `NULL`).
 * @param reset_after If non-zero, counters are reset to zero after snapshot.
 */
void ie_metrics_snapshot(ie_metrics_t *out, int reset_after) {
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

/**
 * @brief Sample **peak** RSS and return it in MiB (ceil).
 *
 * This routine should be called **outside** the timed benchmark window
 * (the caller controls where it’s invoked).
 *
 * Platform behavior:
 *  - **Linux**: prefer `getrusage()` (KiB, peak), fall back to
 *    `/proc/self/status` `VmHWM:` (kB, peak), then `/proc/self/statm`
 *    (current RSS only) as a last resort.
 *  - **macOS**: use `getrusage()` (bytes, peak).
 *  - **Other**: best-effort `getrusage()` assuming KiB.
 *
 * @return Peak RSS in MiB (rounded up), or 0 if unavailable.
 */
uint32_t ie_metrics_sample_rss_peak(void) {
#if defined(__linux__)
  uint64_t kib = 0;
  if (linux_rss_peak_kib(&kib)) {
    return (uint32_t)kib_to_mib_ceil(kib);
  }
  return 0u;

#elif defined(__APPLE__) || defined(__MACH__)
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      /* macOS: ru_maxrss is BYTES */
      const uint64_t bytes = (uint64_t)ru.ru_maxrss;
      return (uint32_t)bytes_to_mib_ceil(bytes);
    }
  }
  return 0u;

#else
  /* Other platforms: conservative assumption ru_maxrss is KiB. */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t kib = (uint64_t)ru.ru_maxrss;
      return (uint32_t)kib_to_mib_ceil(kib);
    }
  }
  return 0u;
#endif
}

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
