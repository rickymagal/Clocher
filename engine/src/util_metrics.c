/**
 * @file util_metrics.c
 * @brief Implementation of runtime metrics (KV counters + RSS sampling).
 *
 * @details
 * This module backs the helpers declared in @ref util_metrics.h:
 *  - Process-wide accumulators for KV hits/misses using C11 atomics.
 *  - A best-effort function that samples the process peak RSS in MiB.
 *
 * Environment:
 *  - Set `IE_DEBUG_RSS=1` to log which source (VmHWM/VmRSS/smaps_rollup/
 *    getrusage/mach) was used and its raw units before conversion.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "util_metrics.h"

#include <errno.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
  #include <unistd.h>
  #include <sys/resource.h>
#elif defined(__APPLE__)
  #include <sys/resource.h>
  #include <mach/mach.h>
#else
  #include <sys/resource.h>
#endif

#include <strings.h> /* strcasecmp */

/* -------------------------------------------------------------------------- */
/* Globals: KV accumulators (process-wide)                                     */
/* -------------------------------------------------------------------------- */

/** @brief Process-wide KV hit counter (relaxed atomic). */
static _Atomic uint64_t g_kv_hits = 0;

/** @brief Process-wide KV miss counter (relaxed atomic). */
static _Atomic uint64_t g_kv_miss = 0;

/* -------------------------------------------------------------------------- */
/* Local helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return 1 if the environment variable @p name is a truthy flag.
 *
 * Accepted truthy values (case-insensitive):
 *  - "1", "true", "yes", "on"
 *
 * @param name Environment variable name.
 * @return 1 if truthy, 0 otherwise.
 */
static int env_flag(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0;
  return (!strcasecmp(s, "1") || !strcasecmp(s, "true") ||
          !strcasecmp(s, "yes") || !strcasecmp(s, "on"));
}

/**
 * @brief Debug print controlled by `IE_DEBUG_RSS=1`.
 * @param fmt printf-style format string.
 */
static void dbg(const char *fmt, ...) {
  if (!env_flag("IE_DEBUG_RSS")) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/**
 * @brief Round up KiB to MiB using ceil(), clamped to 32-bit.
 * @param kib Size in KiB.
 * @return Size in MiB (ceil), or 0 if @p kib is 0.
 */
static uint32_t kib_to_mib_ceil(uint64_t kib) {
  if (kib == 0) return 0u;
  /* ceil(kib/1024) = (kib + 1023) >> 10 */
  const uint64_t mib = (kib + 1023u) >> 10;
  return (mib > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)mib;
}

/**
 * @brief Parse an integer KiB value from a "Key:  12345 kB" style procfs line.
 *
 * @param line Input line (NUL-terminated).
 * @param key  Key prefix to match exactly (e.g., "VmHWM:", "VmRSS:", "Rss:").
 * @param out_kb Receives the parsed KiB value on success.
 * @return 1 if parsed successfully, 0 otherwise.
 */
static int parse_status_kb_line(const char *line, const char *key, uint64_t *out_kb) {
  if (!line || !key || !out_kb) return 0;

  const size_t klen = strlen(key);
  if (strncmp(line, key, klen) != 0) return 0;

  const char *p = line + klen;
  while (*p == ' ' || *p == '\t') ++p;

  uint64_t val = 0;
  int matched = 0;
  while (*p >= '0' && *p <= '9') {
    matched = 1;
    val = val * 10 + (uint64_t)(*p - '0');
    ++p;
  }
  if (!matched) return 0;

  *out_kb = val;
  return 1;
}

/**
 * @brief Read a single numeric (KiB) field from a procfs text file.
 *
 * @param path File path (e.g., "/proc/self/status").
 * @param key  Key to match, e.g., "VmHWM:", "VmRSS:", or "Rss:".
 * @param out_kb Receives parsed value (KiB) on success.
 * @return 0 on success, -ENOENT if not found, negative errno otherwise.
 */
static int read_proc_kb_single(const char *path, const char *key, uint64_t *out_kb) {
#if defined(__linux__)
  FILE *f = fopen(path, "r");
  if (!f) return (errno ? -errno : -1);

  char buf[512];
  while (fgets(buf, sizeof buf, f)) {
    uint64_t kb = 0;
    if (parse_status_kb_line(buf, key, &kb)) {
      fclose(f);
      *out_kb = kb;
      return 0;
    }
  }

  fclose(f);
  return -ENOENT;
#else
  (void)path;
  (void)key;
  (void)out_kb;
  return -ENOSYS;
#endif
}

/* -------------------------------------------------------------------------- */
/* Public API: KV counters                                                     */
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

void ie_metrics_snapshot(ie_metrics_t *out, int reset_after) {
  if (!out) return;

  const uint64_t hits = atomic_load_explicit(&g_kv_hits, memory_order_relaxed);
  const uint64_t miss = atomic_load_explicit(&g_kv_miss, memory_order_relaxed);

  out->kv_hits   = hits;
  out->kv_misses = miss;

  if (reset_after) {
    atomic_store_explicit(&g_kv_hits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_kv_miss, 0, memory_order_relaxed);
  }
}

/* -------------------------------------------------------------------------- */
/* Public API: RSS peak sampler                                                */
/* -------------------------------------------------------------------------- */

uint32_t ie_metrics_sample_rss_peak(void) {
  uint32_t best_mib = 0;

#if defined(__linux__)
  /* 1) Prefer VmHWM (KiB): process peak resident set size. */
  {
    uint64_t kb = 0;
    const int rc = read_proc_kb_single("/proc/self/status", "VmHWM:", &kb);
    if (rc == 0) {
      const uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] VmHWM: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] VmHWM not found (rc=%d)\n", rc);
    }
  }

  /* 2) Fallback: VmRSS (KiB): current resident set size. */
  if (best_mib == 0) {
    uint64_t kb = 0;
    const int rc = read_proc_kb_single("/proc/self/status", "VmRSS:", &kb);
    if (rc == 0) {
      const uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] VmRSS: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] VmRSS not found (rc=%d)\n", rc);
    }
  }

  /* 3) Try smaps_rollup Rss: (KiB): often useful with mmaps. */
  if (best_mib == 0) {
    uint64_t kb = 0;
    const int rc = read_proc_kb_single("/proc/self/smaps_rollup", "Rss:", &kb);
    if (rc == 0) {
      const uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] smaps_rollup Rss: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] smaps_rollup Rss not found (rc=%d)\n", rc);
    }
  }

  /* 4) Final fallback: getrusage (Linux ru_maxrss is KiB). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t kb = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] getrusage ru_maxrss: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] getrusage failed: errno=%d\n", errno);
    }
  }

#elif defined(__APPLE__)
  /* macOS primary: mach_task_basic_info (bytes). */
  {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t kr = task_info(mach_task_self(),
                                       MACH_TASK_BASIC_INFO,
                                       (task_info_t)&info,
                                       &count);
    if (kr == KERN_SUCCESS) {
      uint64_t bytes = (uint64_t)info.resident_size_max;
      if (bytes == 0) bytes = (uint64_t)info.resident_size;

      const uint32_t mib = (uint32_t)((bytes + (1024u * 1024u - 1)) / (1024u * 1024u));
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] mach_task_basic_info: %llu bytes -> %u MiB\n",
          (unsigned long long)bytes, mib);
    } else {
      dbg("[RSS] mach task_info failed: kr=%d\n", (int)kr);
    }
  }

  /* Fallback: getrusage (macOS ru_maxrss is bytes). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t bytes = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = (uint32_t)((bytes + (1024u * 1024u - 1)) / (1024u * 1024u));
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] getrusage ru_maxrss(bytes): %llu -> %u MiB\n",
          (unsigned long long)bytes, mib);
    }
  }

#else
  /* Other OS: getrusage and conservatively treat ru_maxrss as KiB. */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t kb = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] generic getrusage ru_maxrss: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    }
  }
#endif

  return best_mib;
}
