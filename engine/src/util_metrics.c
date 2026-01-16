/**
 * @file util_metrics.c
 * @brief Runtime metrics: KV counters and best-effort RSS peak sampling.
 *
 * @details
 * This module implements the helpers declared in util_metrics.h:
 *  - Process-wide KV hit/miss accumulators (C11 atomics, relaxed).
 *  - A best-effort sampler for peak RSS in MiB, using platform-specific sources.
 *
 * RSS strategy (best effort):
 *  - Linux: prefer /proc/self/status:VmHWM, then VmRSS, then /proc/self/smaps_rollup:Rss,
 *           then getrusage(RUSAGE_SELF).ru_maxrss.
 *  - macOS: task_info(MACH_TASK_BASIC_INFO) resident_size_max/resident_size, then getrusage.
 *  - Other: getrusage and conservatively interpret ru_maxrss as KiB.
 *
 * Debug:
 *  - Set IE_DEBUG_RSS=1 to log which source was used and the raw units before conversion.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "util_metrics.h"

#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
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
/* Globals                                                                     */
/* -------------------------------------------------------------------------- */

/** @brief Process-wide KV hit counter. */
static _Atomic uint64_t g_kv_hits = 0;

/** @brief Process-wide KV miss counter. */
static _Atomic uint64_t g_kv_miss = 0;

/* -------------------------------------------------------------------------- */
/* Local helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return 1 if environment variable @p name is set to a truthy flag.
 *
 * Truthy values (case-insensitive): "1", "true", "yes", "on".
 *
 * @param name Environment variable name.
 * @return 1 if truthy, 0 otherwise.
 */
static int ie_env_flag(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0;
  return (!strcasecmp(s, "1") ||
          !strcasecmp(s, "true") ||
          !strcasecmp(s, "yes") ||
          !strcasecmp(s, "on"));
}

/**
 * @brief Debug logging helper controlled by IE_DEBUG_RSS=1.
 *
 * @param fmt printf-style format string.
 */
static void ie_dbg_rss(const char *fmt, ...) {
  if (!ie_env_flag("IE_DEBUG_RSS")) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/**
 * @brief Convert KiB to MiB (ceil), clamped to uint32_t.
 *
 * @param kib Size in KiB.
 * @return Size in MiB (ceil), or 0 if @p kib is 0.
 */
static uint32_t ie_kib_to_mib_ceil(uint64_t kib) {
  if (kib == 0) return 0u;
  /* ceil(kib/1024) = (kib + 1023) >> 10 */
  const uint64_t mib = (kib + 1023u) >> 10;
  return (mib > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)mib;
}

#if defined(__APPLE__)
/**
 * @brief Convert bytes to MiB (ceil), clamped to uint32_t.
 *
 * @param bytes Size in bytes.
 * @return Size in MiB (ceil), or 0 if @p bytes is 0.
 */
static uint32_t ie_bytes_to_mib_ceil(uint64_t bytes) {
  if (bytes == 0) return 0u;
  const uint64_t denom = 1024ull * 1024ull;
  const uint64_t mib = (bytes + (denom - 1ull)) / denom;
  return (mib > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)mib;
}
#endif

/**
 * @brief Parse an unsigned integer value from a procfs-style line.
 *
 * Expects lines like:
 *   "VmHWM:      12345 kB"
 *   "Rss:         6789 kB"
 *
 * The parser:
 *  - Matches @p key exactly at the beginning of @p line.
 *  - Skips spaces/tabs after the key.
 *  - Parses a base-10 unsigned integer.
 *
 * @param line Input line (NUL-terminated).
 * @param key  Key prefix to match (e.g., "VmHWM:", "VmRSS:", "Rss:").
 * @param out_val Receives parsed value on success.
 * @return 1 if parsed successfully, 0 otherwise.
 */
static int ie_parse_status_u64_line(const char *line, const char *key, uint64_t *out_val) {
  if (!line || !key || !out_val) return 0;

  const size_t klen = strlen(key);
  if (strncmp(line, key, klen) != 0) return 0;

  const char *p = line + klen;
  while (*p == ' ' || *p == '\t') ++p;

  uint64_t val = 0;
  int matched = 0;
  while (*p >= '0' && *p <= '9') {
    matched = 1;
    val = (val * 10u) + (uint64_t)(*p - '0');
    ++p;
  }
  if (!matched) return 0;

  *out_val = val;
  return 1;
}

/**
 * @brief Read a single numeric field from a procfs text file.
 *
 * This scans the file line-by-line and matches @p key using
 * ie_parse_status_u64_line(). The parsed numeric value is returned in
 * @p out_val.
 *
 * @param path Procfs file path.
 * @param key  Key prefix to match.
 * @param out_val Receives parsed value on success.
 * @return 0 on success, -ENOENT if key not found, negative errno otherwise.
 */
static int ie_read_proc_u64_single(const char *path, const char *key, uint64_t *out_val) {
#if defined(__linux__)
  FILE *f = fopen(path, "r");
  if (!f) return (errno ? -errno : -1);

  char buf[512];
  while (fgets(buf, (int)sizeof buf, f)) {
    uint64_t v = 0;
    if (ie_parse_status_u64_line(buf, key, &v)) {
      fclose(f);
      *out_val = v;
      return 0;
    }
  }

  fclose(f);
  return -ENOENT;
#else
  (void)path;
  (void)key;
  (void)out_val;
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
  /* 1) Prefer VmHWM (KiB): process peak RSS. */
  {
    uint64_t kb = 0;
    const int rc = ie_read_proc_u64_single("/proc/self/status", "VmHWM:", &kb);
    if (rc == 0) {
      const uint32_t mib = ie_kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      ie_dbg_rss("[RSS] VmHWM: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      ie_dbg_rss("[RSS] VmHWM not found (rc=%d)\n", rc);
    }
  }

  /* 2) Fallback: VmRSS (KiB): current RSS. */
  if (best_mib == 0) {
    uint64_t kb = 0;
    const int rc = ie_read_proc_u64_single("/proc/self/status", "VmRSS:", &kb);
    if (rc == 0) {
      const uint32_t mib = ie_kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      ie_dbg_rss("[RSS] VmRSS: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      ie_dbg_rss("[RSS] VmRSS not found (rc=%d)\n", rc);
    }
  }

  /* 3) Try smaps_rollup Rss: (KiB). */
  if (best_mib == 0) {
    uint64_t kb = 0;
    const int rc = ie_read_proc_u64_single("/proc/self/smaps_rollup", "Rss:", &kb);
    if (rc == 0) {
      const uint32_t mib = ie_kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      ie_dbg_rss("[RSS] smaps_rollup Rss: %llu kB -> %u MiB\n", (unsigned long long)kb, mib);
    } else {
      ie_dbg_rss("[RSS] smaps_rollup Rss not found (rc=%d)\n", rc);
    }
  }

  /* 4) Final fallback: getrusage (Linux ru_maxrss is KiB). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t kb = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = ie_kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      ie_dbg_rss("[RSS] getrusage ru_maxrss: %llu kB -> %u MiB\n",
                 (unsigned long long)kb, mib);
    } else {
      ie_dbg_rss("[RSS] getrusage failed: errno=%d\n", errno);
    }
  }

#elif defined(__APPLE__)
  /* macOS primary: mach_task_basic_info (bytes). */
  {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t kr =
      task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);

    if (kr == KERN_SUCCESS) {
      uint64_t bytes = (uint64_t)info.resident_size_max;
      if (bytes == 0) bytes = (uint64_t)info.resident_size;

      const uint32_t mib = ie_bytes_to_mib_ceil(bytes);
      if (mib > best_mib) best_mib = mib;

      ie_dbg_rss("[RSS] mach_task_basic_info: %llu bytes -> %u MiB\n",
                 (unsigned long long)bytes, mib);
    } else {
      ie_dbg_rss("[RSS] mach task_info failed: kr=%d\n", (int)kr);
    }
  }

  /* Fallback: getrusage (macOS ru_maxrss is bytes). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t bytes = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = ie_bytes_to_mib_ceil(bytes);
      if (mib > best_mib) best_mib = mib;

      ie_dbg_rss("[RSS] getrusage ru_maxrss(bytes): %llu -> %u MiB\n",
                 (unsigned long long)bytes, mib);
    }
  }

#else
  /* Other OS: getrusage; interpret ru_maxrss as KiB (best-effort). */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      const uint64_t kb = (uint64_t)ru.ru_maxrss;
      const uint32_t mib = ie_kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;

      ie_dbg_rss("[RSS] generic getrusage ru_maxrss: %llu kB -> %u MiB\n",
                 (unsigned long long)kb, mib);
    }
  }
#endif

  return best_mib;
}
