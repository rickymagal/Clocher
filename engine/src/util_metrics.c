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

#include "util_metrics.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdarg.h>

#include "util_metrics.h"

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <strings.h>  /* strcasecmp */


/* Platform headers for RSS sampling */
#if defined(__linux__)
  #include <sys/resource.h>
  #include <unistd.h>
#elif defined(__APPLE__)
  #include <sys/resource.h>
  #include <mach/mach.h>
#else
  #include <sys/resource.h>
#endif

/* -------------------------------------------------------------------------- */
/* Globals: KV accumulators (process-wide)                                     */
/* -------------------------------------------------------------------------- */

static _Atomic uint64_t g_kv_hits = 0;
static _Atomic uint64_t g_kv_miss = 0;

/* -------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return 1 if the env var @p name is a truthy flag (1,true,yes,on).
 */
static int env_flag(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0;
  return (!strcasecmp(s, "1") || !strcasecmp(s, "true") ||
          !strcasecmp(s, "yes") || !strcasecmp(s, "on"));
}

/**
 * @brief Debug print controlled by `IE_DEBUG_RSS=1`.
 */
static void dbg(const char *fmt, ...) {
  if (!env_flag("IE_DEBUG_RSS")) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

/**
 * @brief Round up kB to MiB (ceil), clamped to 32-bit.
 */
static uint32_t kib_to_mib_ceil(uint64_t kib) {
  if (kib == 0) return 0u;
  /* ceil(kib/1024) = (kib + 1023) >> 10 */
  uint64_t mib = (kib + 1023u) >> 10;
  return (mib > 0xFFFFFFFFu) ? 0xFFFFFFFFu : (uint32_t)mib;
}

/**
 * @brief Parse integer kB from a "Key:   12345 kB" style line.
 */
static int parse_status_kb_line(const char* line, const char* key, uint64_t* out_kb) {
  if (!line || !key || !out_kb) return 0;
  size_t klen = strlen(key);
  if (strncmp(line, key, klen) != 0) return 0;

  const char* p = line + klen;
  while (*p == ' ' || *p == '\t') ++p;

  uint64_t val = 0;
  int matched = 0;
  while (*p >= '0' && *p <= '9') {
    matched = 1;
    val = val * 10 + (uint64_t)(*p - '0');
    ++p;
  }
  if (!matched) return 0;

  *out_kb = val; /* procfs reports kB */
  return 1;
}

/**
 * @brief Grep one numeric (kB) key from a procfs text file.
 *
 * @param path File path (e.g., "/proc/self/status").
 * @param key  Key to match, e.g., "VmHWM:" or "VmRSS:" or "Rss:".
 * @param out_kb Receives parsed value (kB) on success.
 * @return 0 on success, -ENOENT if not found, negative errno otherwise.
 */
static int read_proc_kb_single(const char *path, const char *key, uint64_t *out_kb) {
#if defined(__linux__)
  FILE *f = fopen(path, "r");
  if (!f) return -errno ? -errno : -1;
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
  (void)path; (void)key; (void)out_kb;
  return -ENOSYS;
#endif
}

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

  if (reset_after) {
    atomic_store_explicit(&g_kv_hits, 0, memory_order_relaxed);
    atomic_store_explicit(&g_kv_miss, 0, memory_order_relaxed);
  }
}

/* -------------------------------------------------------------------------- */
/* RSS peak sampler                                                            */
/* -------------------------------------------------------------------------- */

uint32_t ie_metrics_sample_rss_peak(void) {
  uint32_t best_mib = 0;

#if defined(__linux__)
  /* 1) Prefer VmHWM (kB) — process peak resident set size. */
  {
    uint64_t kb = 0;
    int rc = read_proc_kb_single("/proc/self/status", "VmHWM:", &kb);
    if (rc == 0) {
      uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] VmHWM: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] VmHWM not found (rc=%d)\n", rc);
    }
  }

  /* 2) Fallback: current VmRSS (kB) as lower bound if HWM is unavailable. */
  if (best_mib == 0) {
    uint64_t kb = 0;
    int rc = read_proc_kb_single("/proc/self/status", "VmRSS:", &kb);
    if (rc == 0) {
      uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] VmRSS: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] VmRSS not found (rc=%d)\n", rc);
    }
  }

  /* 3) Try smaps_rollup Rss: (kB) — often more accurate with mmaps. */
  if (best_mib == 0) {
    uint64_t kb = 0;
    int rc = read_proc_kb_single("/proc/self/smaps_rollup", "Rss:", &kb);
    if (rc == 0) {
      uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] smaps_rollup Rss: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] smaps_rollup Rss not found (rc=%d)\n", rc);
    }
  }

  /* 4) Final fallback: getrusage (Linux ru_maxrss is kB). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t kb = (uint64_t)ru.ru_maxrss;
      uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] getrusage ru_maxrss: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    } else {
      dbg("[RSS] getrusage failed: errno=%d\n", errno);
    }
  }

#elif defined(__APPLE__)
  /* macOS primary: mach_task_basic_info (bytes). */
  {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    if (kr == KERN_SUCCESS) {
      /* resident_size_max is peak if supported; otherwise 0. */
      uint64_t bytes = (uint64_t)info.resident_size_max;
      if (bytes == 0) bytes = (uint64_t)info.resident_size; /* at least current */
      uint32_t mib = (uint32_t)((bytes + (1024u*1024u - 1)) / (1024u*1024u));
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] mach_task_basic_info: %llu bytes -> %u MiB\n",
          (unsigned long long)bytes, mib);
    } else {
      dbg("[RSS] mach task_info failed: kr=%d\n", (int)kr);
    }
  }

  /* Fallback: getrusage (on macOS ru_maxrss is bytes). */
  if (best_mib == 0) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t bytes = (uint64_t)ru.ru_maxrss;
      uint32_t mib = (uint32_t)((bytes + (1024u*1024u - 1)) / (1024u*1024u));
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] getrusage ru_maxrss(bytes): %llu -> %u MiB\n",
          (unsigned long long)bytes, mib);
    }
  }
#else
  /* Other OS: try getrusage and assume kB (conservative). */
  {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
      uint64_t kb = (uint64_t)ru.ru_maxrss;
      uint32_t mib = kib_to_mib_ceil(kb);
      if (mib > best_mib) best_mib = mib;
      dbg("[RSS] generic getrusage ru_maxrss: %llu kB -> %u MiB\n",
          (unsigned long long)kb, mib);
    }
  }
#endif

  return best_mib;
}
