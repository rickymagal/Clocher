/**
 * @file util_metrics.c
 * @brief Implementation of small runtime metrics helpers (RSS, KV plumbing).
 *
 * This module provides:
 *  - `ie_metrics_read_proc_status_kb`: parse `VmHWM`/`VmRSS` from /proc.
 *  - `ie_metrics_sample_rss_peak`: update `rss_peak_mb` from `VmHWM`.
 *  - `ie_metrics_kv_reset` / `ie_metrics_kv_add`: trivial KV counters plumbing.
 *
 * All functions avoid heavy-weight operations. Critically, they do **not**
 * do any “measurement touches” over the model mapping; they are intended to
 * be called **outside** the timed region in the CLI.
 */

#define _POSIX_C_SOURCE 200809L

#include "util_metrics.h"

#include <stdio.h>      /* FILE, fopen, fgets */
#include <string.h>     /* strncmp, strstr */
#include <errno.h>      /* errno */
#include <stdlib.h>     /* strtoul */
#include <inttypes.h>   /* uint64_t */

/* -------------------------------------------------------------------------- */
/* Internal: portable gate                                                     */
/* -------------------------------------------------------------------------- */

#if defined(__linux__)
#  define IE_HAVE_PROC 1
#else
#  define IE_HAVE_PROC 0
#endif

/* -------------------------------------------------------------------------- */
/* /proc parser                                                               */
/* -------------------------------------------------------------------------- */

int ie_metrics_read_proc_status_kb(size_t *VmHWM_kb, size_t *VmRSS_kb) {
#if IE_HAVE_PROC
  FILE *f = fopen("/proc/self/status", "r");
  if (!f) return -1;

  size_t hwm = 0, rss = 0;
  char line[512];

  while (fgets(line, (int)sizeof(line), f)) {
    /* Lines look like: "VmHWM:\t   123456 kB" */
    if (!hwm && !strncmp(line, "VmHWM:", 6)) {
      const char *p = line + 6;
      while (*p == ' ' || *p == '\t') ++p;
      char *end = NULL;
      unsigned long v = strtoul(p, &end, 10);
      if (end && v > 0UL) hwm = (size_t)v;
    } else if (!rss && !strncmp(line, "VmRSS:", 6)) {
      const char *p = line + 6;
      while (*p == ' ' || *p == '\t') ++p;
      char *end = NULL;
      unsigned long v = strtoul(p, &end, 10);
      if (end && v > 0UL) rss = (size_t)v;
    }
    if (hwm && rss) break;
  }
  fclose(f);

  if (VmHWM_kb) *VmHWM_kb = hwm;
  if (VmRSS_kb) *VmRSS_kb = rss;
  return 0;
#else
  if (VmHWM_kb) *VmHWM_kb = 0;
  if (VmRSS_kb) *VmRSS_kb = 0;
  return 0;
#endif
}

int ie_metrics_sample_rss_peak(ie_metrics_t *m) {
  if (!m) return -1;
  size_t hwm_kb = 0;
  if (ie_metrics_read_proc_status_kb(&hwm_kb, NULL) != 0) {
    return -1;
  }
  /* Convert KiB -> MiB (round down). */
  size_t hwm_mb = hwm_kb / 1024u;
  if (hwm_mb > m->rss_peak_mb) {
    m->rss_peak_mb = hwm_mb;
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/* KV counters plumbing                                                       */
/* -------------------------------------------------------------------------- */

void ie_metrics_kv_reset(ie_metrics_t *m) {
  if (!m) return;
  m->kv_hits   = 0u;
  m->kv_misses = 0u;
}

void ie_metrics_kv_add(ie_metrics_t *m, uint64_t hits, uint64_t misses) {
  if (!m) return;
  m->kv_hits   += hits;
  m->kv_misses += misses;
}
