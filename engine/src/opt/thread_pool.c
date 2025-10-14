/**
 * @file thread_pool.c
 * @brief Minimal thread pool with synchronous parallel-for and optional Linux CPU affinity.
 *
 * Design notes:
 *  - The pool owns a set of long-lived worker threads created at `ie_tp_create()`.
 *    (They are idle placeholders in this baseline.)
 *  - `ie_tp_parallel_for()` launches one short-lived task thread **per chunk**,
 *    so the entire [0, n) range is always covered even when `nchunks > nth`.
 *  - Affinity (Linux only) is applied using a modulo mapping so chunks greater
 *    than the worker count still get a stable CPU choice.
 */

#define _GNU_SOURCE  /* for pthread_setaffinity_np on glibc */

#include "ie_threadpool.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

/* -------------------------------------------------------------------------- */
/* Internal data types                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Thread pool definition (opaque to users).
 */
struct ie_threadpool {
  unsigned nth;           /**< Number of workers requested at creation. */
  const char *affinity;   /**< Affinity policy string (advisory). */
  int use_affinity;       /**< Non-zero when `IE_TP_USE_AFFINITY=1` on Linux. */

  /* Long-lived worker threads (idle placeholders in this minimal impl). */
  pthread_t *threads;
};

/* -------------------------------------------------------------------------- */
/* Affinity helper                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Optionally set the current thread's CPU affinity (Linux only).
 *
 * If `use_affinity` is 0 or not Linux, this function is a no-op.
 *
 * @param policy       "auto", "compact", or "scatter".
 * @param tidx         Worker index [0, nth).
 * @param nth          Total number of workers.
 * @param use_affinity Non-zero to actually apply affinity (Linux only).
 */
static void ie_set_affinity(const char *policy, unsigned tidx, unsigned nth, int use_affinity) {
#ifdef __linux__
  if (!use_affinity || !policy) return;

  cpu_set_t set;
  CPU_ZERO(&set);

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu <= 0) return;

  unsigned cpu = tidx % (unsigned)ncpu; /* default "auto" */

  if (strcmp(policy, "compact") == 0) {
    cpu = tidx % (unsigned)ncpu; /* fill CPUs from 0.. */
  } else if (strcmp(policy, "scatter") == 0) {
    /* spread across the machine */
    unsigned stride = (unsigned)(ncpu / (nth ? nth : 1)) + 1u;
    cpu = (unsigned)((tidx * stride) % (unsigned)ncpu);
  } /* else "auto": keep default */

  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)policy; (void)tidx; (void)nth; (void)use_affinity;
#endif
}

/* -------------------------------------------------------------------------- */
/* Worker trampoline (long-lived, idle)                                       */
/* -------------------------------------------------------------------------- */

typedef struct {
  struct ie_threadpool *tp;
  unsigned tidx;
} ie_worker_arg_t;

/**
 * @brief Long-lived worker entry. Currently idle placeholder.
 *
 * @param arg Pointer to `ie_worker_arg_t`.
 * @return NULL.
 */
static void* ie_worker_main(void *arg) {
  ie_worker_arg_t *wa = (ie_worker_arg_t*)arg;
  struct ie_threadpool *tp = wa->tp;
  unsigned tidx = wa->tidx;

  ie_set_affinity(tp->affinity, tidx, tp->nth, tp->use_affinity);

  /* Minimal design: no persistent work queue in this baseline. */
  free(wa);
  return NULL;
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

ie_threadpool_t* ie_tp_create(unsigned nth, const char *affinity) {
  if (nth < 1) nth = 1;

  struct ie_threadpool *tp = (struct ie_threadpool*)calloc(1, sizeof(*tp));
  if (!tp) return NULL;

  tp->nth = nth;
  tp->affinity = (affinity && *affinity) ? affinity : "auto";

#ifdef __linux__
  const char *env = getenv("IE_TP_USE_AFFINITY");
  tp->use_affinity = (env && strcmp(env, "1") == 0) ? 1 : 0;
#else
  tp->use_affinity = 0;
#endif

  /* Spawn long-lived idle workers (not strictly required for the baseline). */
  tp->threads = (pthread_t*)calloc(nth, sizeof(pthread_t));
  if (!tp->threads) { free(tp); return NULL; }

  for (unsigned i = 0; i < nth; ++i) {
    ie_worker_arg_t *wa = (ie_worker_arg_t*)malloc(sizeof(*wa));
    if (!wa) continue;
    wa->tp = tp; wa->tidx = i;
    if (pthread_create(&tp->threads[i], NULL, ie_worker_main, wa) != 0) {
      free(wa);
    }
  }
  return (ie_threadpool_t*)tp;
}

void ie_tp_destroy(ie_threadpool_t *h) {
  if (!h) return;
  struct ie_threadpool *tp = (struct ie_threadpool*)h;

  if (tp->threads) {
    for (unsigned i = 0; i < tp->nth; ++i) {
      if (tp->threads[i]) pthread_join(tp->threads[i], NULL);
    }
    free(tp->threads);
  }
  free(tp);
}

/* -------------------------------------------------------------------------- */
/* Synchronous parallel-for                                                   */
/* -------------------------------------------------------------------------- */

typedef struct {
  const char *affinity;
  int use_affinity;
  unsigned tidx_map;  /* mapped thread index for affinity */
  unsigned nth_map;   /* mapped pool size for affinity */

  unsigned start;
  unsigned end;

  ie_tp_for_body_fn body;
  void *ctx;
} ie_task_arg_t;

/**
 * @brief Task thread trampoline that applies optional affinity and executes a chunk.
 *
 * @param arg Pointer to `ie_task_arg_t`.
 * @return NULL.
 */
static void* ie_task_main(void *arg) {
  ie_task_arg_t *ta = (ie_task_arg_t*)arg;

  ie_set_affinity(ta->affinity, ta->tidx_map, ta->nth_map, ta->use_affinity);

  if (ta->start < ta->end && ta->body) {
    ta->body(ta->ctx, ta->start, ta->end);
  }
  free(ta);
  return NULL;
}

void ie_tp_parallel_for(ie_threadpool_t *h,
                        unsigned n,
                        unsigned grainsize,
                        ie_tp_for_body_fn body_fn,
                        void *ctx) {
  if (n == 0 || !body_fn) return;

  /* Preferred worker count and affinity policy for mapping. */
  unsigned nth_pref = 1;
  const char *aff = "auto";
  int use_aff = 0;

  if (h) {
    struct ie_threadpool *tp = (struct ie_threadpool*)h;
    nth_pref = tp->nth ? tp->nth : 1u;
    aff = tp->affinity ? tp->affinity : "auto";
    use_aff = tp->use_affinity;
  }

  if (nth_pref <= 1) {
    body_fn(ctx, 0u, n);
    return;
  }

  /* Determine chunk size and total number of chunks. */
  unsigned min_chunk = (grainsize > 0 ? grainsize : (n + nth_pref - 1) / nth_pref);
  if (min_chunk == 0) min_chunk = 1;

  unsigned nchunks = (n + min_chunk - 1) / min_chunk;

  /* Launch one task thread per chunk to guarantee full coverage. */
  pthread_t *ths = (pthread_t*)calloc(nchunks, sizeof(pthread_t));
  if (!ths) {
    /* Fallback: single-threaded if allocation fails. */
    body_fn(ctx, 0u, n);
    return;
  }

  for (unsigned c = 0; c < nchunks; ++c) {
    unsigned start = c * min_chunk;
    unsigned end   = start + min_chunk;
    if (start >= n) { ths[c] = (pthread_t)0; continue; }
    if (end > n) end = n;

    ie_task_arg_t *ta = (ie_task_arg_t*)malloc(sizeof(*ta));
    if (!ta) { ths[c] = (pthread_t)0; continue; }

    /* Map chunk index to a virtual worker index for affinity purposes. */
    ta->affinity  = aff;
    ta->use_affinity = use_aff;
    ta->nth_map   = nth_pref;
    ta->tidx_map  = (nth_pref ? (c % nth_pref) : 0u);

    ta->start     = start;
    ta->end       = end;
    ta->body      = body_fn;
    ta->ctx       = ctx;

    if (pthread_create(&ths[c], NULL, ie_task_main, ta) != 0) {
      free(ta);
      ths[c] = (pthread_t)0;
    }
  }

  for (unsigned c = 0; c < nchunks; ++c) {
    if (ths[c]) pthread_join(ths[c], NULL);
  }
  free(ths);
}
