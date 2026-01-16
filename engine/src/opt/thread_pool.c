/**
 * @file thread_pool.c
 * @brief Minimal thread pool with synchronous parallel-for and optional Linux CPU affinity.
 *
 * Design notes:
 *  - The pool owns long-lived worker threads created at `ie_tp_create()`.
 *  - `ie_tp_parallel_for()` distributes chunks to the worker set and waits for completion.
 *  - The calling thread also participates to reduce scheduling overhead.
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
  unsigned nth_created;   /**< Number of workers successfully created. */
  const char *affinity;   /**< Affinity policy string (advisory). */
  int use_affinity;       /**< Non-zero when `IE_TP_USE_AFFINITY=1` on Linux. */
  pthread_t *threads;     /**< Array of worker thread handles. */

  pthread_mutex_t mu;
  pthread_cond_t cv;
  pthread_cond_t done;
  int shutdown;

  unsigned epoch;
  unsigned active;
  unsigned next;
  unsigned n;
  unsigned grainsize;
  ie_tp_for_body_fn body;
  void *ctx;
};

typedef struct {
  struct ie_threadpool *tp;
  unsigned tidx;
} ie_worker_arg_t;

/* -------------------------------------------------------------------------- */
/* Affinity helper                                                            */
/* -------------------------------------------------------------------------- */

static void ie_set_affinity(const char *policy, unsigned tidx, unsigned nth, int use_affinity) {
#ifdef __linux__
  if (!use_affinity || !policy) return;

  cpu_set_t set;
  CPU_ZERO(&set);

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu <= 0) return;

  unsigned cpu = tidx % (unsigned)ncpu; /* default "auto" */

  if (strcmp(policy, "compact") == 0) {
    cpu = tidx % (unsigned)ncpu;
  } else if (strcmp(policy, "scatter") == 0) {
    unsigned stride = (unsigned)(ncpu / (nth ? nth : 1)) + 1u;
    cpu = (unsigned)((tidx * stride) % (unsigned)ncpu);
  }

  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)policy; (void)tidx; (void)nth; (void)use_affinity;
#endif
}

/* -------------------------------------------------------------------------- */
/* Worker logic                                                               */
/* -------------------------------------------------------------------------- */

static void ie_tp_worker_run(struct ie_threadpool *tp) {
  for (;;) {
    pthread_mutex_lock(&tp->mu);
    unsigned start = tp->next;
    if (start >= tp->n) {
      pthread_mutex_unlock(&tp->mu);
      break;
    }
    unsigned end = start + tp->grainsize;
    if (end > tp->n) end = tp->n;
    tp->next = end;

    ie_tp_for_body_fn body = tp->body;
    void *ctx = tp->ctx;
    pthread_mutex_unlock(&tp->mu);

    if (body) body(ctx, start, end);
  }
}

static void* ie_worker_main(void *arg) {
  ie_worker_arg_t *wa = (ie_worker_arg_t*)arg;
  struct ie_threadpool *tp = wa->tp;
  unsigned tidx = wa->tidx;
  free(wa);

  ie_set_affinity(tp->affinity, tidx, tp->nth_created ? tp->nth_created : tp->nth, tp->use_affinity);

  unsigned epoch_seen = 0;
  for (;;) {
    pthread_mutex_lock(&tp->mu);
    while (!tp->shutdown && epoch_seen == tp->epoch) {
      pthread_cond_wait(&tp->cv, &tp->mu);
    }
    if (tp->shutdown) {
      pthread_mutex_unlock(&tp->mu);
      return NULL;
    }
    epoch_seen = tp->epoch;
    pthread_mutex_unlock(&tp->mu);

    ie_tp_worker_run(tp);

    pthread_mutex_lock(&tp->mu);
    if (tp->active > 0) {
      tp->active--;
      if (tp->active == 0) pthread_cond_signal(&tp->done);
    }
    pthread_mutex_unlock(&tp->mu);
  }
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

  tp->threads = (pthread_t*)calloc(nth, sizeof(pthread_t));
  if (!tp->threads) { free(tp); return NULL; }

  pthread_mutex_init(&tp->mu, NULL);
  pthread_cond_init(&tp->cv, NULL);
  pthread_cond_init(&tp->done, NULL);

  for (unsigned i = 0; i < nth; ++i) {
    ie_worker_arg_t *wa = (ie_worker_arg_t*)malloc(sizeof(*wa));
    if (!wa) break;
    wa->tp = tp;
    wa->tidx = i;
    if (pthread_create(&tp->threads[tp->nth_created], NULL, ie_worker_main, wa) != 0) {
      free(wa);
      break;
    }
    tp->nth_created++;
  }

  return (ie_threadpool_t*)tp;
}

void ie_tp_destroy(ie_threadpool_t *h) {
  if (!h) return;
  struct ie_threadpool *tp = (struct ie_threadpool*)h;

  pthread_mutex_lock(&tp->mu);
  tp->shutdown = 1;
  pthread_cond_broadcast(&tp->cv);
  pthread_mutex_unlock(&tp->mu);

  for (unsigned i = 0; i < tp->nth_created; ++i) {
    pthread_join(tp->threads[i], NULL);
  }

  pthread_mutex_destroy(&tp->mu);
  pthread_cond_destroy(&tp->cv);
  pthread_cond_destroy(&tp->done);
  free(tp->threads);
  free(tp);
}

void ie_tp_parallel_for(ie_threadpool_t *h,
                        unsigned n,
                        unsigned grainsize,
                        ie_tp_for_body_fn body_fn,
                        void *ctx) {
  if (n == 0 || !body_fn) return;
  if (!h) {
    body_fn(ctx, 0u, n);
    return;
  }

  struct ie_threadpool *tp = (struct ie_threadpool*)h;
  if (tp->nth_created <= 1) {
    body_fn(ctx, 0u, n);
    return;
  }

  unsigned min_chunk = (grainsize > 0 ? grainsize
                                      : (n + tp->nth_created - 1u) / tp->nth_created);
  if (min_chunk == 0) min_chunk = 1;

  pthread_mutex_lock(&tp->mu);
  tp->n = n;
  tp->grainsize = min_chunk;
  tp->body = body_fn;
  tp->ctx = ctx;
  tp->next = 0;
  tp->active = tp->nth_created + 1u; /* include the calling thread */
  tp->epoch++;
  pthread_cond_broadcast(&tp->cv);
  pthread_mutex_unlock(&tp->mu);

  ie_tp_worker_run(tp);

  pthread_mutex_lock(&tp->mu);
  if (tp->active > 0) {
    tp->active--;
    if (tp->active == 0) pthread_cond_signal(&tp->done);
  }
  while (tp->active != 0) {
    pthread_cond_wait(&tp->done, &tp->mu);
  }
  pthread_mutex_unlock(&tp->mu);
}
