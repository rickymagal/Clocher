/**
 * @file thread_pool.c
 * @brief Minimal thread pool with synchronous parallel-for and optional Linux CPU affinity.
 *
 * Design notes:
 *  - The pool owns long-lived worker threads created at `ie_tp_create()`.
 *  - `ie_tp_parallel_for()` distributes chunks to the worker set and waits for completion.
 *  - The calling thread also participates to reduce scheduling overhead.
 *
 * Thread-safety:
 *  - This pool is designed for one active `ie_tp_parallel_for()` call at a time.
 *  - `ie_tp_parallel_for()` is not re-entrant and must not be called concurrently
 *    on the same pool (including from within `body_fn`).
 */

#define _GNU_SOURCE /* for pthread_setaffinity_np on glibc */

#include "ie_threadpool.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Allocate and copy a C string.
 *
 * @param s Input string (may be NULL).
 * @return Newly allocated copy, or NULL if allocation fails.
 */
static char *ie_strdup_local(const char *s) {
  if (!s) return NULL;
  size_t n = strlen(s);
  char *p = (char *)malloc(n + 1u);
  if (!p) return NULL;
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

/* -------------------------------------------------------------------------- */
/* Internal data types                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @brief Thread pool definition (opaque to users).
 */
struct ie_threadpool {
  unsigned nth;           /**< Requested number of workers. */
  unsigned nth_created;   /**< Number of workers successfully created. */
  char *affinity;         /**< Copied affinity policy string ("auto"/"compact"/"scatter"). */
  int use_affinity;       /**< Non-zero when `IE_TP_USE_AFFINITY=1` on Linux. */
  pthread_t *threads;     /**< Array of worker thread handles. */

  pthread_mutex_t mu;     /**< Protects all fields below. */
  pthread_cond_t cv;      /**< Signals workers to start an epoch of work. */
  pthread_cond_t done;    /**< Signals the caller when all participants finished. */
  int shutdown;           /**< Set to non-zero to request worker termination. */

  unsigned epoch;         /**< Monotonic counter identifying a unit of work. */
  unsigned active;        /**< Number of participants still running this epoch. */
  unsigned next;          /**< Next start index to claim from [0, n). */

  unsigned n;             /**< Total iterations for current epoch. */
  unsigned grainsize;     /**< Chunk size for current epoch. */
  ie_tp_for_body_fn body; /**< Work callback for current epoch. */
  void *ctx;              /**< Opaque context for current epoch. */
};

/**
 * @brief Worker thread bootstrap arguments.
 */
typedef struct {
  struct ie_threadpool *tp;
  unsigned tidx;
} ie_worker_arg_t;

/* -------------------------------------------------------------------------- */
/* Affinity helper                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Apply a best-effort CPU affinity policy on Linux.
 *
 * Policies:
 *  - "auto":   tidx modulo CPU count.
 *  - "compact": same as auto (kept for clarity/compatibility).
 *  - "scatter": spread threads by a simple stride heuristic.
 *
 * @param policy       Policy string (non-NULL).
 * @param tidx         Worker index [0, nth).
 * @param nth          Worker count for stride heuristics.
 * @param use_affinity Non-zero to enable affinity (Linux only).
 */
static void ie_set_affinity(const char *policy, unsigned tidx, unsigned nth, int use_affinity) {
#ifdef __linux__
  if (!use_affinity || !policy) return;

  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (ncpu <= 0) return;

  cpu_set_t set;
  CPU_ZERO(&set);

  unsigned u_ncpu = (unsigned)ncpu;
  unsigned cpu = tidx % u_ncpu; /* default "auto" */

  if (strcmp(policy, "compact") == 0) {
    cpu = tidx % u_ncpu;
  } else if (strcmp(policy, "scatter") == 0) {
    unsigned denom = (nth ? nth : 1u);
    unsigned stride = (u_ncpu / denom) + 1u;
    cpu = (unsigned)((tidx * stride) % u_ncpu);
  }

  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)policy;
  (void)tidx;
  (void)nth;
  (void)use_affinity;
#endif
}

/* -------------------------------------------------------------------------- */
/* Worker logic                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Claim and execute chunks from the current epoch until exhausted.
 *
 * The work scheduling is protected by `tp->mu`. Each participant repeatedly
 * claims a contiguous chunk and executes `tp->body(ctx, start, end)` outside
 * the mutex to minimize contention.
 *
 * @param tp Thread pool instance.
 */
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

/**
 * @brief Worker thread entry point.
 *
 * Each worker waits for the pool epoch to change, executes available work for
 * that epoch, then decrements `tp->active` and signals completion when it hits 0.
 *
 * @param arg Pointer to a dynamically allocated `ie_worker_arg_t`.
 * @return NULL.
 */
static void* ie_worker_main(void *arg) {
  ie_worker_arg_t *wa = (ie_worker_arg_t*)arg;
  struct ie_threadpool *tp = wa->tp;
  unsigned tidx = wa->tidx;
  free(wa);

  ie_set_affinity(tp->affinity ? tp->affinity : "auto",
                  tidx,
                  tp->nth_created ? tp->nth_created : tp->nth,
                  tp->use_affinity);

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

  const char *aff = (affinity && *affinity) ? affinity : "auto";
  tp->affinity = ie_strdup_local(aff);
  if (!tp->affinity) {
    free(tp);
    return NULL;
  }

#ifdef __linux__
  const char *env = getenv("IE_TP_USE_AFFINITY");
  tp->use_affinity = (env && strcmp(env, "1") == 0) ? 1 : 0;
#else
  tp->use_affinity = 0;
#endif

  tp->threads = (pthread_t*)calloc(nth, sizeof(pthread_t));
  if (!tp->threads) {
    free(tp->affinity);
    free(tp);
    return NULL;
  }

  (void)pthread_mutex_init(&tp->mu, NULL);
  (void)pthread_cond_init(&tp->cv, NULL);
  (void)pthread_cond_init(&tp->done, NULL);

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
    (void)pthread_join(tp->threads[i], NULL);
  }

  (void)pthread_mutex_destroy(&tp->mu);
  (void)pthread_cond_destroy(&tp->cv);
  (void)pthread_cond_destroy(&tp->done);

  free(tp->threads);
  free(tp->affinity);
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

  unsigned min_chunk = (grainsize > 0)
                         ? grainsize
                         : (n + tp->nth_created - 1u) / tp->nth_created;
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
