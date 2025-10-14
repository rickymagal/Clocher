/**
 * @file thread_pool.c
 * @brief Minimal pthread-based thread pool with blocking parallel_for.
 *
 * Key properties:
 *  - Deterministic contiguous partition of [0, n) across T workers.
 *  - Robust "task epoch" barrier ensures each worker executes a task
 *    **exactly once** per submission (no double-runs, no misses).
 *  - CPU affinity is disabled by default for portability; can be enabled
 *    by defining IE_TP_USE_AFFINITY=1 at compile time (Linux only).
 */

#if defined(__linux__)
  /* Expose GNU extensions for CPU_* macros and pthread_setaffinity_np. */
  #ifndef _GNU_SOURCE
  #define _GNU_SOURCE 1
  #endif
#endif

#include "ie_threadpool.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
  #include <sched.h>   /* CPU_ZERO, CPU_SET, cpu_set_t */
#endif

/* Toggle: 0 = no affinity calls (default, safer); 1 = try compact/scatter on Linux */
#ifndef IE_TP_USE_AFFINITY
#define IE_TP_USE_AFFINITY 0
#endif

/** @brief Command type for worker state machine. */
typedef enum { IE_CMD_NONE=0, IE_CMD_RUN=1, IE_CMD_STOP=2 } ie_cmd_t;

/** @brief Task payload describing the parallel range job. */
typedef struct {
  ie_range_task_fn fn;   /**< User callback. */
  void *ctx;             /**< User context. */
  size_t n;              /**< Total elements in [0,n). */
  size_t grainsize;      /**< Minimum chunk size (informational). */
} ie_task_t;

/**
 * @brief Per-thread bootstrap argument.
 */
typedef struct {
  struct ie_threadpool *tp; /**< Owning pool. */
  unsigned tid;             /**< Worker logical index [0..nthreads). */
} ie_worker_arg_t;

/** @brief Thread pool structure. */
struct ie_threadpool {
  unsigned nthreads;      /**< Number of worker threads. */
  pthread_t *thr;         /**< Thread handles. */
  ie_worker_arg_t *warg;  /**< Per-worker bootstrap args. */

  pthread_mutex_t mtx;    /**< Control mutex. */
  pthread_cond_t  cv;     /**< Command/epoch condvar. */
  pthread_cond_t  cv_done;/**< Completion condvar. */

  ie_cmd_t  cmd;          /**< Pending command (RUN/STOP/NONE). */
  ie_task_t task;         /**< Current task payload. */

  unsigned  arrived;      /**< Workers that completed this epoch. */
  unsigned  epoch;        /**< Current task epoch (monotonic). */

  char      affinity[16]; /**< Affinity mode ("auto"/"compact"/"scatter"). */
};

/**
 * @brief Get the number of online logical CPUs.
 * @return Number of logical CPUs (>= 1).
 */
static unsigned ie_nproc(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (unsigned)(n > 0 ? n : 1);
}

/**
 * @brief Best-effort worker CPU affinity binding (Linux only).
 *
 * @param tid        Worker index.
 * @param nthreads   Total workers.
 * @param mode       "compact"|"scatter"|"auto" (auto==compact).
 */
static void ie_set_affinity(unsigned tid, unsigned nthreads, const char *mode) {
#if IE_TP_USE_AFFINITY && defined(__linux__)
  if (!mode) mode = "compact";
  cpu_set_t set;
  CPU_ZERO(&set);

  unsigned ncpu = ie_nproc();
  unsigned cpu  = 0;

  if (strcmp(mode, "scatter") == 0) {
    unsigned step = (ncpu >= nthreads) ? (ncpu / nthreads) : 1u;
    if (step == 0) step = 1u;
    cpu = (tid * step) % ncpu;
  } else {
    cpu = tid % ncpu;
  }

  CPU_SET(cpu, &set);
  /* Ignore return code; failure is non-fatal. */
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)tid; (void)nthreads; (void)mode;
#endif
}

/**
 * @brief Execute the user task for the shard [begin, end).
 *
 * @param t     Pointer to immutable task descriptor.
 * @param begin First index in range (inclusive).
 * @param end   One past last index (exclusive).
 */
static void ie_run_shard(const ie_task_t *t, size_t begin, size_t end) {
  t->fn(t->ctx, begin, end);
}

/**
 * @brief Worker main loop with epoch-based single-execution guarantee.
 *
 * Each worker tracks the last processed epoch and runs exactly once whenever
 * the pool's epoch increments under ::IE_CMD_RUN.
 *
 * Contiguous block partition formula (balanced, no gaps/overlaps):
 *   base = n / T; rem = n % T;
 *   begin = tid * base + min(tid, rem);
 *   len   = base + (tid < rem ? 1 : 0);
 *   end   = begin + len;
 *
 * @param arg Pointer to #ie_worker_arg_t containing pool and tid.
 * @return Always NULL.
 */
static void* ie_worker_main(void *arg) {
  ie_worker_arg_t *wa = (ie_worker_arg_t*)arg;
  struct ie_threadpool *tp = wa->tp;
  const unsigned tid = wa->tid;

  unsigned seen_epoch = 0; /* local epoch tracker */

  ie_set_affinity(tid, tp->nthreads, tp->affinity);

  for (;;) {
    /* Wait for a new RUN epoch or STOP command. */
    pthread_mutex_lock(&tp->mtx);
    while (tp->cmd != IE_CMD_STOP &&
           !(tp->cmd == IE_CMD_RUN && tp->epoch != seen_epoch)) {
      pthread_cond_wait(&tp->cv, &tp->mtx);
    }
    if (tp->cmd == IE_CMD_STOP) { pthread_mutex_unlock(&tp->mtx); break; }

    /* Snapshot epoch and task payload under the lock. */
    const unsigned my_epoch = tp->epoch;
    ie_task_t task = tp->task;
    unsigned T = tp->nthreads;
    pthread_mutex_unlock(&tp->mtx);

    /* Compute contiguous block for this worker. */
    size_t n = task.n;
    if (T == 0) T = 1; /* safety */
    size_t base = n / (size_t)T;
    size_t rem  = n % (size_t)T;

    size_t begin = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
    size_t len   = base + (((size_t)tid < rem) ? 1u : 0u);
    size_t end   = begin + len;
    if (end > n) end = n;

    if (begin < end) ie_run_shard(&task, begin, end);

    /* Publish completion for this epoch. */
    pthread_mutex_lock(&tp->mtx);
    if (seen_epoch != my_epoch) {
      seen_epoch = my_epoch;
      tp->arrived++;
      pthread_cond_signal(&tp->cv_done);
    }
    pthread_mutex_unlock(&tp->mtx);
  }
  return NULL;
}

/**
 * @brief Create a new thread pool.
 *
 * @param nthreads Number of worker threads; 0 selects the number of online CPUs.
 * @param affinity Affinity hint ("auto"|"compact"|"scatter"); ignored if disabled.
 * @return Pointer to the initialized pool, or NULL on failure.
 */
ie_threadpool_t* ie_tp_create(unsigned nthreads, const char *affinity) {
  ie_threadpool_t *tp = (ie_threadpool_t*)calloc(1, sizeof(*tp));
  if (!tp) return NULL;

  pthread_mutex_init(&tp->mtx, NULL);
  pthread_cond_init(&tp->cv, NULL);
  pthread_cond_init(&tp->cv_done, NULL);

  tp->nthreads = (nthreads == 0) ? ie_nproc() : nthreads;
  if (tp->nthreads < 1) tp->nthreads = 1;

  /* Store affinity mode (default compact). */
  strncpy(tp->affinity, (affinity && *affinity) ? affinity : "compact", sizeof(tp->affinity) - 1);
  tp->affinity[sizeof(tp->affinity)-1] = '\0';

  tp->cmd     = IE_CMD_NONE;
  tp->arrived = 0;
  tp->epoch   = 0;

  tp->thr  = (pthread_t*)calloc(tp->nthreads, sizeof(pthread_t));
  tp->warg = (ie_worker_arg_t*)calloc(tp->nthreads, sizeof(ie_worker_arg_t));
  if (!tp->thr || !tp->warg) {
    free(tp->thr); free(tp->warg); free(tp);
    return NULL;
  }

  for (unsigned i = 0; i < tp->nthreads; ++i) {
    tp->warg[i].tp = tp;
    tp->warg[i].tid = i;
    if (pthread_create(&tp->thr[i], NULL, ie_worker_main, &tp->warg[i]) != 0) {
      tp->nthreads = i;
      break;
    }
  }
  return tp;
}

/**
 * @brief Destroy a thread pool and join all worker threads.
 *
 * @param tp Pool pointer (NULL allowed; no-op).
 */
void ie_tp_destroy(ie_threadpool_t *tp) {
  if (!tp) return;
  /* Broadcast stop and join. */
  pthread_mutex_lock(&tp->mtx);
  tp->cmd = IE_CMD_STOP;
  pthread_cond_broadcast(&tp->cv);
  pthread_mutex_unlock(&tp->mtx);

  for (unsigned i = 0; i < tp->nthreads; ++i) {
    pthread_join(tp->thr[i], NULL);
  }

  pthread_cond_destroy(&tp->cv_done);
  pthread_cond_destroy(&tp->cv);
  pthread_mutex_destroy(&tp->mtx);

  free(tp->thr);
  free(tp->warg);
  free(tp);
}

/**
 * @brief Execute a parallel for over the range [0, n).
 *
 * Falls back to single-threaded execution when @p tp is NULL, the pool has
 * only one thread, or the range size is <= 1.
 *
 * @param tp        Thread pool (may be NULL).
 * @param n         Number of elements in the range.
 * @param fn        Range task callback (must not be NULL).
 * @param ctx       User context passed to @p fn.
 * @param grainsize Minimum shard size (currently informational).
 * @return true on success; false if @p fn is NULL.
 */
bool ie_tp_parallel_for(ie_threadpool_t *tp,
                        size_t n,
                        ie_range_task_fn fn,
                        void *ctx,
                        size_t grainsize) {
  (void)grainsize; /* contiguous partition ignores grainsize for now */
  if (!fn) return false;

  /* Single-threaded fallback when no pool or trivial range. */
  if (!tp || tp->nthreads <= 1 || n <= 1) {
    fn(ctx, 0, n ? n : 0);
    return true;
  }

  /* Submit task: set payload, bump epoch, broadcast RUN, wait for completion. */
  pthread_mutex_lock(&tp->mtx);
  tp->task.fn = fn;
  tp->task.ctx = ctx;
  tp->task.n = n;
  tp->task.grainsize = 0;
  tp->arrived = 0;
  tp->cmd = IE_CMD_RUN;
  tp->epoch++;                      /* new task generation */
  const unsigned this_epoch = tp->epoch;
  pthread_cond_broadcast(&tp->cv);

  /* Wait until all workers report completion for this epoch. */
  while (tp->arrived < tp->nthreads) {
    pthread_cond_wait(&tp->cv_done, &tp->mtx);
    /* Defensive: ignore stale completions from older epochs. */
    if (tp->cmd != IE_CMD_RUN || tp->epoch != this_epoch) break;
  }
  tp->cmd = IE_CMD_NONE;
  pthread_mutex_unlock(&tp->mtx);
  return true;
}
