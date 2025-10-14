/**
 * @file thread_pool.c
 * @brief Minimal pthread thread pool with blocking parallel_for and optional CPU affinity.
 *
 * Properties:
 *  - Deterministic contiguous partition of [0, n) across T workers.
 *  - Robust epoch/barrier ensures each worker executes once per task.
 *  - CPU affinity is enabled at runtime on Linux only when IE_TP_USE_AFFINITY=1.
 */

#if defined(__linux__)
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

/** @brief Control command for workers. */
typedef enum { IE_CMD_NONE=0, IE_CMD_RUN=1, IE_CMD_STOP=2 } ie_cmd_t;

/** @brief Task descriptor for a single parallel_for. */
typedef struct {
  ie_range_task_fn fn;  /**< User callback. */
  void *ctx;            /**< User context. */
  size_t n;             /**< Total range. */
  size_t grainsize;     /**< Informational grainsize (unused). */
} ie_task_t;

/** @brief Worker argument. */
typedef struct {
  struct ie_threadpool *tp; /**< Owning pool. */
  unsigned tid;             /**< Worker id in [0, nthreads). */
} ie_worker_arg_t;

/** @brief Thread pool state. */
struct ie_threadpool {
  unsigned nthreads;         /**< Number of workers. */
  pthread_t *thr;            /**< Worker thread handles. */
  ie_worker_arg_t *warg;     /**< Worker launch args. */

  pthread_mutex_t mtx;       /**< Mutex for task/epoch. */
  pthread_cond_t  cv;        /**< Condition to signal new work. */
  pthread_cond_t  cv_done;   /**< Condition to signal completion. */

  ie_cmd_t  cmd;             /**< Current command. */
  ie_task_t task;            /**< Current task. */

  unsigned  arrived;         /**< # of workers that completed current epoch. */
  unsigned  epoch;           /**< Task sequence number. */

  char      affinity[16];    /**< Affinity mode hint string. */
  int       affinity_enabled;/**< 1 if runtime affinity is enabled on Linux. */
};

/**
 * @brief Query the number of online processors (>=1).
 *
 * @return Number of CPUs online.
 */
static unsigned ie_nproc(void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (unsigned)(n > 0 ? n : 1);
}

/**
 * @brief Apply best-effort CPU affinity for the calling thread on Linux.
 *
 * @param tid       Worker id.
 * @param nthreads  Total workers.
 * @param mode      Affinity mode ("auto"|"compact"|"scatter").
 * @param enabled   Non-zero to enable; zero to skip.
 */
static void ie_set_affinity(unsigned tid, unsigned nthreads, const char *mode, int enabled) {
#if defined(__linux__)
  if (!enabled) return;
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
    /* "auto" or "compact" -> pack from CPU0 upward */
    cpu = tid % ncpu;
  }

  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)tid; (void)nthreads; (void)mode; (void)enabled;
#endif
}

/**
 * @brief Execute the shard [begin, end) for a given task.
 *
 * @param t     Task descriptor.
 * @param begin Start index (inclusive).
 * @param end   End index (exclusive).
 */
static void ie_run_shard(const ie_task_t *t, size_t begin, size_t end) {
  t->fn(t->ctx, begin, end);
}

/**
 * @brief Worker thread main loop.
 *
 * @param arg Pointer to worker argument structure.
 * @return NULL.
 */
static void* ie_worker_main(void *arg) {
  ie_worker_arg_t *wa = (ie_worker_arg_t*)arg;
  struct ie_threadpool *tp = wa->tp;
  const unsigned tid = wa->tid;

  unsigned seen_epoch = 0;

  ie_set_affinity(tid, tp->nthreads, tp->affinity, tp->affinity_enabled);

  for (;;) {
    pthread_mutex_lock(&tp->mtx);
    while (tp->cmd != IE_CMD_STOP &&
           !(tp->cmd == IE_CMD_RUN && tp->epoch != seen_epoch)) {
      pthread_cond_wait(&tp->cv, &tp->mtx);
    }
    if (tp->cmd == IE_CMD_STOP) { pthread_mutex_unlock(&tp->mtx); break; }

    const unsigned my_epoch = tp->epoch;
    ie_task_t task = tp->task;
    unsigned T = tp->nthreads;
    pthread_mutex_unlock(&tp->mtx);

    size_t n = task.n;
    if (T == 0) T = 1;

    size_t base = n / (size_t)T;
    size_t rem  = n % (size_t)T;

    size_t begin = (size_t)tid * base + ((size_t)tid < rem ? (size_t)tid : rem);
    size_t len   = base + (((size_t)tid < rem) ? 1u : 0u);
    size_t end   = begin + len;
    if (end > n) end = n;

    if (begin < end) ie_run_shard(&task, begin, end);

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
 * @brief Create a thread pool.
 *
 * @param nthreads Number of threads (>=1). 0 auto-detects CPU count.
 * @param affinity Hint string ("auto"|"compact"|"scatter").
 * @return Thread pool pointer or NULL on failure.
 */
ie_threadpool_t* ie_tp_create(unsigned nthreads, const char *affinity) {
  ie_threadpool_t *tp = (ie_threadpool_t*)calloc(1, sizeof(*tp));
  if (!tp) return NULL;

  pthread_mutex_init(&tp->mtx, NULL);
  pthread_cond_init(&tp->cv, NULL);
  pthread_cond_init(&tp->cv_done, NULL);

  tp->nthreads = (nthreads == 0) ? ie_nproc() : nthreads;
  if (tp->nthreads < 1) tp->nthreads = 1;

  strncpy(tp->affinity, (affinity && *affinity) ? affinity : "compact",
          sizeof(tp->affinity) - 1);
  tp->affinity[sizeof(tp->affinity)-1] = '\0';

#if defined(__linux__)
  const char *env = getenv("IE_TP_USE_AFFINITY");
  tp->affinity_enabled = (env && env[0] == '1') ? 1 : 0;
#else
  tp->affinity_enabled = 0;
#endif

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
    tp->warg[i].tp  = tp;
    tp->warg[i].tid = i;
    if (pthread_create(&tp->thr[i], NULL, ie_worker_main, &tp->warg[i]) != 0) {
      tp->nthreads = i;
      break;
    }
  }
  return tp;
}

/**
 * @brief Destroy a thread pool (joins workers and frees memory).
 *
 * @param tp Thread pool handle (may be NULL).
 */
void ie_tp_destroy(ie_threadpool_t *tp) {
  if (!tp) return;
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
 * @brief Blocking parallel-for over [0, n).
 *
 * @param tp        Thread pool (NULL => single-thread).
 * @param n         Total range length.
 * @param fn        Callback (must not be NULL).
 * @param ctx       Opaque user context.
 * @param grainsize Informational only; currently unused.
 * @return true on success; false on invalid arguments.
 */
bool ie_tp_parallel_for(ie_threadpool_t *tp,
                        size_t n,
                        ie_range_task_fn fn,
                        void *ctx,
                        size_t grainsize) {
  (void)grainsize;
  if (!fn) return false;

  if (!tp || tp->nthreads <= 1 || n <= 1) {
    fn(ctx, 0, n ? n : 0);
    return true;
  }

  pthread_mutex_lock(&tp->mtx);
  tp->task.fn = fn;
  tp->task.ctx = ctx;
  tp->task.n = n;
  tp->task.grainsize = 0;
  tp->arrived = 0;
  tp->cmd = IE_CMD_RUN;
  tp->epoch++;
  const unsigned this_epoch = tp->epoch;
  pthread_cond_broadcast(&tp->cv);

  while (tp->arrived < tp->nthreads) {
    pthread_cond_wait(&tp->cv_done, &tp->mtx);
    if (tp->cmd != IE_CMD_RUN || tp->epoch != this_epoch) break;
  }
  tp->cmd = IE_CMD_NONE;
  pthread_mutex_unlock(&tp->mtx);
  return true;
}
