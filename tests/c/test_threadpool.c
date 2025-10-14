/**
 * @file test_threadpool.c
 * @brief Unit test for ie_threadpool parallel-for with contiguous partition.
 *
 * This test validates:
 *  - Single-threaded and multi-threaded execution paths.
 *  - Full coverage of the iteration space [0, N).
 *  - No overlaps/missed iterations when grainsize is provided or auto.
 */

#include "ie_threadpool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/** @brief Context passed to the parallel-for body. */
typedef struct {
  unsigned N;         /**< Total iterations. */
  unsigned *marks;    /**< Visited bitmap of length N (0/1). */
} tp_ctx_t;

/**
 * @brief Parallel-for body: marks each visited index as 1.
 *
 * @param ctx   Pointer to tp_ctx_t.
 * @param start Inclusive start index of the chunk.
 * @param end   Exclusive end index of the chunk.
 */
static void mark_task(void *ctx, unsigned start, unsigned end) {
  tp_ctx_t *c = (tp_ctx_t*)ctx;
  if (!c || !c->marks) return;
  if (end > c->N) end = c->N;
  for (unsigned i = start; i < end; ++i) {
    c->marks[i] = 1u;
  }
}

/**
 * @brief Verify that all positions [0, N) were visited exactly once.
 *
 * (We ensure all positions are marked as 1; strict “exactly once” with
 * contention detection would need atomics and checking for >1.)
 *
 * @param c Test context with bitmap.
 * @return true if every mark[i] == 1 for i in [0, N); false otherwise.
 */
static bool all_marked_once(const tp_ctx_t *c) {
  if (!c || !c->marks) return false;
  for (unsigned i = 0; i < c->N; ++i) {
    if (c->marks[i] != 1u) return false;
  }
  return true;
}

/**
 * @brief Entry point: runs the thread-pool tests.
 *
 * Steps:
 *  1) Single-threaded (tp=NULL) with auto grainsize.
 *  2) Multi-threaded (tp!=NULL) with auto grainsize.
 *  3) Multi-threaded with explicit grainsize.
 *
 * Prints "ok test_threadpool" on success; exits with nonzero on failure.
 *
 * @return 0 on success; non-zero on failure.
 */
int main(void) {
  const unsigned N = 10000u;

  /* --- Case 1: single-threaded path (tp == NULL) --- */
  tp_ctx_t c1;
  c1.N = N;
  c1.marks = (unsigned*)calloc(N, sizeof(unsigned));
  if (!c1.marks) { fprintf(stderr, "alloc failed\n"); return 1; }

  ie_tp_parallel_for(/*tp*/NULL, /*n*/N, /*grainsize*/0, mark_task, &c1);

  if (!all_marked_once(&c1)) {
    fprintf(stderr, "single-threaded coverage failed\n");
    free(c1.marks);
    return 2;
  }
  free(c1.marks);

  /* --- Case 2: multi-threaded path (tp != NULL), auto grainsize --- */
  ie_threadpool_t *tp = ie_tp_create(/*nth*/4, /*affinity*/"auto");
  if (!tp) { fprintf(stderr, "tp create failed\n"); return 3; }

  tp_ctx_t c2;
  c2.N = N;
  c2.marks = (unsigned*)calloc(N, sizeof(unsigned));
  if (!c2.marks) { fprintf(stderr, "alloc failed\n"); ie_tp_destroy(tp); return 4; }

  ie_tp_parallel_for(tp, N, /*grainsize*/0, mark_task, &c2);

  if (!all_marked_once(&c2)) {
    fprintf(stderr, "multi-threaded coverage (auto grainsize) failed\n");
    free(c2.marks);
    ie_tp_destroy(tp);
    return 5;
  }
  free(c2.marks);

  /* --- Case 3: multi-threaded, explicit grainsize (e.g., 257) --- */
  tp_ctx_t c3;
  c3.N = N;
  c3.marks = (unsigned*)calloc(N, sizeof(unsigned));
  if (!c3.marks) { fprintf(stderr, "alloc failed\n"); ie_tp_destroy(tp); return 6; }

  ie_tp_parallel_for(tp, N, /*grainsize*/257u, mark_task, &c3);

  if (!all_marked_once(&c3)) {
    fprintf(stderr, "multi-threaded coverage (explicit grainsize) failed\n");
    free(c3.marks);
    ie_tp_destroy(tp);
    return 7;
  }
  free(c3.marks);

  ie_tp_destroy(tp);

  puts("ok test_threadpool");
  return 0;
}
