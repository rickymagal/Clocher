/**
 * @file test_threadpool.c
 * @brief Unit test for the pthread-based thread pool and parallel_for.
 *
 * The test allocates an array of N flags, runs a parallel_for that sets
 * each position to 1 exactly once, and then verifies full coverage.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_threadpool.h"

/**
 * @brief Context object holding the hits array to be updated.
 */
typedef struct {
  unsigned *hits;  /**< @brief Pointer to an array of length N that will be set to 1. */
} tp_ctx_t;

/**
 * @brief Range task that marks indices in [begin, end) as visited.
 *
 * @param ctx   Pointer to #tp_ctx_t containing the hits array.
 * @param begin First index (inclusive).
 * @param end   One past last index (exclusive).
 */
static void mark_task(void *ctx, size_t begin, size_t end) {
  tp_ctx_t *c = (tp_ctx_t*)ctx;
  for (size_t i = begin; i < end; ++i) c->hits[i] = 1u;
}

/**
 * @brief Test entry point.
 *
 * Builds a thread pool (auto thread count), executes a parallel_for over N
 * elements with the @ref mark_task callback, and then asserts that every
 * element has been set exactly once.
 *
 * @return 0 on success, non-zero on failure.
 */
int main(void) {
  const size_t N = 1000u;
  unsigned *hits = (unsigned*)calloc(N, sizeof(unsigned));
  if (!hits) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  tp_ctx_t ctx = { hits };

  ie_threadpool_t *tp = ie_tp_create(0, "auto");
  assert(tp != NULL);

  const bool ok = ie_tp_parallel_for(tp, N, mark_task, &ctx, 0);
  assert(ok);

  ie_tp_destroy(tp);

  size_t sum = 0u;
  for (size_t i = 0; i < N; ++i) sum += (size_t)hits[i];
  free(hits);

  assert(sum == N);
  printf("ok test_threadpool\n");
  return 0;
}
