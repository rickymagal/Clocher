/**
 * @file ie_batcher.c
 * @brief Prefetch + tokenization batcher with preserved output order.
 *
 * @details
 * Implementation notes:
 * - The public header declares an opaque type via `typedef struct ie_batcher ie_batcher_t;`.
 *   This file **defines** that struct as `struct ie_batcher { ... }` (same tag name),
 *   avoiding any conflicting typedefs.
 * - To preserve strict input order == output order and to keep unit tests deterministic,
 *   this implementation runs with a single producer worker regardless of the requested
 *   `n_workers`. If you want parallel tokenization later, add an ordered merge stage.
 * - The consumer obtains contiguous slices from a ring buffer up to `microbatch` items
 *   and never across the ring wrap boundary.
 */

#include "ie_batcher.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================== */
/* Concrete type                                                              */
/* ========================================================================== */

/**
 * @brief Concrete batcher (matches the opaque tag in the public header).
 */
struct ie_batcher {
  /* Input source */
  const char **prompts;          /**< Input prompts (owned by caller). */
  size_t n_prompts;              /**< Total number of prompts. */

  /* Ring buffer of ready (tokenized) items */
  ie_batcher_item_t *ring;       /**< Circular buffer of produced items. */
  size_t cap;                    /**< Capacity of the ring. */
  size_t r;                      /**< Read index. */
  size_t w;                      /**< Write index. */
  size_t size;                   /**< Number of items currently in the ring. */

  /* Production progress */
  size_t produced;               /**< Number of prompt indices assigned to producer. */
  int producer_done;             /**< 1 when all indices have been assigned. */

  /* Concurrency primitives */
  pthread_mutex_t mu;            /**< Mutex protecting the ring and counters. */
  pthread_cond_t  cv_not_full;   /**< Signaled when the ring transitions from full. */
  pthread_cond_t  cv_not_empty;  /**< Signaled when the ring transitions from empty. */

  /* Workers */
  unsigned n_workers;            /**< Number of producer threads (forced to 1). */
  pthread_t *threads;            /**< Array of thread handles. */
  int stop;                      /**< Stop signal for teardown. */

  /* Tokenization */
  ie_batcher_tokenize_fn tokenize; /**< User-provided tokenization callback. */
  void *user_ctx;                   /**< Opaque user context passed to callback. */

  /* Controls */
  size_t microbatch;             /**< Max items per view slice returned to consumer. */
};

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

/**
 * @brief calloc wrapper (zero-initialized).
 * @param n   Number of elements.
 * @param sz  Size of each element in bytes.
 * @return Pointer to allocated memory or NULL.
 */
static void *xcalloc(size_t n, size_t sz) { return calloc(n, sz); }

/**
 * @brief Free an item's payload (token ids) and reset its fields.
 * @param it  The item to clear (may be NULL).
 */
static void clear_item(ie_batcher_item_t *it) {
  if (!it) return;
  if (it->ids) free(it->ids);
  it->ids = NULL;
  it->n_ids = 0;
  it->status = 0;
  it->prompt = NULL;
}

/* ========================================================================== */
/* Producer side                                                              */
/* ========================================================================== */

/**
 * @brief Thread-safe fetch of the next prompt index to process.
 * @param b        Batcher handle.
 * @param out_idx  Output index in [0, n_prompts).
 * @return 1 if an index was produced; 0 if there is no more work.
 */
static int pop_next(ie_batcher_t *b, size_t *out_idx) {
  int ret = 0;
  pthread_mutex_lock(&b->mu);
  if (b->produced < b->n_prompts) {
    *out_idx = b->produced++;
    ret = 1;
    if (b->produced == b->n_prompts) {
      b->producer_done = 1;
    }
  }
  pthread_mutex_unlock(&b->mu);
  return ret;
}

/**
 * @brief Enqueue a tokenized item into the ring (blocks if full).
 *
 * Ownership of `it->ids` transfers to the ring on success. If the batcher
 * is stopping, the function frees `it->ids` to avoid leaking the payload.
 *
 * @param b   Batcher handle.
 * @param it  Item to enqueue (copied by value).
 */
static void enqueue_item(ie_batcher_t *b, const ie_batcher_item_t *it) {
  pthread_mutex_lock(&b->mu);
  while (b->size == b->cap && !b->stop) {
    pthread_cond_wait(&b->cv_not_full, &b->mu);
  }
  if (!b->stop) {
    b->ring[b->w] = *it; /* struct copy; moves pointer ownership */
    b->w = (b->w + 1) % b->cap;
    b->size++;
    pthread_cond_signal(&b->cv_not_empty);
  } else {
    if (it->ids) free((void*)it->ids);
  }
  pthread_mutex_unlock(&b->mu);
}

/**
 * @brief Producer thread entry: tokenizes prompts in ascending index order.
 * @param arg  `ie_batcher_t*` passed as void*.
 * @return NULL
 */
static void *worker_main(void *arg) {
  ie_batcher_t *b = (ie_batcher_t*)arg;

  for (;;) {
    size_t idx = 0;
    if (!pop_next(b, &idx)) break;

    ie_batcher_item_t it;
    memset(&it, 0, sizeof(it));
    it.prompt = b->prompts[idx];

    uint32_t *ids = NULL;
    size_t n = 0;
    int st = b->tokenize(it.prompt, &ids, &n, b->user_ctx);
    it.ids = ids;
    it.n_ids = n;
    it.status = st;

    enqueue_item(b, &it);
  }
  return NULL;
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

ie_batcher_t *ie_batcher_create(const char **prompts,
                                size_t n_prompts,
                                size_t microbatch,
                                size_t queue_capacity,
                                unsigned n_workers,
                                ie_batcher_tokenize_fn tokenize_cb,
                                void *user_ctx) {
  (void)n_workers; /* determinism: always 1 producer thread */

  if (!prompts || n_prompts == 0 || !tokenize_cb) return NULL;
  if (queue_capacity == 0) queue_capacity = microbatch ? microbatch : 8;

  ie_batcher_t *b = (ie_batcher_t*)xcalloc(1, sizeof(*b));
  if (!b) return NULL;

  b->prompts = prompts;
  b->n_prompts = n_prompts;
  b->microbatch = microbatch ? microbatch : 1;
  b->cap = queue_capacity;
  b->ring = (ie_batcher_item_t*)xcalloc(b->cap, sizeof(*b->ring));
  if (!b->ring) { free(b); return NULL; }

  pthread_mutex_init(&b->mu, NULL);
  pthread_cond_init(&b->cv_not_full, NULL);
  pthread_cond_init(&b->cv_not_empty, NULL);

  b->tokenize = tokenize_cb;
  b->user_ctx = user_ctx;
  b->n_workers = 1;
  b->threads = (pthread_t*)xcalloc(b->n_workers, sizeof(pthread_t));
  if (!b->threads) {
    free(b->ring);
    free(b);
    return NULL;
  }

  if (pthread_create(&b->threads[0], NULL, worker_main, b) != 0) {
    free(b->threads);
    free(b->ring);
    free(b);
    return NULL;
  }

  return b;
}

int ie_batcher_get(ie_batcher_t *b, ie_batcher_view_t *out_view) {
  if (!b || !out_view) return 0;

  pthread_mutex_lock(&b->mu);
  while (b->size == 0 && !b->producer_done && !b->stop) {
    pthread_cond_wait(&b->cv_not_empty, &b->mu);
  }

  if (b->size == 0 && (b->producer_done || b->stop)) {
    out_view->items = NULL;
    out_view->count = 0;
    out_view->done = 1;
    pthread_mutex_unlock(&b->mu);
    return 0;
  }

  size_t avail = b->size;
  size_t first = b->r;
  size_t can_take = b->microbatch;
  if (can_take > avail) can_take = avail;

  /* Return a contiguous tail segment only (no wrap across r->end). */
  size_t tail = b->cap - first;
  if (can_take > tail) can_take = tail;

  out_view->items = &b->ring[first];
  out_view->count = can_take;
  out_view->done = (b->producer_done && (b->size == can_take));

  pthread_mutex_unlock(&b->mu);
  return 1;
}

void ie_batcher_advance(ie_batcher_t *b, size_t n_consumed) {
  if (!b || n_consumed == 0) return;

  pthread_mutex_lock(&b->mu);

  for (size_t i = 0; i < n_consumed; ++i) {
    size_t idx = (b->r + i) % b->cap;
    clear_item(&b->ring[idx]);
  }

  b->r = (b->r + n_consumed) % b->cap;
  b->size -= n_consumed;

  pthread_cond_signal(&b->cv_not_full);
  pthread_mutex_unlock(&b->mu);
}

void ie_batcher_destroy(ie_batcher_t *b) {
  if (!b) return;

  pthread_mutex_lock(&b->mu);
  b->stop = 1;
  pthread_cond_broadcast(&b->cv_not_full);
  pthread_cond_broadcast(&b->cv_not_empty);
  pthread_mutex_unlock(&b->mu);

  for (unsigned i = 0; i < b->n_workers; ++i) {
    if (b->threads[i]) pthread_join(b->threads[i], NULL);
  }

  /* Free any remaining items in the ring. */
  for (size_t i = 0; i < b->cap; ++i) clear_item(&b->ring[i]);

  pthread_cond_destroy(&b->cv_not_full);
  pthread_cond_destroy(&b->cv_not_empty);
  pthread_mutex_destroy(&b->mu);

  free(b->threads);
  free(b->ring);
  free(b);
}
