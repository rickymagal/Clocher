/**
 * @file ie_batcher.c
 * @brief Asynchronous prompt prefetch + tokenization batcher with in-order commit.
 *
 * Workers pre-tokenize input prompts and place results into a pending array
 * indexed by the original prompt order. A single commit path (under the mutex)
 * flushes the pending items into a bounded ring buffer **in order** so that the
 * consumer always observes FIFO semantics w.r.t. the input prompt list.
 */

#include "ie_batcher.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Opaque struct definition (matches forward-declaration in header)
 * ==========================================================================*/

/**
 * @brief Internal batcher state.
 */
struct ie_batcher {
  /* Inputs */
  const char **prompts;      /**< Array of input prompts (size n_prompts). */
  size_t       n_prompts;    /**< Number of input prompts. */

  /* Ring buffer (visible to the consumer via ie_batcher_get/advance) */
  ie_batcher_item_t *ring;   /**< Circular queue of produced items. */
  size_t cap;                /**< Capacity of @ref ring in items. */
  size_t r;                  /**< Read index (consumer side). */
  size_t w;                  /**< Write index (producer commit side). */
  size_t size;               /**< Number of queued items currently in @ref ring. */

  /* Production progress */
  size_t produced;           /**< Number of indices assigned to workers so far. */
  size_t committed;          /**< Number of items committed to ring in-order. */
  int    assigned_done;      /**< 1 when all indices have been assigned. */
  int    producer_done;      /**< 1 when all items have been committed to ring. */

  /* Pending area for in-order commit */
  ie_batcher_item_t *pend;   /**< Pending items, indexed by original order. */
  unsigned char     *ready;  /**< Ready flags per pending slot (0/1). */

  /* Concurrency primitives */
  pthread_mutex_t mu;        /**< Global mutex protecting all shared state. */
  pthread_cond_t  cv_not_full;  /**< Ring not full signal. */
  pthread_cond_t  cv_not_empty; /**< Ring not empty signal. */

  /* Workers */
  unsigned    n_workers;     /**< Number of worker threads. */
  pthread_t  *threads;       /**< Worker thread handles. */
  int         stop;          /**< Cooperative stop flag for teardown. */

  /* Tokenization callback */
  ie_batcher_tokenize_fn tokenize; /**< User-provided tokenizer callback. */
  void *user_ctx;                  /**< User context for @ref tokenize. */

  /* Controls */
  size_t microbatch;         /**< Max items returned per view (contiguous slice). */
};

/* ============================================================================
 * Internal utilities
 * ==========================================================================*/

/**
 * @brief Allocate zero-initialized memory.
 *
 * @param n Number of elements to allocate.
 * @param sz Size in bytes for each element.
 * @return Pointer to zeroed memory or NULL on failure.
 */
static void *xcalloc(size_t n, size_t sz) {
  return calloc(n, sz);
}

/**
 * @brief Free an item's dynamic payload and reset its fields.
 *
 * @param it Item to clear (may be NULL).
 * @return void
 */
static void clear_item(ie_batcher_item_t *it) {
  if (!it) return;
  if (it->ids) free((void *)it->ids);
  it->ids    = NULL;
  it->n_ids  = 0;
  it->status = 0;
  it->prompt = NULL;
}

/* ============================================================================
 * Producer-side helpers (all under mutex unless stated)
 * ==========================================================================*/

/**
 * @brief Claim the next prompt index to tokenize (thread-safe).
 *
 * Increments @ref ie_batcher::produced and returns that index to the caller.
 * Sets @ref ie_batcher::assigned_done when all indices have been handed out.
 *
 * @param b Batcher (non-NULL).
 * @param out_idx Output index in range [0, n_prompts).
 * @return 1 if an index was produced; 0 if there is no more work.
 */
static int pop_next(struct ie_batcher *b, size_t *out_idx) {
  int ret = 0;
  pthread_mutex_lock(&b->mu);
  if (b->produced < b->n_prompts) {
    *out_idx = b->produced++;
    ret = 1;
    if (b->produced == b->n_prompts) {
      b->assigned_done = 1; /* All indices distributed to workers */
    }
  }
  pthread_mutex_unlock(&b->mu);
  return ret;
}

/**
 * @brief Commit ready pending items to the ring in input order.
 *
 * Flushes @ref ie_batcher::pend starting at @ref ie_batcher::committed while
 * the corresponding @ref ie_batcher::ready flag is set and the ring has space.
 * This preserves strict FIFO order w.r.t. the original prompt list.
 *
 * @param b Batcher (non-NULL). Caller must hold @ref ie_batcher::mu.
 * @return void
 */
static void flush_commits_locked(struct ie_batcher *b) {
  while (b->committed < b->n_prompts && b->ready[b->committed]) {
    /* Wait for ring space if full. */
    while (b->size == b->cap && !b->stop) {
      pthread_cond_wait(&b->cv_not_full, &b->mu);
    }
    if (b->stop) break;

    /* Move pending[committed] into ring at write index. */
    const size_t idx = b->committed;
    b->ring[b->w] = b->pend[idx];         /* struct move; transfers ownership */
    b->ready[idx] = 0;                     /* consumed from pending */
    b->w = (b->w + 1) % b->cap;
    b->size++;
    b->committed++;

    pthread_cond_signal(&b->cv_not_empty);
  }

  /* If everything has been committed, mark producer_done. */
  if (b->committed == b->n_prompts) {
    b->producer_done = 1;
  }
}

/**
 * @brief Place a tokenized item into the pending slot and attempt flush.
 *
 * This stores @p it into @ref ie_batcher::pend at position @p idx, marks it
 * ready, and then calls @ref flush_commits_locked() to push any now-contiguous
 * range into the ring buffer in-order.
 *
 * @param b Batcher (non-NULL).
 * @param idx Original prompt index for this item.
 * @param it Fully-formed item (payload ownership transfers into @p b->pend[idx]).
 * @return void
 */
static void commit_item(struct ie_batcher *b, size_t idx, const ie_batcher_item_t *it) {
  pthread_mutex_lock(&b->mu);

  /* Store into pending area and mark ready. */
  b->pend[idx]  = *it;   /* struct copy; moves pointers (ids) */
  b->ready[idx] = 1;

  /* Try to flush any contiguous run starting at committed. */
  flush_commits_locked(b);

  pthread_mutex_unlock(&b->mu);
}

/**
 * @brief Worker thread main loop: tokenize and commit results.
 *
 * @param arg Pointer to @ref ie_batcher (struct ie_batcher*).
 * @return Always NULL.
 */
static void *worker_main(void *arg) {
  struct ie_batcher *b = (struct ie_batcher *)arg;

  for (;;) {
    size_t idx = 0;
    if (!pop_next(b, &idx)) break; /* all assigned */

    /* Build the item for this prompt. */
    ie_batcher_item_t it;
    memset(&it, 0, sizeof(it));
    it.prompt = b->prompts[idx];

    uint32_t *ids = NULL;
    size_t n_ids = 0;
    int st = b->tokenize(it.prompt, &ids, &n_ids, b->user_ctx);

    it.ids    = ids;
    it.n_ids  = n_ids;
    it.status = st;

    /* Commit into pending and try to flush in-order. */
    commit_item(b, idx, &it);
  }

  return NULL;
}

/* ============================================================================
 * Public API
 * ==========================================================================*/

/**
 * @brief Create a batcher and spawn worker threads.
 *
 * @param prompts         Array of NUL-terminated prompt strings (size @p n_prompts).
 * @param n_prompts       Number of prompts in @p prompts.
 * @param microbatch      Maximum items returned per view. If zero, defaults to 1.
 * @param queue_capacity  Ring capacity. If zero, defaults to max(8, microbatch).
 * @param n_workers       Number of worker threads. If zero, defaults to 1.
 * @param tokenize_cb     Tokenization callback (non-NULL).
 * @param user_ctx        Opaque pointer passed to @p tokenize_cb.
 * @return Opaque handle on success; NULL on allocation or init failure.
 */
ie_batcher_t *ie_batcher_create(const char **prompts,
                                size_t n_prompts,
                                size_t microbatch,
                                size_t queue_capacity,
                                unsigned n_workers,
                                ie_batcher_tokenize_fn tokenize_cb,
                                void *user_ctx) {
  if (!prompts || n_prompts == 0 || !tokenize_cb) return NULL;
  if (microbatch == 0)  microbatch = 1;
  if (queue_capacity == 0) queue_capacity = (microbatch > 8 ? microbatch : 8);
  if (n_workers == 0)   n_workers = 1;

  struct ie_batcher *b = (struct ie_batcher *)xcalloc(1, sizeof(*b));
  if (!b) return NULL;

  b->prompts    = prompts;
  b->n_prompts  = n_prompts;
  b->microbatch = microbatch;

  b->cap  = queue_capacity;
  b->ring = (ie_batcher_item_t *)xcalloc(b->cap, sizeof(*b->ring));
  b->pend = (ie_batcher_item_t *)xcalloc(n_prompts, sizeof(*b->pend));
  b->ready = (unsigned char *)xcalloc(n_prompts, sizeof(*b->ready));
  if (!b->ring || !b->pend || !b->ready) {
    free(b->ring); free(b->pend); free(b->ready); free(b);
    return NULL;
  }

  pthread_mutex_init(&b->mu, NULL);
  pthread_cond_init(&b->cv_not_full, NULL);
  pthread_cond_init(&b->cv_not_empty, NULL);

  b->tokenize  = tokenize_cb;
  b->user_ctx  = user_ctx;

  b->n_workers = n_workers;
  b->threads   = (pthread_t *)xcalloc(n_workers, sizeof(pthread_t));
  if (!b->threads) {
    free(b->ready); free(b->pend); free(b->ring); free(b);
    return NULL;
  }

  /* Spawn workers; pass stable pointer b as the thread arg. */
  for (unsigned i = 0; i < n_workers; ++i) {
    if (pthread_create(&b->threads[i], NULL, worker_main, b) != 0) {
      b->stop = 1; /* best effort; still safe to use if some started */
      break;
    }
  }

  return (ie_batcher_t *)b;
}

/**
 * @brief Obtain a contiguous slice of produced items.
 *
 * This blocks while the ring is empty and production is not finished, then
 * returns at most @ref ie_batcher::microbatch items limited by the contiguous
 * region from the current read index (never wraps within a single view).
 *
 * @param b         Batcher handle (non-NULL).
 * @param out_view  Output view (non-NULL). On success (@c 1), fields are:
 *                  - @c items : pointer to first item in the ring slice
 *                  - @c count : number of items in the slice (>= 1)
 *                  - @c done  : 1 only when production is fully committed and
 *                               the ring becomes empty after consumption; 0 otherwise
 * @return 1 if a non-empty view is provided; 0 if fully drained (no more items).
 */
int ie_batcher_get(ie_batcher_t *b, ie_batcher_view_t *out_view) {
  if (!b || !out_view) return 0;
  struct ie_batcher *hb = (struct ie_batcher *)b;

  pthread_mutex_lock(&hb->mu);

  while (hb->size == 0 && !hb->producer_done && !hb->stop) {
    pthread_cond_wait(&hb->cv_not_empty, &hb->mu);
  }

  if (hb->size == 0) {
    /* Drained or stopping with no queued items. */
    out_view->items = NULL;
    out_view->count = 0;
    out_view->done  = (hb->producer_done != 0);
    pthread_mutex_unlock(&hb->mu);
    return 0;
  }

  const size_t avail = hb->size;
  const size_t first = hb->r;

  size_t take = hb->microbatch;
  if (take > avail) take = avail;

  const size_t tail = hb->cap - first; /* contiguous to end-of-ring */
  if (take > tail) take = tail;

  out_view->items = &hb->ring[first];
  out_view->count = take;
  out_view->done  = 0; /* not drained yet */

  pthread_mutex_unlock(&hb->mu);
  return 1;
}

/**
 * @brief Release @p n_consumed items from the ring and signal space.
 *
 * Frees the payloads of the consumed items (their @c ids arrays) and advances
 * the read index. Signals @ref cv_not_full so producers can make progress.
 *
 * @param b           Batcher handle (non-NULL).
 * @param n_consumed  Number of items to release from the current read position.
 *                    Values larger than the queue size are clamped.
 * @return void
 */
void ie_batcher_advance(ie_batcher_t *b, size_t n_consumed) {
  if (!b || n_consumed == 0) return;
  struct ie_batcher *hb = (struct ie_batcher *)b;

  pthread_mutex_lock(&hb->mu);

  size_t to_free = (n_consumed > hb->size) ? hb->size : n_consumed;
  for (size_t i = 0; i < to_free; ++i) {
    const size_t idx = (hb->r + i) % hb->cap;
    clear_item(&hb->ring[idx]);
  }
  hb->r = (hb->r + to_free) % hb->cap;
  hb->size -= to_free;

  pthread_cond_signal(&hb->cv_not_full);
  pthread_mutex_unlock(&hb->mu);
}

/**
 * @brief Destroy the batcher, join workers, and free all resources.
 *
 * Safe to call with NULL (no-op).
 *
 * @param b Batcher handle to destroy (may be NULL).
 * @return void
 */
void ie_batcher_destroy(ie_batcher_t *b) {
  if (!b) return;
  struct ie_batcher *hb = (struct ie_batcher *)b;

  pthread_mutex_lock(&hb->mu);
  hb->stop = 1;
  pthread_cond_broadcast(&hb->cv_not_full);
  pthread_cond_broadcast(&hb->cv_not_empty);
  pthread_mutex_unlock(&hb->mu);

  for (unsigned i = 0; i < hb->n_workers; ++i) {
    if (hb->threads[i]) (void)pthread_join(hb->threads[i], NULL);
  }

  /* Free any remaining items in the ring and pending slots. */
  for (size_t i = 0; i < hb->cap; ++i) clear_item(&hb->ring[i]);
  for (size_t i = 0; i < hb->n_prompts; ++i) {
    if (hb->ready[i]) clear_item(&hb->pend[i]);
  }

  pthread_cond_destroy(&hb->cv_not_full);
  pthread_cond_destroy(&hb->cv_not_empty);
  pthread_mutex_destroy(&hb->mu);

  free(hb->threads);
  free(hb->ready);
  free(hb->pend);
  free(hb->ring);
  free(hb);
}
