/**
 * @file ie_batcher.h
 * @brief Public API for asynchronous prompt prefetch + tokenization batcher.
 *
 * The batcher asynchronously tokenizes input prompts on background worker
 * threads and enqueues results in a bounded ring buffer. The consumer reads
 * *contiguous microbatches* (views) directly from the ring without copying,
 * then advances to release those items.
 *
 * Ownership:
 * - The tokenizer callback must allocate `*out_ids` for each item.
 * - The batcher takes ownership on enqueue and frees payloads on advance.
 */

#ifndef IE_BATCHER_H
#define IE_BATCHER_H

#include <stddef.h>   /* size_t */
#include <stdint.h>   /* uint32_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Tokenized item produced by the batcher.
 */
typedef struct {
  const char *prompt;   /**< Borrowed pointer to the original prompt string. */
  uint32_t *ids;        /**< Owned array of token IDs (malloc'ed by callback). */
  size_t n_ids;         /**< Number of token IDs in @ref ids. */
  int status;           /**< 0 on success; implementation-defined non-zero on error. */
} ie_batcher_item_t;

/**
 * @brief Microbatch view returned by @ref ie_batcher_get.
 *
 * The view points *into* the batcher's ring; do not free or retain the
 * pointers beyond the call to @ref ie_batcher_advance.
 */
typedef struct ie_batcher_view {
  ie_batcher_item_t *items; /**< Contiguous slice of items inside the ring. */
  size_t count;             /**< Number of items in @ref items. */
  int done;                 /**< 1 if the stream is drained; 0 otherwise. */
} ie_batcher_view_t;

/**
 * @brief Opaque batcher handle.
 */
typedef struct ie_batcher ie_batcher_t;

/**
 * @brief Tokenizer callback signature used by the batcher workers.
 *
 * The implementation must allocate `*out_ids` (e.g., via `malloc`) and set
 * `*out_n`. The batcher takes ownership of `*out_ids` after enqueue.
 *
 * @param prompt Zero-terminated input string to tokenize.
 * @param out_ids Output: allocated array of token IDs.
 * @param out_n Output: number of IDs in @p *out_ids.
 * @param user_ctx Opaque pointer passed through from @ref ie_batcher_create.
 * @return int 0 on success, non-zero on failure.
 */
typedef int (*ie_batcher_tokenize_fn)(const char *prompt,
                                      uint32_t **out_ids,
                                      size_t *out_n,
                                      void *user_ctx);

/**
 * @brief Create a new asynchronous tokenizer batcher.
 *
 * Spawns up to @p n_workers threads that prefetch and tokenize prompts into a
 * bounded ring buffer. The consumer should call @ref ie_batcher_get followed by
 * @ref ie_batcher_advance in a loop until the stream is drained.
 *
 * Defaults:
 * - If @p microbatch == 0, it is treated as 1.
 * - If @p queue_capacity == 0, it is treated as max(microbatch, 8).
 * - If @p n_workers == 0, it is treated as 1.
 *
 * @param prompts Array of input UTF-8 prompt strings (not owned; must outlive the batcher).
 * @param n_prompts Number of prompts in @p prompts.
 * @param microbatch Maximum items returned per view. Use 0 for default.
 * @param queue_capacity Ring capacity in items. Use 0 for default.
 * @param n_workers Number of tokenizer worker threads. Use 0 for default.
 * @param tokenize_cb User tokenizer callback (must not be NULL).
 * @param user_ctx Opaque pointer forwarded to @p tokenize_cb; may be NULL.
 * @return ie_batcher_t* Handle on success, or NULL on allocation/spawn error.
 */
ie_batcher_t *ie_batcher_create(const char **prompts,
                                size_t n_prompts,
                                size_t microbatch,
                                size_t queue_capacity,
                                unsigned n_workers,
                                ie_batcher_tokenize_fn tokenize_cb,
                                void *user_ctx);

/**
 * @brief Obtain a contiguous slice of ready items without copying.
 *
 * If no items are ready but production is still in progress, this function
 * blocks until an item is available (or the batcher is stopping).
 *
 * When this function returns 0, the stream is drained; @p out_view->done
 * will be set to 1 and @p out_view->items will be NULL.
 *
 * @param b Batcher handle (non-NULL).
 * @param out_view Output view describing a contiguous span of items.
 * @return int 1 if a non-empty view was produced; 0 if the stream is drained.
 */
int ie_batcher_get(ie_batcher_t *b, ie_batcher_view_t *out_view);

/**
 * @brief Release items previously returned by @ref ie_batcher_get.
 *
 * Frees payloads and advances the ring read pointer, then signals producers
 * that space is available.
 *
 * @param b Batcher handle (non-NULL).
 * @param n_consumed Number of items to release; must not exceed the last view count.
 */
void ie_batcher_advance(ie_batcher_t *b, size_t n_consumed);

/**
 * @brief Destroy a batcher, joining workers and freeing all resources.
 *
 * Safe to call with NULL.
 *
 * @param b Batcher handle (may be NULL).
 */
void ie_batcher_destroy(ie_batcher_t *b);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_BATCHER_H */
