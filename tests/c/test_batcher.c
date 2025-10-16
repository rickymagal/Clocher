/* tests/c/test_batcher.c */
/**
 * @file test_batcher.c
 * @brief Unit tests for ie_batcher (async prefetch + microbatching).
 *
 * This test uses a deterministic tokenizer that maps each prompt to a short,
 * synthetic token sequence; then it validates:
 *  - Production/consumption completes (done flag).
 *  - Microbatch view respects capacity and ring tail.
 *  - Items are in the same order as input prompts.
 *  - Payload ownership lifecycle is correct (no double frees / leaks).
 */

#include "ie_batcher.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- deterministic "tokenizer" callback ---------- */

/**
 * @brief Deterministic tokenizer used for testing.
 *
 * Produces ids = { len(prompt), sum(bytes)%2048 }.
 * Allocates @p *out_ids with malloc; caller (batcher) takes ownership.
 *
 * @param prompt Zero-terminated input string.
 * @param out_ids Output pointer receiving malloc'ed uint32 array.
 * @param out_n Output count (number of tokens).
 * @param user_ctx Opaque pointer (unused).
 * @return int 0 on success.
 */
static int test_tokenize_cb(const char *prompt, uint32_t **out_ids, size_t *out_n, void *user_ctx) {
  (void)user_ctx;
  size_t L = strlen(prompt);
  unsigned sum = 0;
  for (size_t i = 0; i < L; ++i) sum += (unsigned)(unsigned char)prompt[i];

  *out_n = 2;
  *out_ids = (uint32_t*)malloc(sizeof(uint32_t) * (*out_n));
  if (!*out_ids) return -12;
  (*out_ids)[0] = (uint32_t)L;
  (*out_ids)[1] = (uint32_t)(sum % 2048u);
  return 0;
}

/* ----------------------------- tests ----------------------------- */

/**
 * @brief Validate basic prefetch, ordering, microbatch slicing and draining.
 *
 * Steps:
 * 1) Create batcher with 10 prompts, microbatch=3, ring=5, workers=2.
 * 2) Repeatedly call get/advance until done.
 * 3) Check that prompts order matches outputs' implicit prompt order.
 * 4) Check that each item received 2 tokens and status==0.
 */
static void test_basic_prefetch_and_order(void) {
  const char *prompts[10] = {
    "alpha", "beta", "gamma", "delta", "epsilon",
    "zeta", "eta", "theta", "iota", "kappa"
  };

  ie_batcher_t *b = ie_batcher_create(
      prompts, 10, /*microbatch=*/3, /*queue_capacity=*/5,
      /*n_workers=*/2, test_tokenize_cb, /*user_ctx=*/NULL);
  assert(b && "batcher create");

  size_t seen = 0;
  for (;;) {
    ie_batcher_view_t v;
    int ok = ie_batcher_get(b, &v);
    if (!ok) {
      /* drained: ensure flag set */
      assert(v.done == 1);
      break;
    }
    assert(v.count >= 1 && v.count <= 3);
    for (size_t i = 0; i < v.count; ++i) {
      const ie_batcher_item_t *it = &v.items[i];
      assert(it->prompt == prompts[seen] && "preserve input order");
      assert(it->status == 0);
      assert(it->ids && it->n_ids == 2);
      /* quick content check */
      size_t L = strlen(prompts[seen]);
      assert(it->ids[0] == (uint32_t)L);
      /* ids[1] tested indirectly by tokenizer shape only */
      ++seen;
    }
    ie_batcher_advance(b, v.count);
  }

  assert(seen == 10);
  ie_batcher_destroy(b);
  printf("ok test_basic_prefetch_and_order\n");
}

/**
 * @brief Validate ring wrap behavior (tail truncation) and view sizes.
 *
 * Uses small ring and microbatch to force wrap-around at least once.
 */
static void test_wrap_and_view_limits(void) {
  const char *prompts[7] = {"a","b","c","d","e","f","g"};
  ie_batcher_t *b = ie_batcher_create(
      prompts, 7, /*microbatch=*/4, /*queue_capacity=*/4,
      /*n_workers=*/2, test_tokenize_cb, NULL);
  assert(b);

  size_t total = 0;
  size_t iterations = 0;
  for (;;) {
    ie_batcher_view_t v;
    int ok = ie_batcher_get(b, &v);
    if (!ok) { assert(v.done == 1); break; }
    /* With cap=4, view.count cannot exceed 4, but also cannot exceed the ring tail. */
    assert(v.count >= 1 && v.count <= 4);
    total += v.count;
    iterations++;
    ie_batcher_advance(b, v.count);
  }
  assert(total == 7);
  assert(iterations >= 2); /* should have needed more than one view */
  ie_batcher_destroy(b);
  printf("ok test_wrap_and_view_limits\n");
}

/**
 * @brief Validate graceful teardown: destroy after partial consumption.
 *
 * Requests one view then destroys without advancing further; ensures
 * no crashes and no double-free scenarios.
 */
static void test_partial_teardown(void) {
  const char *prompts[4] = {"p","q","r","s"};
  ie_batcher_t *b = ie_batcher_create(
      prompts, 4, /*microbatch=*/2, /*queue_capacity=*/2,
      /*n_workers=*/1, test_tokenize_cb, NULL);
  assert(b);

  ie_batcher_view_t v;
  int ok = ie_batcher_get(b, &v);
  assert(ok == 1);
  assert(v.count >= 1);

  /* Intentionally skip advance and destroy to force internal cleanup. */
  ie_batcher_destroy(b);
  printf("ok test_partial_teardown\n");
}

int main(void) {
  test_basic_prefetch_and_order();
  test_wrap_and_view_limits();
  test_partial_teardown();
  return 0;
}
