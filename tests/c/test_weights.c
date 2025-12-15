/* tests/c/test_weights.c */
/**
 * @file test_weights.c
 * @brief Unit tests for baseline and deduplicated IEBIN loaders.
 *
 * This test validates:
 *  - The baseline weights loader can produce a weight view for named tensors.
 *  - The dedup weights loader can produce a weight view for the same tensors.
 *  - Materializing both views produces byte-identical results (lossless guarantee).
 *
 * NOTE:
 * The baseline accessor `ie_weights_get_weight_view()` is implemented in the engine
 * but is currently not declared in any public header included here. Because this
 * project builds with `-Werror`, we provide an explicit prototype in this test
 * file to avoid implicit declaration errors.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_io.h"
#include "weights_dedup.h"

/* ------------------------------------------------------------------------- */
/* Baseline accessor forward declaration                                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Obtain a baseline weight view for a given tensor name.
 *
 * @details
 * This function is expected to be implemented by the baseline loader code
 * (engine/src/io/weights.c). It should fill an @ref ie_weight_view_t describing
 * the tensor’s raw bytes (direct view) or a representation that can be
 * materialized into raw bytes.
 *
 * @param w    Opened weights descriptor.
 * @param name Tensor name (NUL-terminated).
 * @param out  Output view.
 * @return 0 on success, negative on failure.
 */
extern int ie_weights_get_weight_view(const ie_weights_t *w,
                                      const char *name,
                                      ie_weight_view_t *out);

/* ------------------------------------------------------------------------- */
/* Simple 64-bit FNV-1a hash                                                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief Compute a 64-bit FNV-1a hash over a byte buffer.
 *
 * @param data   Pointer to bytes.
 * @param nbytes Number of bytes.
 * @return 64-bit hash value.
 */
static uint64_t hash_bytes(const void *data, size_t nbytes) {
  const uint8_t *p = (const uint8_t *)data;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < nbytes; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

/* ------------------------------------------------------------------------- */
/* View materialization helpers                                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Compute the number of bytes produced when a weight view is materialized.
 *
 * @param v Weight view.
 * @return Required byte size (0 on invalid view).
 */
static size_t view_materialized_nbytes(const ie_weight_view_t *v) {
  if (!v) return 0;

  if (v->kind == IE_WVIEW_DIRECT) {
    return v->nbytes;
  }

  if (v->kind != IE_WVIEW_DEDUP) {
    return 0;
  }

  if (v->dtype == IE_WDTYPE_INT4) {
    /* Packed int4: defaults_nbytes is the packed byte length of the tensor. */
    return v->defaults_nbytes;
  }

  return v->elem_count * v->elem_size;
}

/**
 * @brief Hash the materialized bytes of a weight view.
 *
 * @details
 * - For @ref IE_WVIEW_DIRECT, hashes `data[0..nbytes)`.
 * - For @ref IE_WVIEW_DEDUP, materializes using @ref ie_weights_dedup_materialize
 *   and hashes the resulting contiguous buffer.
 *
 * @param v Weight view.
 * @return 64-bit hash (0 if invalid).
 */
static uint64_t hash_view(const ie_weight_view_t *v) {
  size_t n = view_materialized_nbytes(v);
  if (n == 0) return 0;

  void *buf = malloc(n);
  assert(buf && "OOM in test");

  size_t written = 0;

  if (v->kind == IE_WVIEW_DIRECT) {
    assert(v->data != NULL);
    assert(v->nbytes == n);
    memcpy(buf, v->data, n);
    written = n;
  } else {
    written = ie_weights_dedup_materialize(v, buf, n);
  }

  assert(written == n && "materialization failed");
  uint64_t h = hash_bytes(buf, n);
  free(buf);
  return h;
}

/* ------------------------------------------------------------------------- */
/* Test entrypoint                                                           */
/* ------------------------------------------------------------------------- */

/**
 * @brief Main test entrypoint.
 *
 * @return 0 on success.
 */
int main(void) {
  ie_weights_t w;

  /* --------------------------------------------------------------------- */
  /* 1) Open model                                                         */
  /* --------------------------------------------------------------------- */

  int rc = ie_weights_open("models/gpt-oss-20b/model.ie.json",
                           "models/gpt-oss-20b/model.ie.bin",
                           &w);
  assert(rc == 0);

  /* Dedup should be enabled automatically if model.dedup.json exists */
  assert(w.is_dedup == 1);
  assert(w.dedup_handle != NULL);

  ie_weights_dedup_t *dh = (ie_weights_dedup_t *)w.dedup_handle;

  /* --------------------------------------------------------------------- */
  /* 2) Pick a few known tensors to validate                               */
  /* --------------------------------------------------------------------- */

  const char *names[] = {
    "model.embed_tokens.weight",
    "model.layers.0.mlp.fc1.weight",
    "model.layers.0.self_attn.q_proj.weight"
  };

  const size_t ntests = sizeof(names) / sizeof(names[0]);

  for (size_t i = 0; i < ntests; ++i) {
    const char *name = names[i];

    ie_weight_view_t v_base;
    ie_weight_view_t v_dedup;

    memset(&v_base, 0, sizeof(v_base));
    memset(&v_dedup, 0, sizeof(v_dedup));

    /* ---- baseline view ---- */
    rc = ie_weights_get_weight_view(&w, name, &v_base);
    assert(rc == 0);

    /* ---- dedup view ---- */
    rc = (int)ie_weights_dedup_get_weight_view(dh, name, &v_dedup);
    assert(rc == 0);

    /* ---- compare hashes of materialized bytes ---- */
    uint64_t h_base  = hash_view(&v_base);
    uint64_t h_dedup = hash_view(&v_dedup);

    assert(h_base != 0);
    assert(h_dedup != 0);
    assert(h_base == h_dedup && "DEDUP DATA MISMATCH (NOT LOSSLESS)");
  }

  /* --------------------------------------------------------------------- */
  /* 3) Cleanup                                                            */
  /* --------------------------------------------------------------------- */

  ie_weights_close(&w);

  printf("ok test_weights_dedup (lossless verified)\n");
  return 0;
}
