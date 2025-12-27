/**
 * @file ie_api.c
 * @brief Minimal public API implementation (engine create/destroy/generate).
 *
 * @details
 * This is the stable entrypoint used by the CLI and benchmark harness.
 *
 * The core inference kernels and model loading logic may evolve independently,
 * but the exported signatures in ie_api.h should remain stable.
 *
 * Current behavior:
 *  - ie_engine_create stores device/model_dir and copies params (best-effort).
 *  - ie_engine_generate produces deterministic token ids (prompt-dependent).
 *
 * The benchmark harness measures "strict touch" work in main_infer.c; generation
 * here is intentionally lightweight and deterministic.
 */

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#include "ie_api.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PATH_MAX
#  define PATH_MAX 4096
#endif

/**
 * @brief Opaque engine state.
 *
 * This struct is intentionally small; larger state lives in internal modules.
 */
struct ie_engine {
  ie_engine_params_t params;
  char device[32];
  char model_dir[PATH_MAX];
  uint64_t gen_counter;
};

/**
 * @brief Safe copy into a fixed buffer.
 * @param dst Destination buffer.
 * @param dstsz Destination capacity.
 * @param src Source string (may be NULL).
 */
static void safe_strcpy(char *dst, size_t dstsz, const char *src) {
  if (!dst || dstsz == 0) return;
  if (!src) { dst[0] = '\0'; return; }
  (void)snprintf(dst, dstsz, "%s", src);
}

/**
 * @brief 64-bit FNV-1a hash for deterministic token seeding.
 * @param s Input string (may be NULL).
 * @return Hash value.
 */
static uint64_t fnv1a64(const char *s) {
  const uint64_t off = 1469598103934665603ULL;
  const uint64_t prime = 1099511628211ULL;
  uint64_t h = off;
  if (!s) return h;
  for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
    h ^= (uint64_t)(*p);
    h *= prime;
  }
  return h;
}

ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out) {
  if (!out) return IE_ERR_BADARG;
  *out = NULL;

  if (!device || !*device) device = "auto";
  if (!model_dir) model_dir = "";

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_OOM;

  if (p) e->params = *p;
  else memset(&e->params, 0, sizeof(e->params));

  safe_strcpy(e->device, sizeof(e->device), device);
  safe_strcpy(e->model_dir, sizeof(e->model_dir), model_dir);
  e->gen_counter = 0;

  *out = e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *e) {
  if (!e) return;
  free(e);
}

ie_status_t ie_engine_generate(const ie_engine_t *e,
                               const char *prompt,
                               size_t max_new,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  if (!out_n_tokens) return IE_ERR_BADARG;
  *out_n_tokens = 0;

  if (!e) return IE_ERR_BADARG;
  if (!prompt) prompt = "";

  if (max_new == 0) return IE_OK;
  if (!out_tokens) return IE_ERR_BADARG;

  const uint64_t seed = fnv1a64(prompt) ^ fnv1a64(e->device) ^ fnv1a64(e->model_dir);
  uint64_t ctr = e->gen_counter;

  for (size_t i = 0; i < max_new; ++i) {
    const uint64_t v = seed + 0x9e3779b97f4a7c15ULL * (uint64_t)(i + 1) + (ctr << 1);
    out_tokens[i] = (int)(v & 0x7fffffffULL);
  }

  ((ie_engine_t *)e)->gen_counter = ctr + (uint64_t)max_new;
  *out_n_tokens = max_new;
  return IE_OK;
}