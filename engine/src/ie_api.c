/* ============================================================================
 * File: engine/src/ie_api.c
 * ============================================================================
 */

/**
 * @file ie_api.c
 * @brief Public engine API implementation (model loading + generation entrypoint).
 *
 * @details
 * Responsibilities:
 *  - Create/destroy device backends (CPU/CUDA).
 *  - Load weights (including dedup reconstruction through weights_dedup.c).
 *  - Load tokenizer.json when available for vocab sizing.
 *
 * IMPORTANT:
 * This repository snapshot does not ship a transformer forward pass, so real
 * generation is unavailable unless runtime/infer_gptoss.c is implemented.
 * By default, generate() returns IE_ERR_UNSUPPORTED. A deterministic fake-token
 * path exists only for plumbing tests when IE_ALLOW_FAKE_TOKENS=1.
 */

#include "ie_api.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "ie_device.h"
#include "ie_io.h"
#include "ie_tokenizer_gptoss.h"

/**
 * @brief Parse common boolean environment variables.
 * @param name Environment variable name.
 * @param default_value Value to return when unset or unrecognized.
 * @return 1 if truthy, 0 if falsy, otherwise @p default_value.
 */
static int env_truthy(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !v[0]) return default_value;
  if (strcmp(v, "1") == 0) return 1;
  if (strcmp(v, "true") == 0) return 1;
  if (strcmp(v, "yes") == 0) return 1;
  if (strcmp(v, "on") == 0) return 1;
  if (strcmp(v, "0") == 0) return 0;
  if (strcmp(v, "false") == 0) return 0;
  if (strcmp(v, "no") == 0) return 0;
  if (strcmp(v, "off") == 0) return 0;
  return default_value;
}

/**
 * @brief Engine object (opaque to users of the public API).
 */
struct ie_engine {
  /** @brief Engine parameters (precision, affinity, etc.). */
  ie_engine_params_t params;

  /** @brief Path to model directory. */
  char model_dir[1024];

  /** @brief Device capability info. */
  ie_device_caps_t caps;

  /** @brief Device handle. */
  ie_device_t *dev;

  /** @brief Loaded weights handle. */
  ie_weights_t weights;

  /** @brief Whether weights were opened and must be closed. */
  int weights_open;

  /** @brief GPT-OSS tokenizer handle (optional). */
  ie_tok_gptoss_t *tok;

  /** @brief Vocabulary size; falls back to a default if tokenizer not found. */
  uint32_t vocab_size;
};

/**
 * @brief Compute 32-bit FNV-1a hash of a byte span.
 * @param data Pointer to bytes.
 * @param n Number of bytes.
 * @return 32-bit hash.
 */
static uint32_t fnv1a32_bytes(const void *data, size_t n) {
  const uint8_t *p = (const uint8_t *)data;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;
  }
  return h;
}

/**
 * @brief Compute 32-bit FNV-1a hash of a C string (excluding terminator).
 * @param s NUL-terminated string (may be NULL).
 * @return 32-bit hash.
 */
static uint32_t fnv1a32_cstr(const char *s) {
  return fnv1a32_bytes(s ? s : "", s ? strlen(s) : 0u);
}

/**
 * @brief Simple xorshift PRNG (used only for the fake-token escape hatch).
 * @param x Seed/state value.
 * @return Next state value.
 */
static uint32_t xorshift32(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

/**
 * @brief Populate default engine parameters.
 * @param p Output parameter structure.
 */
static void params_set_defaults(ie_engine_params_t *p) {
  if (!p) return;

  p->precision = "fp32";
  p->sparsity = "none";
  p->affinity = "auto";
  p->pretranspose = "none";
  p->prefetch = "auto";
  p->threads = 0;
}

/**
 * @brief Merge non-empty fields from @p src into @p dst.
 * @param dst Destination parameter structure.
 * @param src Source parameter structure.
 */
static void params_merge(ie_engine_params_t *dst, const ie_engine_params_t *src) {
  if (!dst || !src) return;

  if (src->precision && src->precision[0] != '\0') dst->precision = src->precision;
  if (src->sparsity && src->sparsity[0] != '\0') dst->sparsity = src->sparsity;
  if (src->affinity && src->affinity[0] != '\0') dst->affinity = src->affinity;
  if (src->pretranspose && src->pretranspose[0] != '\0') dst->pretranspose = src->pretranspose;
  if (src->prefetch && src->prefetch[0] != '\0') dst->prefetch = src->prefetch;

  if (src->threads > 0) dst->threads = src->threads;
}

ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out_engine) {
  if (!out_engine || !device || !model_dir) return IE_ERR_BADARG;
  *out_engine = NULL;

  struct ie_engine *e = (struct ie_engine *)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_OOM;

  params_set_defaults(&e->params);
  if (p) params_merge(&e->params, p);

  (void)snprintf(e->model_dir, sizeof(e->model_dir), "%s", model_dir);

  const ie_device_kind_t kind = ie_device_kind_from_str(device);
  if (ie_device_create(kind, &e->dev) != 0 || !e->dev) {
    free(e);
    return IE_ERR_UNSUPPORTED;
  }
  (void)ie_device_caps(e->dev, &e->caps);

  /**
   * Dedup policy:
   * - We do not mutate process environment from inside the library.
   * - If IE_DEDUP=1 is requested, the weight loader will attempt to load dedup
   *   artifacts. If those are missing, ie_weights_open() should fail.
   * - Defaulting IE_DEDUP=1 is handled by the harness/Makefile.
   */
  (void)env_truthy("IE_DEDUP", 0);

  /* Load weights. Dedup handling is inside ie_weights_open()/weights_dedup.c. */
  {
    char json_path[2048];
    int n = snprintf(json_path, sizeof(json_path), "%s/model.ie.json", model_dir);
    if (n <= 0 || (size_t)n >= sizeof(json_path)) {
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_MODEL;
    }
    if (ie_weights_open(json_path, NULL, &e->weights) != 0) {
      ie_device_destroy(e->dev);
      free(e);
      return IE_ERR_MODEL;
    }
    e->weights_open = 1;
  }

  /* Try GPT-OSS tokenizer.json in the model directory. */
  e->tok = NULL;
  e->vocab_size = 0;
  {
    char tok_path[2048];
    int n = snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    if (n > 0 && (size_t)n < sizeof(tok_path)) {
      ie_tok_gptoss_t *tok = NULL;
      if (ie_tok_gptoss_open(tok_path, &tok) == 0 && tok) {
        e->tok = tok;
        e->vocab_size = ie_tok_gptoss_vocab_size(tok);
      }
    }
  }

  if (e->vocab_size == 0u) {
    e->vocab_size = 50257u;
  }

  *out_engine = (ie_engine_t *)e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *engine) {
  struct ie_engine *e = (struct ie_engine *)engine;
  if (!e) return;

  if (e->tok) {
    ie_tok_gptoss_close(e->tok);
    e->tok = NULL;
  }

  if (e->weights_open) {
    ie_weights_close(&e->weights);
    e->weights_open = 0;
  }

  if (e->dev) {
    ie_device_destroy(e->dev);
    e->dev = NULL;
  }

  free(e);
}

ie_status_t ie_engine_generate(const ie_engine_t *engine,
                               const char *prompt,
                               size_t max_new_tokens,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  const struct ie_engine *e = (const struct ie_engine *)engine;
  if (!e || !prompt || !out_tokens || !out_n_tokens) return IE_ERR_BADARG;

  *out_n_tokens = 0;
  if (max_new_tokens == 0) return IE_OK;

  /* Default behavior: refuse to generate without real inference. */
  if (!env_truthy("IE_ALLOW_FAKE_TOKENS", 0)) {
    (void)fprintf(stderr,
                  "ERROR: This snapshot does not ship a transformer forward pass.\n"
                  "ERROR: Implement ie_gptoss_infer_prefill/step to enable real text generation.\n"
                  "ERROR: For harness-only testing, set IE_ALLOW_FAKE_TOKENS=1.\n");
    return IE_ERR_UNSUPPORTED;
  }

  /* Escape hatch: deterministic token IDs for harness testing only. */
  {
    const uint32_t vocab = (e->vocab_size != 0u) ? e->vocab_size : 1u;
    uint32_t seed = fnv1a32_cstr(prompt);
    seed ^= 0x9e3779b9u;

    for (size_t i = 0; i < max_new_tokens; ++i) {
      uint32_t x = seed + (uint32_t)(i * 0x9e3779b9u);
      x = xorshift32(x);
      out_tokens[i] = (int)(x % vocab);
    }
  }

  *out_n_tokens = max_new_tokens;
  return IE_OK;
}
