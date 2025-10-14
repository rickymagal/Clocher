/**
 * @file ie_api.c
 * @brief Minimal FP32 decode path with real compute and per-token latency.
 *
 * Implements:
 *  - Engine lifecycle (create/destroy)
 *  - FP32 toy model math (procedural parameters; no external deps)
 *  - Tokenization via ie_tok_* and metadata via ie_weights_*
 *  - Per-token latency ring with p50/p95 calculation
 */

#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "ie_api.h"
#include "ie_io.h"   /* weights + tokenizer API */

/* ---------- portable time ---------- */
/**
 * @brief Get current timestamp in seconds (C11).
 *
 * @return Time in seconds as a double.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---------- PRNG ---------- */
/**
 * @brief xorshift32 PRNG step.
 * @param state Pointer to state.
 * @return Next pseudo-random 32-bit value.
 */
static inline uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

/**
 * @brief Generate a uniform float in [-scale, +scale].
 * @param state PRNG state.
 * @param scale Half range (positive).
 * @return Pseudo-random float.
 */
static float frand_uniform(uint32_t *state, float scale) {
  uint32_t r = xorshift32(state);
  float u = (float)(r / 4294967296.0f); /* [0,1) */
  return (u * 2.0f - 1.0f) * scale;
}

/* ---------- math helpers ---------- */
/**
 * @brief Set a vector to zeros.
 * @param v Pointer to vector.
 * @param n Number of elements.
 */
static void vec_clear(float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = 0.0f;
}

/**
 * @brief Add src to dst elementwise.
 * @param dst Destination vector.
 * @param src Source vector.
 * @param n   Number of elements.
 */
static void vec_add(float *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

/**
 * @brief Add bias to dst elementwise.
 * @param dst Vector to receive bias.
 * @param bias Bias vector.
 * @param n   Elements.
 */
static void vec_add_bias(float *dst, const float *bias, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] += bias[i];
}

/**
 * @brief Apply tanh to each element of a vector (in-place).
 * @param v Vector.
 * @param n Elements.
 */
static void vec_tanh(float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = tanhf(v[i]);
}

/**
 * @brief Row-major GEMV: y = W * x
 * @param W    Row-major weights of shape [rows x cols].
 * @param x    Input vector of length cols.
 * @param y    Output vector of length rows.
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
static void gemv_rowmajor(const float *W,
                          const float *x,
                          float *y,
                          size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *w = W + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      acc += w[c] * x[c];
    }
    y[r] = acc;
  }
}

/**
 * @brief Argmax over a float vector.
 * @param v Vector.
 * @param n Elements.
 * @return Index of the maximum element.
 */
static int argmax(const float *v, size_t n) {
  size_t idx = 0;
  float best = v[0];
  for (size_t i = 1; i < n; ++i) {
    if (v[i] > best) { best = v[i]; idx = i; }
  }
  return (int)idx;
}

/* ---------- percentile helpers ---------- */
/**
 * @brief Copy src to dst and sort dst ascending via insertion sort.
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param n   Elements.
 */
static void copy_and_sort(double *dst, const double *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
  for (size_t i = 1; i < n; ++i) {
    double key = dst[i];
    size_t j = i;
    while (j > 0 && dst[j-1] > key) { dst[j] = dst[j-1]; --j; }
    dst[j] = key;
  }
}

/**
 * @brief Return approximate percentile from a sorted array.
 * @param vals Sorted values.
 * @param n    Elements.
 * @param p    Percentile in [0,1].
 * @return Value at percentile p.
 */
static double percentile(const double *vals, size_t n, double p) {
  if (n == 0) return 0.0;
  size_t idx = (size_t)(p * (double)(n - 1) + 0.5);
  return vals[idx];
}

/* ---------- engine state ---------- */
typedef struct ie_engine {
  uint32_t H;     /**< Hidden size */
  uint32_t V;     /**< Logit size (simulated vocab length) */

  /* parameters */
  float *Wxh;     /**< [H x H] */
  float *Whh;     /**< [H x H] */
  float *Woh;     /**< [V x H] */
  float *bh;      /**< [H] */
  float *bo;      /**< [V] */

  /* temporaries */
  float *h;       /**< [H] hidden state */
  float *x;       /**< [H] input embedding */
  float *tmp;     /**< [H] scratch */
  float *logits;  /**< [V] logits */

  /* IO handles */
  ie_weights_t weights;
  ie_vocab_t   vocab;

  /* metrics */
  unsigned long long tokens_generated_total;
  double tok_lat_ms_ring[2048];
  size_t tok_lat_count;

  /* rng */
  uint32_t rng;
} ie_engine;

/* ---------- allocation/init ---------- */
/**
 * @brief Allocate all parameter and buffer arrays for the engine.
 * @param e Engine.
 * @return 0 on success; -1 on allocation failure.
 */
static int alloc_params(ie_engine *e) {
  size_t H = e->H, V = e->V;
  e->Wxh    = (float*)malloc(H * H * sizeof(float));
  e->Whh    = (float*)malloc(H * H * sizeof(float));
  e->Woh    = (float*)malloc(V * H * sizeof(float));
  e->bh     = (float*)malloc(H * sizeof(float));
  e->bo     = (float*)malloc(V * sizeof(float));
  e->h      = (float*)malloc(H * sizeof(float));
  e->x      = (float*)malloc(H * sizeof(float));
  e->tmp    = (float*)malloc(H * sizeof(float));
  e->logits = (float*)malloc(V * sizeof(float));
  if (!e->Wxh || !e->Whh || !e->Woh || !e->bh || !e->bo ||
      !e->h || !e->x || !e->tmp || !e->logits) {
    return -1;
  }
  return 0;
}

/**
 * @brief Free all parameter and buffer arrays.
 * @param e Engine.
 */
static void free_params(ie_engine *e) {
  free(e->Wxh); free(e->Whh); free(e->Woh);
  free(e->bh);  free(e->bo);
  free(e->h);   free(e->x);   free(e->tmp); free(e->logits);
}

/**
 * @brief Initialize parameters procedurally with a deterministic seed.
 * @param e Engine.
 */
static void init_params_procedural(ie_engine *e) {
  e->rng = 0xC0FFEEu ^ (uint32_t)(e->weights.bin_size_bytes % 104729u);
  const float scaleH = 1.0f / 32.0f;
  const float scaleV = 1.0f / 64.0f;
  for (size_t i = 0; i < (size_t)e->H * e->H; ++i) {
    e->Wxh[i] = frand_uniform(&e->rng, scaleH);
    e->Whh[i] = frand_uniform(&e->rng, scaleH);
  }
  for (size_t i = 0; i < (size_t)e->V * e->H; ++i) {
    e->Woh[i] = frand_uniform(&e->rng, scaleV);
  }
  for (size_t i = 0; i < e->H; ++i) e->bh[i] = frand_uniform(&e->rng, scaleH);
  for (size_t i = 0; i < e->V; ++i) e->bo[i] = frand_uniform(&e->rng, scaleV);
  vec_clear(e->h, e->H);
}

/**
 * @brief Produce a deterministic H-dim embedding from a token id.
 * @param tok Token id.
 * @param x   Output embedding buffer [H].
 * @param H   Hidden size.
 */
static void embed_token(uint32_t tok, float *x, size_t H) {
  uint32_t seed = 0x9E3779B9u ^ (tok * 0x85EBCA6Bu);
  for (size_t i = 0; i < H; ++i) {
    seed ^= (seed << 13); seed ^= (seed >> 17); seed ^= (seed << 5);
    float t = (float)(seed & 0xFFFFu) / 65536.0f; /* [0,1) */
    x[i] = (t * 2.0f - 1.0f) + 0.1f * sinf((float)(i + (tok % 31)) * 0.07f);
  }
}

/**
 * @brief One decode step: updates hidden state and returns next token id.
 * @param e          Engine.
 * @param prev_token Previous token id (used for embedding).
 * @return Next token id in the range [1000, 1000+65535].
 */
static uint32_t decode_step(ie_engine *e, uint32_t prev_token) {
  embed_token(prev_token, e->x, e->H);
  gemv_rowmajor(e->Wxh, e->x, e->tmp, e->H, e->H);
  float *h_new = e->h;
  gemv_rowmajor(e->Whh, e->h, h_new, e->H, e->H);
  vec_add(h_new, e->tmp, e->H);
  vec_add_bias(h_new, e->bh, e->H);
  vec_tanh(h_new, e->H);
  gemv_rowmajor(e->Woh, e->h, e->logits, e->V, e->H);
  vec_add_bias(e->logits, e->bo, e->V);
  int idx = argmax(e->logits, e->V);
  uint32_t next_id = (uint32_t)(1000 + (idx % 65536));
  return next_id;
}

/* ---------- public API ---------- */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return IE_ERR_INVALID_ARGUMENT;

  ie_engine *e = (ie_engine*)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_INTERNAL;

  e->H = 256;
  e->V = 1024;

  const char *json = p ? p->shape_json_path : NULL;
  const char *bin  = p ? p->weights_path    : NULL;
  if (ie_weights_open(json ? json : "models/gpt-oss-20b/model.ie.json",
                      bin  ? bin  : "models/gpt-oss-20b/model.ie.bin",
                      &e->weights) != 0) {
    memset(&e->weights, 0, sizeof(e->weights));
  }
  (void)ie_vocab_load(p && p->vocab_path ? p->vocab_path : "models/gpt-oss-20b/vocab.json", &e->vocab);

  if (alloc_params(e) != 0) { free(e); return IE_ERR_INTERNAL; }
  init_params_procedural(e);

  e->tokens_generated_total = 0;
  e->tok_lat_count = 0;

  *out = (ie_engine_t*)e;
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  ie_engine *e = (ie_engine*)h;
  ie_vocab_free(&e->vocab);
  ie_weights_close(&e->weights);
  free_params(e);
  free(e);
}

ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_tokens || !out_count) return IE_ERR_INVALID_ARGUMENT;
  ie_engine *e = (ie_engine*)h;

  uint32_t needed = 0;
  if (ie_tok_encode(&e->vocab, prompt ? prompt : "", NULL, &needed) != 0) return IE_ERR_INVALID_ARGUMENT;

  uint32_t *prompt_ids = NULL;
  if (needed > 0) {
    prompt_ids = (uint32_t*)malloc(sizeof(uint32_t) * needed);
    if (!prompt_ids) return IE_ERR_INTERNAL;
    uint32_t got = 0;
    if (ie_tok_encode(&e->vocab, prompt, prompt_ids, &got) != 0 || got != needed) {
      free(prompt_ids); return IE_ERR_INTERNAL;
    }
    for (uint32_t i = 0; i < got; ++i) { (void)decode_step(e, prompt_ids[i]); }
  }

  uint32_t n = max_new_tokens;
  if (n == 0) { if (prompt_ids) free(prompt_ids); *out_count = 0; return IE_OK; }

  uint32_t prev = (needed > 0) ? prompt_ids[needed - 1] : 1000u;

  for (uint32_t t = 0; t < n; ++t) {
    double t0 = now_s();
    uint32_t next_id = decode_step(e, prev);
    double t1 = now_s();

    out_tokens[t] = next_id;
    prev = next_id;

    double tok_ms = (t1 - t0) * 1000.0;
    if (e->tok_lat_count < sizeof(e->tok_lat_ms_ring)/sizeof(e->tok_lat_ms_ring[0])) {
      e->tok_lat_ms_ring[e->tok_lat_count++] = tok_ms;
    }
  }

  e->tokens_generated_total += n;

  if (prompt_ids) free(prompt_ids);
  *out_count = n;
  return IE_OK;
}

ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return IE_ERR_INVALID_ARGUMENT;
  const ie_engine *e = (const ie_engine*)h;

  double sorted[2048];
  size_t n = e->tok_lat_count;
  if (n > 0) {
    copy_and_sort(sorted, e->tok_lat_ms_ring, n);
    out->latency_p50_ms = percentile(sorted, n, 0.50);
    out->latency_p95_ms = percentile(sorted, n, 0.95);
  } else {
    out->latency_p50_ms = 0.0;
    out->latency_p95_ms = 0.0;
  }

  out->tps_true = (out->latency_p50_ms > 0.0) ? (1000.0 / out->latency_p50_ms) : 0.0;
  out->rss_peak_mb = 0;
  out->kv_hits     = 0;
  out->kv_misses   = e->tokens_generated_total;
  return IE_OK;
}
