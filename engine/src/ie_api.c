/**
 * @file ie_api.c
 * @brief Minimal FP32 decode path with runtime kernel dispatch and a pthread pool.
 *
 * Responsibilities:
 *  - Engine lifecycle (create/destroy).
 *  - Tokenization & prompt prefill.
 *  - Decode loop: embed → GEMV (Wxh/Whh/Woh) → bias → tanh → logits → argmax.
 *  - Per-token latency ring (p50/p95) and metrics snapshot.
 *
 * This file intentionally keeps the math simple and deterministic so we can
 * build a reproducible CPU baseline and measure improvements from step 4+.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "ie_api.h"
#include "ie_io.h"
#include "ie_cpu.h"
#include "ie_threadpool.h"
#include "ie_kernels.h"
#include "ie_math.h"

/**
 * @brief Return current timestamp in seconds using C11 timespec_get.
 *
 * @return Wall-clock seconds as double.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief xorshift32 pseudo-random number generator step.
 *
 * @param state Pointer to internal PRNG state (updated in place).
 * @return Next 32-bit pseudo-random value.
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
 * @brief Generate a uniform random float in [-scale, +scale].
 *
 * @param state PRNG state.
 * @param scale Half-range of the uniform distribution (must be > 0).
 * @return Random float in [-scale, +scale].
 */
static float frand_uniform(uint32_t *state, float scale) {
  uint32_t r = xorshift32(state);
  float u = (float)(r / 4294967296.0f);
  return (u * 2.0f - 1.0f) * scale;
}

/**
 * @brief Zero a float vector (set all elements to 0.0f).
 *
 * @param v Vector pointer (may be NULL if @p n == 0).
 * @param n Number of elements to set.
 */
static void vec_clear(float *v, size_t n) {
  for (size_t i = 0; i < n; ++i) v[i] = 0.0f;
}

/**
 * @brief Add a bias vector elementwise (y[i] += b[i]).
 *
 * @param y Destination vector to be incremented.
 * @param b Bias vector.
 * @param n Number of elements.
 */
static void vec_add_bias(float *y, const float *b, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += b[i];
}

/**
 * @brief Argmax over a float vector.
 *
 * @param v Input vector.
 * @param n Number of elements.
 * @return Index of the largest element (first max on ties).
 */
static int argmax(const float *v, size_t n) {
  size_t idx = 0;
  float best = v[0];
  for (size_t i = 1; i < n; ++i) {
    if (v[i] > best) { best = v[i]; idx = i; }
  }
  return (int)idx;
}

/**
 * @brief Copy values into @p dst and sort ascending (insertion sort).
 *
 * @param dst Destination buffer (size >= n).
 * @param src Source buffer with n values.
 * @param n   Number of elements.
 */
static void copy_and_sort(double *dst, const double *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
  for (size_t i = 1; i < n; ++i) {
    double key = dst[i];
    size_t j = i;
    while (j > 0 && dst[j - 1] > key) { dst[j] = dst[j - 1]; --j; }
    dst[j] = key;
  }
}

/**
 * @brief Return approximate percentile from a sorted array.
 *
 * @param vals Sorted values (ascending).
 * @param n    Number of elements.
 * @param p    Percentile in [0,1], e.g., 0.50 for p50.
 * @return Value at the requested percentile (nearest index).
 */
static double percentile(const double *vals, size_t n, double p) {
  if (n == 0) return 0.0;
  size_t idx = (size_t)(p * (double)(n - 1) + 0.5);
  return vals[idx];
}

/**
 * @brief Opaque engine state (internal fields).
 */
typedef struct ie_engine {
  /* model sizes */
  uint32_t H;             /**< Hidden size. */
  uint32_t V;             /**< Logit/vocab size (simulated). */

  /* parameters */
  float *Wxh, *Whh, *Woh; /**< Weight matrices. */
  float *bh, *bo;         /**< Bias vectors. */

  /* temporaries */
  float *h, *x, *tmp, *logits;

  /* IO handles */
  ie_weights_t weights;
  ie_vocab_t   vocab;

  /* metrics */
  unsigned long long tokens_generated_total;
  double tok_lat_ms_ring[4096];
  size_t tok_lat_count;

  /* rng */
  uint32_t rng;

  /* optimizations (step 4) */
  ie_threadpool_t *tp;    /**< Thread pool (NULL = single-thread). */
  int fast_tanh;          /**< Non-zero to enable fast tanh approximation. */
} ie_engine;

/**
 * @brief Allocate all parameter and workspace buffers.
 *
 * @param e Engine pointer.
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
  return (!e->Wxh || !e->Whh || !e->Woh || !e->bh || !e->bo ||
          !e->h || !e->x || !e->tmp || !e->logits) ? -1 : 0;
}

/**
 * @brief Free all parameter and workspace buffers.
 *
 * @param e Engine pointer (must be valid).
 */
static void free_params(ie_engine *e) {
  free(e->Wxh); free(e->Whh); free(e->Woh);
  free(e->bh);  free(e->bo);
  free(e->h);   free(e->x);   free(e->tmp); free(e->logits);
}

/**
 * @brief Initialize parameters procedurally with a deterministic seed.
 *
 * We avoid model file dependencies by synthesizing weights from a fixed seed
 * (influenced by the binary size just to vary across datasets).
 *
 * @param e Engine pointer.
 */
static void init_params_procedural(ie_engine *e) {
  e->rng = 0xC0FFEEu ^ (uint32_t)(e->weights.bin_size_bytes % 104729u);
  const float sH = 1.0f / 32.0f;
  const float sV = 1.0f / 64.0f;

  for (size_t i = 0; i < (size_t)e->H * e->H; ++i) {
    e->Wxh[i] = frand_uniform(&e->rng, sH);
    e->Whh[i] = frand_uniform(&e->rng, sH);
  }
  for (size_t i = 0; i < (size_t)e->V * e->H; ++i) {
    e->Woh[i] = frand_uniform(&e->rng, sV);
  }
  for (size_t i = 0; i < e->H; ++i) e->bh[i] = frand_uniform(&e->rng, sH);
  for (size_t i = 0; i < e->V; ++i) e->bo[i] = frand_uniform(&e->rng, sV);

  vec_clear(e->h, e->H);
}

/**
 * @brief Produce a deterministic H-dimensional embedding from a token id.
 *
 * @param tok Token id.
 * @param x   Output buffer of length H.
 * @param H   Hidden size.
 */
static void embed_token(uint32_t tok, float *x, size_t H) {
  uint32_t s = 0x9E3779B9u ^ (tok * 0x85EBCA6Bu);
  for (size_t i = 0; i < H; ++i) {
    s ^= (s << 13); s ^= (s >> 17); s ^= (s << 5);
    float t = (float)(s & 0xFFFFu) / 65536.0f;
    x[i] = (t * 2.0f - 1.0f) + 0.1f * sinf((float)(i + (tok % 31)) * 0.07f);
  }
}

/**
 * @brief GEMV shard context for row splitting.
 */
typedef struct {
  const float *W; /**< Base of weight matrix (row-major). */
  const float *x; /**< Input vector. */
  float *y;       /**< Output vector (one value per row). */
  size_t rows;    /**< Number of rows to process. */
  size_t cols;    /**< Number of columns per row. */
} gemv_ctx_t;

/**
 * @brief Range task that calls GEMV on a contiguous row block.
 *
 * @param ctx   Pointer to #gemv_ctx_t.
 * @param begin First row index (inclusive).
 * @param end   One past last row index (exclusive).
 */
static void shard_gemv(void *ctx, size_t begin, size_t end) {
  gemv_ctx_t *c = (gemv_ctx_t*)ctx;
  const float *Wrow = c->W + begin * c->cols;
  ie_gemv_f32(Wrow, c->x, c->y + begin, end - begin, c->cols);
}

/**
 * @brief Parallel GEMV over rows using the thread pool.
 *
 * @param tp    Thread pool (NULL = single-thread).
 * @param W     Row-major weights.
 * @param x     Input vector.
 * @param y     Output vector.
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 */
static void gemv_parallel(ie_threadpool_t *tp,
                          const float *W, const float *x, float *y,
                          size_t rows, size_t cols) {
  gemv_ctx_t c = { W, x, y, rows, cols };
  ie_tp_parallel_for(tp, rows, shard_gemv, &c, 0);
}

/**
 * @brief Execute a single decode step and return the next token id.
 *
 * The step performs:
 *   tmp = Wxh * x(prev)        (GEMV)
 *   h   = Whh * h + tmp + bh   (GEMV + bias)
 *   h   = tanh(h)              (vector nonlinearity)
 *   logits = Woh * h + bo      (GEMV + bias)
 *   next = argmax(logits)
 *
 * @param e           Engine pointer.
 * @param prev_token  Previous token id (for embedding).
 * @return Next token id in [1000, 1000+65535].
 */
static uint32_t decode_step(ie_engine *e, uint32_t prev_token) {
  embed_token(prev_token, e->x, e->H);

  /* tmp = Wxh * x */
  gemv_parallel(e->tp, e->Wxh, e->x, e->tmp, e->H, e->H);

  /* h = Whh * h + tmp + bh; tanh */
  gemv_parallel(e->tp, e->Whh, e->h, e->h, e->H, e->H);
  for (size_t i = 0; i < e->H; ++i) e->h[i] += e->tmp[i] + e->bh[i];
  ie_vec_tanh_f32(e->h, e->H, e->fast_tanh);

  /* logits = Woh * h + bo */
  gemv_parallel(e->tp, e->Woh, e->h, e->logits, e->V, e->H);
  vec_add_bias(e->logits, e->bo, e->V);

  int idx = argmax(e->logits, e->V);
  return (uint32_t)(1000 + (idx & 0xFFFF));
}

/**
 * @brief Create an inference engine instance.
 *
 * Behavior:
 *  - Detects CPU features and installs best kernels (AVX2 if available).
 *  - By default uses **single-thread** execution (stable for tests/CI).
 *    To enable threading, set `params->threads > 1`.
 *
 * @param p    Optional parameters (paths, precision, threads, affinity).
 * @param out  Output handle to the created engine.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return IE_ERR_INVALID_ARGUMENT;

  ie_engine *e = (ie_engine*)calloc(1, sizeof(*e));
  if (!e) return IE_ERR_INTERNAL;

  /* Default sizes for the baseline model shape. */
  e->H = 256;
  e->V = 1024;

  /* Load (relaxed) metadata and vocab (both optional in baseline). */
  const char *json = p ? p->shape_json_path : NULL;
  const char *bin  = p ? p->weights_path    : NULL;
  if (ie_weights_open(json ? json : "models/gpt-oss-20b/model.ie.json",
                      bin  ? bin  : "models/gpt-oss-20b/model.ie.bin",
                      &e->weights) != 0) {
    memset(&e->weights, 0, sizeof(e->weights));
  }
  (void)ie_vocab_load(p && p->vocab_path ? p->vocab_path : "models/gpt-oss-20b/vocab.json", &e->vocab);

  /* Allocate and initialize parameters. */
  if (alloc_params(e) != 0) { free(e); return IE_ERR_INTERNAL; }
  init_params_procedural(e);

  /* Install best kernels based on runtime features. */
  ie_cpu_features_t feat; ie_cpu_detect(&feat);
  const int want_avx2 = 1;
  ie_kernels_install((want_avx2 && feat.avx2) ? 1 : 0);

  /* Create thread pool:
     - Default: single-thread (n=1) to maximize stability in CI/tests.
     - If caller requests threads>1, we honor it. */
  unsigned nth = (p && p->threads > 0) ? p->threads : 1u;
  const char *aff = (p && p->affinity) ? p->affinity : "auto";
  e->tp = (nth > 1u) ? ie_tp_create(nth, aff) : NULL;

  /* Toggle fast tanh for reduced precision modes (kept 0 for pure fp32). */
  e->fast_tanh = (p && p->precision &&
                  (strcmp(p->precision, "bf16") == 0 || strcmp(p->precision, "fp16") == 0))
                  ? 1 : 0;

  /* Metrics init. */
  e->tokens_generated_total = 0;
  e->tok_lat_count = 0;

  *out = (ie_engine_t*)e;
  return IE_OK;
}

/**
 * @brief Destroy an inference engine instance and release resources.
 *
 * @param h Engine handle (NULL allowed; no-op).
 */
void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  ie_engine *e = (ie_engine*)h;
  ie_tp_destroy(e->tp);
  ie_vocab_free(&e->vocab);
  ie_weights_close(&e->weights);
  free_params(e);
  free(e);
}

/**
 * @brief Generate up to @p max_new_tokens tokens from a text @p prompt.
 *
 * The function records per-token latencies into a ring (up to 4096 tokens)
 * for later percentile computation in ::ie_engine_metrics.
 *
 * @param h               Engine handle.
 * @param prompt          UTF-8 input string (may be NULL/empty).
 * @param max_new_tokens  Number of tokens to generate.
 * @param out_tokens      Output buffer (length >= max_new_tokens). May be NULL
 *                        iff @p max_new_tokens == 0.
 * @param out_count       Number of tokens actually generated.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_count) return IE_ERR_INVALID_ARGUMENT;
  /* Allow out_tokens == NULL iff max_new_tokens == 0 */
  if (max_new_tokens > 0 && !out_tokens) return IE_ERR_INVALID_ARGUMENT;

  ie_engine *e = (ie_engine*)h;

  /* Prefill from prompt. */
  uint32_t needed = 0;
  if (ie_tok_encode(&e->vocab, prompt ? prompt : "", NULL, &needed) != 0)
    return IE_ERR_INVALID_ARGUMENT;

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

  /* Decode loop. */
  const uint32_t n = max_new_tokens;
  if (n == 0) {
    if (prompt_ids) free(prompt_ids);
    *out_count = 0;
    return IE_OK;
  }

  uint32_t prev = (needed > 0) ? prompt_ids[needed - 1] : 1000u;
  for (uint32_t t = 0; t < n; ++t) {
    const double t0 = now_s();
    const uint32_t next_id = decode_step(e, prev);
    const double t1 = now_s();

    out_tokens[t] = next_id;
    prev = next_id;

    const double tok_ms = (t1 - t0) * 1000.0;
    if (e->tok_lat_count < sizeof(e->tok_lat_ms_ring) / sizeof(e->tok_lat_ms_ring[0])) {
      e->tok_lat_ms_ring[e->tok_lat_count++] = tok_ms;
    }
  }
  e->tokens_generated_total += n;

  if (prompt_ids) free(prompt_ids);
  *out_count = n;
  return IE_OK;
}

/**
 * @brief Snapshot performance metrics accumulated by the engine.
 *
 * Computes p50/p95 from the per-token latency ring. `tps_true` is computed
 * as 1000 / p50_ms, serving as a conservative throughput proxy.
 *
 * @param h   Engine handle.
 * @param out Output metrics structure to fill.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return IE_ERR_INVALID_ARGUMENT;
  const ie_engine *e = (const ie_engine*)h;

  double sorted[4096];
  const size_t n = e->tok_lat_count;
  if (n > 0) {
    copy_and_sort(sorted, e->tok_lat_ms_ring, n);
    out->latency_p50_ms = percentile(sorted, n, 0.50);
    out->latency_p95_ms = percentile(sorted, n, 0.95);
  } else {
    out->latency_p50_ms = 0.0;
    out->latency_p95_ms = 0.0;
  }

  out->tps_true    = (out->latency_p50_ms > 0.0) ? (1000.0 / out->latency_p50_ms) : 0.0;
  out->rss_peak_mb = 0;
  out->kv_hits     = 0;
  out->kv_misses   = e->tokens_generated_total;
  return IE_OK;
}
