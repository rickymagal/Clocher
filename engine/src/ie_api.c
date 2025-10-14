/**
 * @file ie_api.c
 * @brief FP32 decode path with runtime kernel dispatch, fused ops, and pretranspose.
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
#include "ie_layout.h"

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
 * @brief xorshift32 PRNG step.
 *
 * @param state Pointer to PRNG state (updated).
 * @return Next 32-bit pseudo-random value.
 */
static inline uint32_t xorshift32(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *state = x;
  return x;
}

/**
 * @brief Generate uniform float in [-scale, +scale].
 *
 * @param state PRNG state.
 * @param scale Half-range of uniform distribution.
 * @return Random float.
 */
static float frand_uniform(uint32_t *state, float scale) {
  uint32_t r = xorshift32(state);
  float u = (float)(r / 4294967296.0f);
  return (u * 2.0f - 1.0f) * scale;
}

/**
 * @brief Argmax index of a float vector.
 *
 * @param v Input vector.
 * @param n Number of elements.
 * @return Index of maximum value.
 */
static int argmax(const float *v, size_t n) {
  size_t idx = 0; float best = v[0];
  for (size_t i = 1; i < n; ++i) { if (v[i] > best) { best = v[i]; idx = i; } }
  return (int)idx;
}

/** @brief Internal engine state. */
typedef struct ie_engine {
  /* model sizes */
  uint32_t H;                    /**< Hidden size. */
  uint32_t V;                    /**< Logit/vocab size (simulated). */

  /* parameters (row-major originals) */
  float *Wxh_rm, *Whh_rm, *Woh_rm; /**< Weight matrices row-major. */
  float *bh, *bo;                   /**< Bias vectors. */

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

  /* optimizations */
  ie_threadpool_t *tp;           /**< Thread pool (NULL = single-thread). */
  int fast_tanh;                 /**< Non-zero => use fast tanh approximation. */
  size_t grainsize_hint;         /**< Informational grainsize (unused by partition). */

  /* pretranspose (blocked-K) */
  ie_wblocked_desc_t Wxh_blk;    /**< Blocked-K copy of Wxh (optional). */
  ie_wblocked_desc_t Woh_blk;    /**< Blocked-K copy of Woh (optional). */
  int use_wxh_blk;               /**< Non-zero to use Wxh_blk. */
  int use_woh_blk;               /**< Non-zero to use Woh_blk. */
} ie_engine;

/**
 * @brief Allocate engine parameters and workspace.
 *
 * @param e Engine pointer.
 * @return 0 on success; -1 on allocation failure.
 */
static int alloc_params(ie_engine *e) {
  size_t H = e->H, V = e->V;
  e->Wxh_rm=(float*)malloc(H*H*sizeof(float));
  e->Whh_rm=(float*)malloc(H*H*sizeof(float));
  e->Woh_rm=(float*)malloc(V*H*sizeof(float));
  e->bh =(float*)malloc(H*sizeof(float));
  e->bo =(float*)malloc(V*sizeof(float));
  e->h  =(float*)malloc(H*sizeof(float));
  e->x  =(float*)malloc(H*sizeof(float));
  e->tmp=(float*)malloc(H*sizeof(float));
  e->logits=(float*)malloc(V*sizeof(float));
  return (!e->Wxh_rm||!e->Whh_rm||!e->Woh_rm||!e->bh||!e->bo||!e->h||!e->x||!e->tmp||!e->logits)?-1:0;
}

/**
 * @brief Free engine parameters and workspace.
 *
 * @param e Engine pointer (must be valid).
 */
static void free_params(ie_engine *e){
  free(e->Wxh_rm); free(e->Whh_rm); free(e->Woh_rm);
  free(e->bh);  free(e->bo);
  free(e->h);   free(e->x);   free(e->tmp); free(e->logits);
  ie_layout_free(&e->Wxh_blk);
  ie_layout_free(&e->Woh_blk);
}

/**
 * @brief Initialize parameters procedurally for a deterministic baseline.
 *
 * @param e Engine pointer.
 */
static void init_params_procedural(ie_engine *e) {
  e->rng = 0xC0FFEEu ^ (uint32_t)(e->weights.bin_size_bytes % 104729u);
  const float sH = 1.0f/32.0f, sV = 1.0f/64.0f;
  for (size_t i=0;i<(size_t)e->H*e->H;++i){ e->Wxh_rm[i]=frand_uniform(&e->rng,sH); e->Whh_rm[i]=frand_uniform(&e->rng,sH); }
  for (size_t i=0;i<(size_t)e->V*e->H;++i){ e->Woh_rm[i]=frand_uniform(&e->rng,sV); }
  for (size_t i=0;i<e->H;++i) e->bh[i]=frand_uniform(&e->rng,sH);
  for (size_t i=0;i<e->V;++i) e->bo[i]=frand_uniform(&e->rng,sV);
  for (size_t i=0;i<e->H;++i) e->h[i]=0.0f;
}

/**
 * @brief Produce a deterministic H-dim embedding from a token id.
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
    x[i] = (t*2.0f - 1.0f) + 0.1f * sinf((float)(i + (tok % 31)) * 0.07f);
  }
}

/**
 * @brief Fused bias + tanh with optional clamp for fast-tanh path.
 *
 * Computes, for i in [0,H):
 *   s = h[i] + tmp[i] + bh[i];
 *   if (fast) s = clamp(s, -3, 3);
 *   h[i] = fast ? ie_fast_tanhf(s) : tanhf(s);
 *
 * @param h         Hidden vector (updated in place).
 * @param tmp       Temporary vector added into h.
 * @param bias      Bias vector.
 * @param H         Length of all vectors.
 * @param fast_tanh Non-zero to enable clamp + fast tanh approximation.
 */
static void fuse_bias_tanh(float *h, const float *tmp, const float *bias,
                           size_t H, int fast_tanh) {
  if (fast_tanh) {
    for (size_t i = 0; i < H; ++i) {
      float s = h[i] + tmp[i] + bias[i];
      if (s > 3.0f) s = 3.0f;
      if (s < -3.0f) s = -3.0f;
      h[i] = ie_fast_tanhf(s);
    }
  } else {
    for (size_t i = 0; i < H; ++i) {
      float s = h[i] + tmp[i] + bias[i];
      h[i] = tanhf(s);
    }
  }
}

/**
 * @brief Execute a single decode step and return the next token id.
 *
 * tmp = Wxh * x                                  (GEMV)
 * h   = Whh * h                                  (GEMV)
 * h   = tanh(h + tmp + bh)                       (fused bias+tanh; optional clamp)
 * logits = Woh * h + bo                          (GEMV with epilogue bias)
 * next = argmax(logits)
 *
 * @param e           Engine pointer.
 * @param prev_token  Previous token id (for embedding).
 * @return Next token id.
 */
static uint32_t decode_step(ie_engine *e, uint32_t prev_token) {
  embed_token(prev_token, e->x, e->H);

  /* tmp = Wxh * x (optionally use blocked) */
  if (e->use_wxh_blk) {
    ie_gemv_f32(e->Wxh_blk.data, e->x, e->tmp, e->Wxh_blk.rows, e->Wxh_blk.cols, /*bias*/NULL, e->Wxh_blk.blk_k);
  } else {
    ie_gemv_f32(e->Wxh_rm, e->x, e->tmp, e->H, e->H, /*bias*/NULL, /*blk_k*/0);
  }

  /* h = Whh * h (no blocked path for Whh; keep simple) */
  ie_gemv_f32(e->Whh_rm, e->h, e->h, e->H, e->H, /*bias*/NULL, /*blk_k*/0);

  /* fused: h = tanh( h + tmp + bh ) with optional clamp for fast tanh */
  fuse_bias_tanh(e->h, e->tmp, e->bh, e->H, e->fast_tanh);

  /* logits = Woh * h + bo (use epilogue bias and optional blocked) */
  if (e->use_woh_blk) {
    ie_gemv_f32(e->Woh_blk.data, e->h, e->logits, e->Woh_blk.rows, e->Woh_blk.cols, e->bo, e->Woh_blk.blk_k);
  } else {
    ie_gemv_f32(e->Woh_rm, e->h, e->logits, e->V, e->H, e->bo, /*blk_k*/0);
  }

  int idx = argmax(e->logits, e->V);
  return (uint32_t)(1000 + (idx & 0xFFFF));
}

/**
 * @brief Create an inference engine instance.
 *
 * @param p    Optional parameters (may be NULL for defaults).
 * @param out  Output engine handle (must not be NULL).
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_create(const ie_engine_params_t *p, ie_engine_t **out) {
  if (!out) return IE_ERR_INVALID_ARGUMENT;
  ie_engine *e = (ie_engine*)calloc(1,sizeof(*e)); if(!e) return IE_ERR_INTERNAL;

  e->H=256; e->V=1024;

  const char *json = p ? p->shape_json_path : NULL;
  const char *bin  = p ? p->weights_path    : NULL;
  if (ie_weights_open(json ? json : "models/gpt-oss-20b/model.ie.json",
                      bin  ? bin  : "models/gpt-oss-20b/model.ie.bin",
                      &e->weights) != 0) { memset(&e->weights,0,sizeof(e->weights)); }
  (void)ie_vocab_load(p && p->vocab_path ? p->vocab_path : "models/gpt-oss-20b/vocab.json", &e->vocab);

  if (alloc_params(e)!=0){ free(e); return IE_ERR_INTERNAL; }
  init_params_procedural(e);

  ie_cpu_features_t feat; ie_cpu_detect(&feat);
  const int want_avx2 = 1;
  ie_kernels_install((want_avx2 && feat.avx2) ? 1 : 0);

  unsigned nth = (p && p->threads > 0) ? p->threads : 1u;
  const char *aff = (p && p->affinity) ? p->affinity : "auto";
  e->tp = (nth > 1u) ? ie_tp_create(nth, aff) : NULL;

  e->fast_tanh = (p && p->precision &&
                  (strcmp(p->precision,"bf16")==0 || strcmp(p->precision,"fp16")==0)) ? 1 : 0;
  e->grainsize_hint = (p ? (size_t)p->grainsize : 0u);

  /* Pretranspose policy */
  const char *pt = (p && p->pretranspose) ? p->pretranspose : "none";
  const int want_wxh = (!strcmp(pt,"wxh")||!strcmp(pt,"all"));
  const int want_woh = (!strcmp(pt,"woh")||!strcmp(pt,"all"));
  const size_t BK = 128; /* simple column-block size */

  if (want_wxh) {
    char cache[4096];
    ie_layout_cache_name(bin, "wxh", e->H, e->H, BK, cache, sizeof(cache));
    if (ie_layout_build_blockedK(e->Wxh_rm, e->H, e->H, BK, cache, &e->Wxh_blk) == 0) {
      e->use_wxh_blk = 1;
    }
  }
  if (want_woh) {
    char cache[4096];
    ie_layout_cache_name(bin, "woh", e->V, e->H, BK, cache, sizeof(cache));
    if (ie_layout_build_blockedK(e->Woh_rm, e->V, e->H, BK, cache, &e->Woh_blk) == 0) {
      e->use_woh_blk = 1;
    }
  }

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
 * @brief Generate tokens from a prompt.
 *
 * @param h               Engine handle.
 * @param prompt          UTF-8 string (may be NULL/empty).
 * @param max_new_tokens  Number of tokens to generate.
 * @param out_tokens      Output buffer (NULL iff max_new_tokens==0).
 * @param out_count       Receives number of tokens generated.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_generate(ie_engine_t *h,
                               const char *prompt,
                               uint32_t max_new_tokens,
                               uint32_t *out_tokens,
                               uint32_t *out_count) {
  if (!h || !out_count) return IE_ERR_INVALID_ARGUMENT;
  if (max_new_tokens > 0 && !out_tokens) return IE_ERR_INVALID_ARGUMENT;

  ie_engine *e = (ie_engine*)h;

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
    if (e->tok_lat_count < sizeof(e->tok_lat_ms_ring)/sizeof(e->tok_lat_ms_ring[0])) {
      e->tok_lat_ms_ring[e->tok_lat_count++] = tok_ms;
    }
  }
  e->tokens_generated_total += n;

  if (prompt_ids) free(prompt_ids);
  *out_count = n;
  return IE_OK;
}

/**
 * @brief Compute a metrics snapshot (p50/p95, tps_true, kv stats).
 *
 * @param h   Engine handle.
 * @param out Output metrics pointer to fill.
 * @return IE_OK on success; error code otherwise.
 */
ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return IE_ERR_INVALID_ARGUMENT;
  const ie_engine *e = (const ie_engine*)h;

  /* simple selection sort copy to compute percentiles */
  double tmp[4096];
  const size_t n = e->tok_lat_count;
  if (n > 0) {
    for (size_t i=0;i<n;++i) tmp[i]=e->tok_lat_ms_ring[i];
    for (size_t i=1;i<n;++i){ double key=tmp[i]; size_t j=i; while(j>0 && tmp[j-1]>key){tmp[j]=tmp[j-1];--j;} tmp[j]=key; }
    const size_t p50 = (size_t)(0.50 * (double)(n - 1) + 0.5);
    const size_t p95 = (size_t)(0.95 * (double)(n - 1) + 0.5);
    out->latency_p50_ms = tmp[p50];
    out->latency_p95_ms = tmp[p95];
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
