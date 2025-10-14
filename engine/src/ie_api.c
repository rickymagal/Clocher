/**
 * @file ie_api.c
 * @brief FP32 decode path with runtime kernel dispatch, fused ops, low-precision
 *        round-trip (BF16/FP16 -> FP32 accumulate), and optional pretranspose.
 *
 * All code and comments are in English and Doxygen-ready.
 */

#define _POSIX_C_SOURCE 200112L  /* posix_memalign on POSIX */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

#include "ie_api.h"
#include "ie_io.h"
#include "ie_cpu.h"
#include "ie_threadpool.h"
#include "ie_kernels.h"
#include "ie_math.h"
#include "ie_layout.h"
#include "ie_floatx.h"

/* -------------------------------------------------------------------------- */
/* Portable aligned allocation                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Allocate size bytes with at least @p alignment alignment.
 *
 * Tries posix_memalign (POSIX), then C11 aligned_alloc, then falls back to
 * malloc (which may be unaligned). The pointer must be freed with free().
 *
 * @param alignment Alignment in bytes (power of two), e.g., 64.
 * @param size      Number of bytes to allocate.
 * @return Allocated pointer or NULL on failure.
 */
static void* ie_aligned_malloc(size_t alignment, size_t size) {
#if defined(_POSIX_VERSION)
  void *p = NULL;
  if (posix_memalign(&p, alignment, size) == 0) return p;
  return NULL;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  /* aligned_alloc requires size to be multiple of alignment */
  size_t rounded = (size + alignment - 1) / alignment * alignment;
  return aligned_alloc(alignment, rounded);
#else
  /* Fallback (may be unaligned on some platforms) */
  (void)alignment;
  return malloc(size);
#endif
}

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return current timestamp in seconds using C11 timespec_get.
 * @return Wall-clock seconds as double.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief xorshift32 PRNG step.
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
 * @param state PRNG state.
 * @param scale Half-range of uniform distribution.
 * @return Random float in [-scale, +scale].
 */
static float frand_uniform(uint32_t *state, float scale) {
  uint32_t r = xorshift32(state);
  float u = (float)(r / 4294967296.0f);
  return (u * 2.0f - 1.0f) * scale;
}

/**
 * @brief Argmax index of a float vector.
 * @param v Input vector.
 * @param n Number of elements.
 * @return Index of maximum value (0 <= idx < n).
 */
static int argmax(const float *v, size_t n) {
  size_t idx = 0; float best = v[0];
  for (size_t i = 1; i < n; ++i) { if (v[i] > best) { best = v[i]; idx = i; } }
  return (int)idx;
}

/* -------------------------------------------------------------------------- */
/* Engine definition                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Opaque engine structure (private to this translation unit).
 */
typedef struct ie_engine {
  /* model sizes */
  uint32_t H;                    /**< Hidden size. */
  uint32_t V;                    /**< Logit/vocab size (simulated). */

  /* parameter tensors (row-major originals) */
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

  /* threading / precision / hints */
  ie_threadpool_t *tp;           /**< Thread pool (NULL = single-thread). */
  int fast_tanh;                 /**< Non-zero => use fast tanh approximation. */
  size_t grainsize_hint;         /**< Informational grainsize (unused by partition). */

  /* pretranspose (blocked-K) descriptors */
  ie_wblocked_desc_t Wxh_blk;    /**< Blocked-K copy of Wxh (optional). */
  ie_wblocked_desc_t Woh_blk;    /**< Blocked-K copy of Woh (optional). */
  int use_wxh_blk;               /**< Non-zero to use Wxh_blk. */
  int use_woh_blk;               /**< Non-zero to use Woh_blk. */

  /* low-precision round-trip scratch */
  float    *x_q32, *h_q32;       /**< FP32 after round-trip from 16-bit. */
  uint16_t *x_b16, *h_b16;       /**< Temporary 16-bit storage (bf16/fp16). */
  int       lowp_mode;           /**< 0=fp32, 1=bf16, 2=fp16 */
} ie_engine;

/* -------------------------------------------------------------------------- */
/* Aligned allocation and init                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Allocate engine parameters and workspace with 64B alignment.
 * @param e Engine pointer.
 * @return 0 on success; -1 on allocation failure.
 */
static int alloc_params(ie_engine *e) {
  size_t H = e->H, V = e->V;

  e->Wxh_rm = (float*)ie_aligned_malloc(64, H*H*sizeof(float));
  e->Whh_rm = (float*)ie_aligned_malloc(64, H*H*sizeof(float));
  e->Woh_rm = (float*)ie_aligned_malloc(64, V*H*sizeof(float));
  e->bh     = (float*)ie_aligned_malloc(64, H*sizeof(float));
  e->bo     = (float*)ie_aligned_malloc(64, V*sizeof(float));
  e->h      = (float*)ie_aligned_malloc(64, H*sizeof(float));
  e->x      = (float*)ie_aligned_malloc(64, H*sizeof(float));
  e->tmp    = (float*)ie_aligned_malloc(64, H*sizeof(float));
  e->logits = (float*)ie_aligned_malloc(64, V*sizeof(float));

  e->x_q32  = (float*)ie_aligned_malloc(64, H*sizeof(float));
  e->h_q32  = (float*)ie_aligned_malloc(64, H*sizeof(float));

  e->x_b16 = (uint16_t*)malloc(H * sizeof(uint16_t));
  e->h_b16 = (uint16_t*)malloc(H * sizeof(uint16_t));

  return (!e->Wxh_rm||!e->Whh_rm||!e->Woh_rm||
          !e->bh||!e->bo||!e->h||!e->x||!e->tmp||!e->logits||
          !e->x_q32||!e->h_q32||!e->x_b16||!e->h_b16) ? -1 : 0;
}

/**
 * @brief Free engine parameters and workspace.
 * @param e Engine pointer (must be valid).
 */
static void free_params(ie_engine *e){
  free(e->Wxh_rm); free(e->Whh_rm); free(e->Woh_rm);
  free(e->bh);  free(e->bo);
  free(e->h);   free(e->x);   free(e->tmp); free(e->logits);
  free(e->x_q32); free(e->h_q32);
  free(e->x_b16); free(e->h_b16);
  ie_layout_free(&e->Wxh_blk);
  ie_layout_free(&e->Woh_blk);
}

/**
 * @brief Initialize parameters procedurally for a deterministic baseline.
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

/* -------------------------------------------------------------------------- */
/* Token embedding + fused ops                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Produce a deterministic H-dim embedding from a token id.
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
 * @brief Round-trip a FP32 vector through BF16/FP16 to emulate low-precision matmul.
 * @param in      FP32 input vector.
 * @param out32   FP32 output vector (quantize->dequant result).
 * @param n       Number of elements.
 * @param mode    0=no-op, 1=bf16, 2=fp16.
 * @param tmp16   Temporary 16-bit buffer of length n.
 */
static void maybe_roundtrip_lowp(const float *in, float *out32,
                                 size_t n, int mode,
                                 uint16_t *tmp16) {
  if (mode == 1) { /* BF16 */
    ie_fp32_to_bf16(in, tmp16, n);
    ie_bf16_to_fp32(tmp16, out32, n);
  } else if (mode == 2) { /* FP16 */
    ie_fp32_to_fp16(in, tmp16, n);
    ie_fp16_to_fp32(tmp16, out32, n);
  } else {
    for (size_t i = 0; i < n; ++i) out32[i] = in[i];
  }
}

/* -------------------------------------------------------------------------- */
/* Decode step                                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Execute a single decode step and return the next token id.
 *
 * tmp = Wxh * x_rt; h = Whh * h_rt; h = tanh(h + tmp + bh); logits = Woh * h_rt + bo
 *
 * @param e           Engine pointer.
 * @param prev_token  Previous token id (for embedding).
 * @return Next token id.
 */
static uint32_t decode_step(ie_engine *e, uint32_t prev_token) {
  embed_token(prev_token, e->x, e->H);

  /* tmp = Wxh * (x round-tripped to requested precision) */
  maybe_roundtrip_lowp(e->x, e->x_q32, e->H, e->lowp_mode, e->x_b16);
  if (e->use_wxh_blk) {
    ie_gemv_f32(e->Wxh_blk.data, e->x_q32, e->tmp,
                e->Wxh_blk.rows, e->Wxh_blk.cols, /*bias*/NULL, e->Wxh_blk.blk_k);
  } else {
    ie_gemv_f32(e->Wxh_rm, e->x_q32, e->tmp, e->H, e->H, /*bias*/NULL, /*blk_k*/0);
  }

  /* h = Whh * (h round-tripped) */
  maybe_roundtrip_lowp(e->h, e->h_q32, e->H, e->lowp_mode, e->h_b16);
  ie_gemv_f32(e->Whh_rm, e->h_q32, e->h, e->H, e->H, /*bias*/NULL, /*blk_k*/0);

  /* fused: h = tanh( h + tmp + bh ) */
  fuse_bias_tanh(e->h, e->tmp, e->bh, e->H, e->fast_tanh);

  /* logits = Woh * (h round-tripped) + bo */
  maybe_roundtrip_lowp(e->h, e->h_q32, e->H, e->lowp_mode, e->h_b16);
  if (e->use_woh_blk) {
    ie_gemv_f32(e->Woh_blk.data, e->h_q32, e->logits,
                e->Woh_blk.rows, e->Woh_blk.cols, e->bo, e->Woh_blk.blk_k);
  } else {
    ie_gemv_f32(e->Woh_rm, e->h_q32, e->logits, e->V, e->H, e->bo, /*blk_k*/0);
  }

  int idx = argmax(e->logits, e->V);
  return (uint32_t)(1000 + (idx & 0xFFFF));
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

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
  e->lowp_mode = 0;
  if (p && p->precision) {
    if (strcmp(p->precision,"bf16")==0) e->lowp_mode = 1;
    else if (strcmp(p->precision,"fp16")==0) e->lowp_mode = 2;
  }
  e->grainsize_hint = (p ? (size_t)p->grainsize : 0u);

  const char *pt = (p && p->pretranspose) ? p->pretranspose : "none";
  const int want_wxh = (!strcmp(pt,"wxh")||!strcmp(pt,"all"));
  const int want_woh = (!strcmp(pt,"woh")||!strcmp(pt,"all"));
  const size_t BK = 128;

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

void ie_engine_destroy(ie_engine_t *h) {
  if (!h) return;
  ie_engine *e = (ie_engine*)h;
  ie_tp_destroy(e->tp);
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

ie_status_t ie_engine_metrics(const ie_engine_t *h, ie_metrics_t *out) {
  if (!h || !out) return IE_ERR_INVALID_ARGUMENT;
  const ie_engine *e = (const ie_engine*)h;

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
