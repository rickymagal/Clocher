/* ============================================================================
 * File: engine/src/ie_api.c
 * ============================================================================
 */
/**
 * @file ie_api.c
 * @brief Stable public API implementation for creating, destroying, and running an engine instance.
 *
 * @details
 * This module implements the functions declared in ie_api.h. It keeps the public
 * surface small and stable while delegating model-specific logic to internal subsystems:
 *  - Device creation/teardown (CPU/CUDA selection)
 *  - Tokenizer open/encode/decode (GPT-OSS tokenizer JSON)
 *  - Weights open/close (model.ie.json + model.ie.bin)
 *  - Hyperparameter loading (base + extended HF config)
 *  - GPT-OSS runtime creation and token-by-token inference
 *  - Sampling to select the next token
 *
 * Logging policy:
 *  - Stage-level INFO logs during create/generate to quickly locate failures.
 *  - ERROR logs with rc/status codes and key parameters on failure paths.
 *  - Per-token logs are opt-in via environment variables.
 *
 * Environment variables (optional):
 *  - IE_API_LOG_TOKENS:         If set to 1, log prompt token ids (truncated).
 *  - IE_API_LOG_EVERY:          If set to N>0, log every N decode steps (token id).
 *  - IE_API_DEBUG_DECODE:       If set to 1, attempt to decode generated tokens to UTF-8 and log (best-effort).
 *  - IE_API_DEBUG_DECODE_EVERY: If set to N>0, decode/log every N decode steps when IE_API_DEBUG_DECODE=1.
 *  - IE_DEBUG_TOPK:             If set to K>0, dump top-K logits decoded (prefill and optionally during decode).
 *  - IE_DEBUG_TOPK_EVERY:       If set to N>0, dump top-K logits every N decode steps (requires IE_DEBUG_TOPK>0).
 *
 * Header layout constraint:
 *  - The include directory is flat. All includes must reference headers by filename only.
 */

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#ifndef _XOPEN_SOURCE
#  define _XOPEN_SOURCE 700
#endif

#include "ie_api.h"

#include <ctype.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <unistd.h>

#include "gptoss_hparams.h"
#include "ie_device.h"
#include "ie_infer.h"
#include "ie_io.h"
#include "ie_kv_cache.h"
#include "ie_kv_instrumentation.h"
#include "ie_sampling.h"
#include "ie_tokenizer_gptoss.h"
#include "tensor_map.h"
#include "util_logging.h"
#include "ie_cpu.h"

/**
 * @brief Get the selected Q4 GEMV kernel name for logging.
 *
 * @details
 * The Q4_0 GEMV implementation is runtime-dispatched. This helper is used to
 * print a deterministic "expected" backend name based on CPU feature flags.
 *
 * @return A short backend name string.
 */
static const char *ie_q4_kernel_selected_name(void) {
    ie_cpu_features_t feat;
    ie_cpu_features_detect(&feat);

    if (feat.avx2 && feat.fma) {
        return "q4_avx2_fma";
    }
    return "q4_generic";
}

/* ============================================================================
 * Threading helpers
 * ========================================================================== */

/**
 * @brief Check whether an environment variable is set to a non-empty value.
 *
 * @param key Environment variable name.
 * @return 1 if set and non-empty, 0 otherwise.
 */
static int ie_api_env_is_set_(const char *key) {
  const char *v = getenv(key);
  return (v && v[0] != '\0') ? 1 : 0;
}

/**
 * @brief Set a process environment variable to an integer value.
 *
 * @details
 * This helper exists because some kernel modules use environment variables
 * (e.g. IE_THREADS / IE_BF16_THREADS) to pick a thread count lazily on first use.
 *
 * @param key Environment variable name.
 * @param value Integer value to store.
 * @param overwrite Non-zero to overwrite existing values.
 * @return 0 on success, non-zero on failure.
 */
static int ie_api_setenv_int_(const char *key, int value, int overwrite) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%d", value);
#if defined(_WIN32)
  (void)overwrite;
  return _putenv_s(key, buf);
#else
  return setenv(key, buf, overwrite ? 1 : 0);
#endif
}

/**
 * @brief Choose a default thread count for "auto" behavior.
 *
 * @details
 * The default is derived from the number of online CPUs. It is clamped to a
 * conservative maximum to prevent accidental oversubscription on large hosts.
 *
 * @return A thread count in [1, 256].
 */
static int ie_api_default_threads_(void) {
  long n = 0;
#if defined(_SC_NPROCESSORS_ONLN)
  n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  if (n < 1) n = 1;
  if (n > 256) n = 256;
  return (int)n;
}

/**
 * @brief Apply thread-count parameters to the environment for kernel selection.
 *
 * @details
 * Some kernels (notably BF16 GEMV) currently read IE_BF16_THREADS and/or
 * IE_THREADS once on first invocation to decide whether to use the internal
 * thread pool. This function bridges the public API thread parameter to those
 * kernels without requiring call sites to set environment variables manually.
 *
 * Semantics:
 *  - If p->threads > 0, IE_THREADS and IE_BF16_THREADS are set/overwritten.
 *  - If p->threads == 0 (auto) and neither variable is set, defaults are set.
 *  - If p is NULL and neither variable is set, defaults are set.
 *
 * @param p Optional engine parameters.
 */
static void ie_api_apply_thread_env_(const ie_engine_params_t *p) {
  const int req = p ? p->threads : 0;
  if (req > 0) {
    int t = req;
    if (t < 1) t = 1;
    if (t > 256) t = 256;
    (void)ie_api_setenv_int_("IE_THREADS", t, 1);
    (void)ie_api_setenv_int_("IE_BF16_THREADS", t, 1);
    return;
  }

  if (ie_api_env_is_set_("IE_BF16_THREADS") || ie_api_env_is_set_("IE_THREADS")) {
    return;
  }

  const int t = ie_api_default_threads_();
  (void)ie_api_setenv_int_("IE_THREADS", t, 0);
  (void)ie_api_setenv_int_("IE_BF16_THREADS", t, 0);
}

/* ============================================================================
 * Internal engine object
 * ========================================================================== */

struct ie_engine {
  ie_device_t *dev;
  ie_weights_t w;
  ie_tok_gptoss_t *tok;

  /* Base hparams (shapes/sizes) + extended config (eps/theta/rope scaling). */
  ie_gptoss_hparams_t hp;
  gptoss_hparams_ex_t hpx;

  ie_gptoss_infer_t *infer;

  float *logits;
  size_t logits_len;

  uint32_t *scratch_idx;
  float *scratch_prob;
  size_t scratch_cap;

  ie_sample_cfg_t sample_cfg;
  ie_rng_t rng;
};

/* ============================================================================
 * Time helpers
 * ========================================================================== */

static double now_s_(void) {
  struct timespec ts;
#if defined(CLOCK_MONOTONIC)
  const clockid_t clk = CLOCK_MONOTONIC;
#else
  const clockid_t clk = CLOCK_REALTIME;
#endif
  if (clock_gettime(clk, &ts) != 0) return 0.0;
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* ============================================================================
 * Small helpers
 * ========================================================================== */

static void join_path_(char *dst, size_t cap, const char *a, const char *b) {
  if (!dst || cap == 0) return;

  size_t i = 0;
  if (a) {
    for (; a[i] && i + 1 < cap; ++i) dst[i] = a[i];
  }

  const int need_slash = (i > 0 && dst[i - 1] != '/');
  if (need_slash && i + 1 < cap) dst[i++] = '/';

  if (b) {
    size_t j = 0;
    for (; b[j] && i + 1 < cap; ++j, ++i) dst[i] = b[j];
  }

  dst[i < cap ? i : (cap - 1)] = '\0';
}

static int env_flag_(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 ||
      strcmp(v, "no") == 0 || strcmp(v, "NO") == 0 || strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0) {
    return 0;
  }
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 ||
      strcmp(v, "yes") == 0 || strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 || strcmp(v, "ON") == 0) {
    return 1;
  }
  return 1;
}

static int env_int_(const char *name, int default_value, int min_value, int max_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (!end || end == v) return default_value;
  if (x < (long)min_value) x = (long)min_value;
  if (x > (long)max_value) x = (long)max_value;
  return (int)x;
}

static unsigned long long env_u64_(const char *name, unsigned long long default_value,
                                  unsigned long long min_value, unsigned long long max_value)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == 0) {
        return default_value;
    }
    char *end = NULL;
    unsigned long long x = strtoull(v, &end, 10);
    if (end == v) {
        return default_value;
    }
    if (x < min_value) {
        return min_value;
    }
    if (x > max_value) {
        return max_value;
    }
    return x;
}

static int path_exists_(const char *path)
{
  struct stat st;
  if (path == NULL || path[0] == 0) return 0;
  return (stat(path, &st) == 0);
}

static const char *status_str_(ie_status_t st) {
  switch (st) {
    case IE_OK: return "IE_OK";
    case IE_ERR_BADARG: return "IE_ERR_BADARG";
    case IE_ERR_OOM: return "IE_ERR_OOM";
    case IE_ERR_MODEL: return "IE_ERR_MODEL";
    case IE_ERR_UNSUPPORTED: return "IE_ERR_UNSUPPORTED";
    case IE_ERR_INTERNAL: return "IE_ERR_INTERNAL";
    default: return "IE_STATUS_UNKNOWN";
  }
}

static const char *sample_kind_str_(ie_sample_kind_t k)
{
    switch (k) {
        case IE_SAMPLE_GREEDY: return "greedy";
        case IE_SAMPLE_TOPK:   return "topk";
        case IE_SAMPLE_TOPP:   return "topp";
        default:               return "unknown";
    }
}


static int streq_icase_(const char *a, const char *b)
{
    /* ASCII case-insensitive string equality. */
    if (a == NULL || b == NULL) {
        return 0;
    }
    for (;;) {
        unsigned char ca = (unsigned char)*a++;
        unsigned char cb = (unsigned char)*b++;
        ca = (unsigned char)tolower((int)ca);
        cb = (unsigned char)tolower((int)cb);
        if (ca != cb) {
            return 0;
        }
        if (ca == 0) {
            return 1;
        }
    }
}

static void log_hparams_(const ie_gptoss_hparams_t *hp) {
  if (!hp) return;
  ie_log_info(
      "hparams: layers=%" PRIu32 " d_model=%" PRIu32 " heads=%" PRIu32 " kv_heads=%" PRIu32
      " head_dim=%" PRIu32 " vocab=%" PRIu32 " max_seq=%" PRIu32,
      hp->n_layers,
      hp->d_model,
      hp->n_heads,
      hp->n_kv_heads,
      hp->d_head,
      hp->vocab_size,
      hp->max_seq);
}

static void log_hparams_ex_(const gptoss_hparams_ex_t *hpx) {
  if (!hpx) return;

  /* This struct intentionally avoids assuming a "present" boolean; treat empty type as "absent". */
  ie_log_info("hparams_ex: rms_norm_eps=%.10g rope_theta=%.10g rope_scaling_type=%s rope_scaling_factor=%.10g",
              (double)hpx->rms_norm_eps,
              (double)hpx->rope_theta,
              hpx->rope_scaling_type,
              (double)hpx->rope_scaling_factor);
}

static void maybe_correct_head_dim_(const char *model_dir, ie_gptoss_hparams_t *hp) {
  if (!model_dir || !hp) return;
  if (hp->n_heads == 0) return;

  char path[4096];
  join_path_(path, sizeof(path), model_dir, "tensor_map.json");

  tensor_map_t map;
  if (tensor_map_load(path, &map) != 0) return;

  const tensor_desc_t *q = tensor_map_find(&map, "model.layers.0.self_attn.q_proj.weight");
  if (q && q->ndim >= 2 && q->shape) {
    const uint32_t n_heads = hp->n_heads;
    const uint32_t q0 = q->shape[0];
    const uint32_t q1 = q->shape[1];

    uint32_t new_d_head = hp->d_head;

    if (q0 != 0 && (q0 % n_heads) == 0 && q0 != hp->d_model) {
      new_d_head = q0 / n_heads;
    } else if (q1 != 0 && (q1 % n_heads) == 0 && q1 != hp->d_model) {
      new_d_head = q1 / n_heads;
    }

    if (new_d_head != 0 && new_d_head != hp->d_head) {
      ie_log_warn("correcting head_dim from %" PRIu32 " to %" PRIu32 " using %s (shape=[%" PRIu32 ",%" PRIu32 "])",
                  hp->d_head,
                  new_d_head,
                  "model.layers.0.self_attn.q_proj.weight",
                  q0,
                  q1);
      hp->d_head = new_d_head;
    }
  }

  tensor_map_free(&map);
}

static ie_status_t status_from_rc_(int rc) {
  if (rc == 0) return IE_OK;
  if (rc == -12 || rc == -5) return IE_ERR_OOM;
  return IE_ERR_INTERNAL;
}

static void log_prompt_preview_(const char *prompt) {
  if (!prompt) {
    ie_log_info("prompt: (null)");
    return;
  }

  char tmp[128];
  size_t n = 0;
  for (; prompt[n] && n + 1 < sizeof(tmp) && n < 96; ++n) {
    char ch = prompt[n];
    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    tmp[n] = ch;
  }
  tmp[n] = '\0';

  ie_log_info("prompt: bytes=%zu preview=\"%s%s\"", strlen(prompt), tmp, (prompt[n] ? "..." : ""));
}

static pthread_mutex_t trace_ids_mu_ = PTHREAD_MUTEX_INITIALIZER;
static FILE *trace_ids_fp_ = NULL;
static int trace_ids_init_ = 0;

static FILE *trace_ids_open_(void) {
  pthread_mutex_lock(&trace_ids_mu_);
  if (trace_ids_init_) {
    FILE *fp = trace_ids_fp_;
    pthread_mutex_unlock(&trace_ids_mu_);
    return fp;
  }

  trace_ids_init_ = 1;
  const char *path = getenv("IE_TRACE_IDS_JSONL");
  if (!path || path[0] == '\0') {
    trace_ids_fp_ = NULL;
    pthread_mutex_unlock(&trace_ids_mu_);
    return NULL;
  }

  trace_ids_fp_ = fopen(path, "a");
  if (!trace_ids_fp_) {
    ie_log_warn("ie_trace_ids: failed to open '%s'", path);
  } else {
    (void)setvbuf(trace_ids_fp_, NULL, _IOLBF, 0);
  }

  FILE *fp = trace_ids_fp_;
  pthread_mutex_unlock(&trace_ids_mu_);
  return fp;
}

static int trace_ids_prompt_index_(size_t *out) {
  if (!out) return 0;
  const char *s = getenv("IE_TRACE_PROMPT_INDEX");
  if (!s || s[0] == '\0') return 0;
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (end == s || (end && *end)) return 0;
  *out = (size_t)v;
  return 1;
}

static void json_write_escaped_(FILE *fp, const char *s) {
  fputc('"', fp);
  if (s) {
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
      const unsigned char ch = *p;
      switch (ch) {
        case '"': fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
          if (ch < 0x20) {
            fprintf(fp, "\\u%04x", (unsigned)ch);
          } else {
            fputc((int)ch, fp);
          }
          break;
      }
    }
  }
  fputc('"', fp);
}

static void trace_ids_prompt_(size_t prompt_index,
                              const char *prompt,
                              const uint32_t *prompt_ids,
                              uint32_t n_prompt) {
  FILE *fp = trace_ids_open_();
  if (!fp) return;

  pthread_mutex_lock(&trace_ids_mu_);
  fprintf(fp, "{\"kind\":\"prompt\",\"prompt_index\":%zu,", prompt_index);
  fputs("\"prompt\":", fp);
  json_write_escaped_(fp, prompt ? prompt : "");
  fputs(",\"prompt_token_ids\":[", fp);
  for (uint32_t i = 0; i < n_prompt; ++i) {
    if (i) fputc(',', fp);
    fprintf(fp, "%u", (unsigned)prompt_ids[i]);
  }
  fputs("]}\n", fp);
  pthread_mutex_unlock(&trace_ids_mu_);
}

static void trace_ids_step_(size_t prompt_index, size_t step, uint32_t next_id) {
  FILE *fp = trace_ids_open_();
  if (!fp) return;

  pthread_mutex_lock(&trace_ids_mu_);
  fprintf(fp,
          "{\"kind\":\"gen_step\",\"prompt_index\":%zu,\"step\":%zu,\"next_id\":%u}\n",
          prompt_index,
          step,
          (unsigned)next_id);
  pthread_mutex_unlock(&trace_ids_mu_);
}


static int is_precision_int4_(const ie_engine_params_t *p) {
  if (!p || !p->precision) return 0;
  /* CLI uses "int4". Keep a few aliases to avoid brittle call sites. */
  return (strcmp(p->precision, "int4") == 0) ||
         (strcmp(p->precision, "q4") == 0) ||
         (strcmp(p->precision, "q4_0") == 0) ||
         (strcmp(p->precision, "q4_0_f32") == 0);
}

/* ============================================================================
 * Top-k decoded logits debug ("truth serum")
 * ========================================================================== */

static void decode_piece_preview_(const ie_tok_gptoss_t *tok, uint32_t token_id, char *dst, size_t cap) {
  if (!dst || cap == 0) return;
  dst[0] = '\0';
  if (!tok) return;

  size_t need = 0;
  const int rc0 = ie_tok_gptoss_decode(tok, &token_id, 1u, NULL, &need);
  if (rc0 != 0 || need == 0 || need > (1u << 16)) {
    (void)snprintf(dst, cap, "<decode_err>");
    return;
  }

  char *tmp = (char *)malloc(need);
  if (!tmp) {
    (void)snprintf(dst, cap, "<oom>");
    return;
  }

  size_t inout = need;
  const int rc1 = ie_tok_gptoss_decode(tok, &token_id, 1u, tmp, &inout);
  if (rc1 != 0) {
    free(tmp);
    (void)snprintf(dst, cap, "<decode_err>");
    return;
  }

  tmp[need - 1] = '\0';

  size_t j = 0;
  for (size_t i = 0; tmp[i] && j + 1 < cap; ++i) {
    unsigned char c = (unsigned char)tmp[i];
    if (c < 0x20 || c == 0x7f) c = ' ';
    dst[j++] = (char)c;
  }
  dst[j] = '\0';

  free(tmp);
}

static void debug_dump_topk_(const ie_tok_gptoss_t *tok,
                             const float *logits,
                             size_t logits_len,
                             int k,
                             const char *tag) {
  if (!tok || !logits || logits_len == 0 || k <= 0 || !tag) return;
  if (k > 64) k = 64;
  if ((size_t)k > logits_len) k = (int)logits_len;

  uint32_t top_id[64];
  float top_val[64];
  for (int i = 0; i < k; ++i) {
    top_id[i] = 0;
    top_val[i] = -INFINITY;
  }

  for (size_t i = 0; i < logits_len; ++i) {
    const float v = logits[i];
    if (v <= top_val[k - 1]) continue;

    int pos = k - 1;
    while (pos > 0 && v > top_val[pos - 1]) {
      top_val[pos] = top_val[pos - 1];
      top_id[pos] = top_id[pos - 1];
      --pos;
    }
    top_val[pos] = v;
    top_id[pos] = (uint32_t)i;
  }

  ie_log_info("debug_topk(%s): k=%d", tag, k);
  for (int i = 0; i < k; ++i) {
    char piece[160];
    decode_piece_preview_(tok, top_id[i], piece, sizeof(piece));
    ie_log_info("debug_topk(%s): rank=%d id=%" PRIu32 " logit=%.9g piece=\"%s\"",
                tag, i, top_id[i], (double)top_val[i], piece);
  }
}

/* ============================================================================
 * Public API
 * ========================================================================== */

ie_status_t ie_engine_create(const ie_engine_params_t *p,
                             const char *device,
                             const char *model_dir,
                             ie_engine_t **out) {
  if (!model_dir || !out) {
    ie_log_error("ie_engine_create: bad args (model_dir=%p out=%p)", (const void *)model_dir, (void *)out);
    return IE_ERR_BADARG;
  }
  *out = NULL;

  const char *dev_str = device ? device : "cpu";
  ie_log_info("ie_engine_create: begin (device=%s model_dir=%s)", dev_str, model_dir);

  if (p) {
    ie_log_info("ie_engine_create: params precision=%s sparsity=%s affinity=%s pretranspose=%s prefetch=%s threads=%d",
                (p->precision ? p->precision : "(null)"),
                (p->sparsity ? p->sparsity : "(null)"),
                (p->affinity ? p->affinity : "(null)"),
                (p->pretranspose ? p->pretranspose : "(null)"),
                (p->prefetch ? p->prefetch : "(null)"),
                p->threads);

    if (is_precision_int4_(p)) {
      ie_log_info("ie_engine_create: q4 kernel=%s (runtime-dispatched)", ie_q4_kernel_selected_name());
    }
  }

  /* Ensure kernel thread settings are applied before any lazy kernel init. */
  ie_api_apply_thread_env_(p);

  {
    const char *t0 = getenv("IE_THREADS");
    const char *t1 = getenv("IE_BF16_THREADS");
    if ((t0 && *t0) || (t1 && *t1)) {
      ie_log_info("ie_engine_create: threads env IE_THREADS=%s IE_BF16_THREADS=%s",
                  (t0 && *t0) ? t0 : "(unset)",
                  (t1 && *t1) ? t1 : "(unset)");
    }
  }

  ie_engine_t *e = (ie_engine_t *)calloc(1, sizeof(*e));
  if (!e) {
    ie_log_error("ie_engine_create: OOM allocating engine struct");
    return IE_ERR_OOM;
  }

  ie_log_info("ie_engine_create: device_create (device=%s)", dev_str);
  const ie_device_kind_t kind = ie_device_kind_from_str(dev_str);
  const int dev_rc = ie_device_create(kind, &e->dev);
  if (dev_rc != 0 || !e->dev) {
    ie_log_error("ie_engine_create: device_create failed (rc=%d device=%s)", dev_rc, dev_str);
    free(e);
    return IE_ERR_UNSUPPORTED;
  }

  const char *tok_path = NULL;
  char tok_path_buf[4096];
  if (p && p->tokenizer_path && p->tokenizer_path[0] != '\0') {
    tok_path = p->tokenizer_path;
  } else {
    join_path_(tok_path_buf, sizeof(tok_path_buf), model_dir, "tokenizer.json");
    tok_path = tok_path_buf;
  }
  ie_log_info("ie_engine_create: tokenizer_open (path=%s)", tok_path);

  const int tok_rc = ie_tok_gptoss_open(tok_path, &e->tok);
  if (tok_rc != 0 || !e->tok) {
    ie_log_error("ie_engine_create: tokenizer_open failed (rc=%d path=%s)", tok_rc, tok_path);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  const uint32_t tok_vocab = ie_tok_gptoss_vocab_size(e->tok);
  ie_log_info("ie_engine_create: tokenizer_open ok (vocab=%" PRIu32 ")", tok_vocab);

  const char *weights_json = NULL;
  const char *bin_override = "";

  char weights_json_buf[PATH_MAX];
  char compat_json_buf[PATH_MAX];
  char q4_bin_buf[PATH_MAX];

  const int want_int4 = (p && is_precision_int4_(p));

  if (p && p->weights_json_path && p->weights_json_path[0] != '\0') {
    weights_json = p->weights_json_path;
  } else if (want_int4) {
    join_path_(compat_json_buf, sizeof(compat_json_buf), model_dir, "model.ie.compat.json");
    if (path_exists_(compat_json_buf)) {
      weights_json = compat_json_buf;
    }
  }

  if (weights_json == NULL) {
    join_path_(weights_json_buf, sizeof(weights_json_buf), model_dir, "model.ie.json");
    weights_json = weights_json_buf;
  }

  if (p && p->weights_bin_path && p->weights_bin_path[0] != '\0') {
    bin_override = p->weights_bin_path;
  } else if (want_int4) {
    join_path_(q4_bin_buf, sizeof(q4_bin_buf), model_dir, "model.q4.bin");
    if (path_exists_(q4_bin_buf)) {
      bin_override = q4_bin_buf;
    }
  }

  ie_log_info("ie_engine_create: weights_open (json=%s bin_override=%s)", weights_json,
              (bin_override[0] != '\0' ? bin_override : "(null)"));
  const int w_rc = ie_weights_open(weights_json, bin_override, &e->w);
  if (w_rc != 0) {
    ie_log_error("ie_engine_create: weights_open failed (rc=%d json=%s)", w_rc, weights_json);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  ie_log_info("ie_engine_create: weights_open ok (dtype=%s bin=%s bin_size_bytes=%zu dedup=%d)",
              e->w.dtype,
              e->w.weights_path,
              e->w.bin_size_bytes,
              e->w.is_dedup ? 1 : 0);

  /* Load base hparams first (shapes/sizes). */
  ie_log_info("ie_engine_create: hparams_load (dir=%s)", model_dir);
  const int hp_rc = gptoss_hparams_load(model_dir, &e->hp);
  if (hp_rc != 0) {
    ie_log_error("ie_engine_create: hparams_load failed (rc=%d dir=%s)", hp_rc, model_dir);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  /* Load extended config (eps/theta/rope scaling). */
  ie_log_info("ie_engine_create: hparams_load_ex (dir=%s)", model_dir);
  const int ex_rc = gptoss_hparams_load_ex(model_dir, &e->hpx);
  if (ex_rc != 0) {
    ie_log_warn("ie_engine_create: hparams_load_ex failed (rc=%d dir=%s). Continuing with base hparams only.", ex_rc, model_dir);
    memset(&e->hpx, 0, sizeof(e->hpx));
  }

  log_hparams_(&e->hp);
  if (ex_rc == 0) log_hparams_ex_(&e->hpx);

  if (tok_vocab != 0 && e->hp.vocab_size != 0 && tok_vocab != e->hp.vocab_size) {
    ie_log_warn("ie_engine_create: tokenizer vocab (%" PRIu32 ") != hparams vocab (%" PRIu32 ")",
                tok_vocab,
                e->hp.vocab_size);
  }

  maybe_correct_head_dim_(model_dir, &e->hp);

  ie_log_info("ie_engine_create: infer_create");
  const int inf_rc = ie_gptoss_infer_create(e->dev, &e->w, &e->hp, &e->infer);
  if (inf_rc != 0 || !e->infer) {
    ie_log_error("ie_engine_create: infer_create failed (rc=%d)", inf_rc);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_INTERNAL;
  }

  e->logits_len = (size_t)e->hp.vocab_size;
  if (e->logits_len == 0) {
    ie_log_error("ie_engine_create: invalid vocab_size=0; cannot allocate logits");
    ie_gptoss_infer_destroy(e->infer);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_MODEL;
  }

  const size_t bytes_logits = e->logits_len * sizeof(float);
  const size_t bytes_idx = e->logits_len * sizeof(uint32_t);
  const size_t bytes_prob = e->logits_len * sizeof(float);

  ie_log_info("ie_engine_create: allocating scratch (logits=%zu bytes idx=%zu bytes prob=%zu bytes)",
              bytes_logits,
              bytes_idx,
              bytes_prob);

  e->logits = (float *)malloc(bytes_logits);
  e->scratch_idx = (uint32_t *)malloc(bytes_idx);
  e->scratch_prob = (float *)malloc(bytes_prob);
  e->scratch_cap = e->logits_len;

  if (!e->logits || !e->scratch_idx || !e->scratch_prob) {
    ie_log_error("ie_engine_create: OOM allocating scratch (logits=%p idx=%p prob=%p)",
                 (void *)e->logits,
                 (void *)e->scratch_idx,
                 (void *)e->scratch_prob);
    free(e->scratch_prob);
    free(e->scratch_idx);
    free(e->logits);
    ie_gptoss_infer_destroy(e->infer);
    ie_weights_close(&e->w);
    ie_tok_gptoss_close(e->tok);
    ie_device_destroy(e->dev);
    free(e);
    return IE_ERR_OOM;
  }

  /*
   * Sampling configuration
   *
   * The benchmarking harness measures token-id generation. For deterministic
   * verification runs, the CLI can set IE_GREEDY=1 (via --greedy) or you can
   * set IE_SAMPLE_KIND=greedy to force greedy decoding.
   */

  ie_sample_cfg_t sample_cfg;
  sample_cfg.kind = IE_SAMPLE_TOPP;
  sample_cfg.temperature = 0.8000f;
  sample_cfg.top_p = 0.9500f;
  sample_cfg.top_k = 0u;
  sample_cfg.disallow_token0 = 1u;

  const char *kind_env = getenv("IE_SAMPLE_KIND");
  if (kind_env && kind_env[0]) {
    if (streq_icase_(kind_env, "greedy")) {
      sample_cfg.kind = IE_SAMPLE_GREEDY;
    } else if (streq_icase_(kind_env, "topk")) {
      sample_cfg.kind = IE_SAMPLE_TOPK;
    } else if (streq_icase_(kind_env, "topp")) {
      sample_cfg.kind = IE_SAMPLE_TOPP;
    } else {
      ie_log_warn("ie_engine_create: unrecognized IE_SAMPLE_KIND=\"%s\"; using defaults", kind_env);
    }
  }

  if (env_flag_("IE_GREEDY", 0)) {
    sample_cfg.kind = IE_SAMPLE_GREEDY;
  }

  if (sample_cfg.kind == IE_SAMPLE_GREEDY) {
    /* Greedy ignores temperature/top-p/top-k; keep them in a neutral state. */
    sample_cfg.temperature = 1.0000f;
    sample_cfg.top_p = 1.0000f;
    sample_cfg.top_k = 0u;
  }

  ie_log_info("ie_engine_create: sampling config (kind=%d/%s temp=%.4f top_p=%.4f top_k=%u disallow_token0=%u)",
              (int)sample_cfg.kind,
              sample_kind_str_(sample_cfg.kind),
              (double)sample_cfg.temperature,
              (double)sample_cfg.top_p,
              (unsigned)sample_cfg.top_k,
              (unsigned)sample_cfg.disallow_token0);

  e->sample_cfg = sample_cfg;

  unsigned long long seed = env_u64_("IE_SEED", 1ull, 0ull, ~0ull);
  if (seed == 0ull) seed = 1ull;
  ie_rng_init(&e->rng, (uint64_t)seed);
  ie_log_info("ie_engine_create: RNG seeded (seed=%llu)", seed);

  *out = e;
  ie_log_info("ie_engine_create: success (engine=%p)", (void *)e);
  return IE_OK;
}

void ie_engine_destroy(ie_engine_t *e) {
  if (!e) return;

  ie_log_info("ie_engine_destroy: begin (engine=%p)", (void *)e);

  free(e->scratch_prob);
  free(e->scratch_idx);
  free(e->logits);

  if (e->infer) {
    ie_log_info("ie_engine_destroy: infer_destroy");
    ie_gptoss_infer_destroy(e->infer);
  }

  ie_log_info("ie_engine_destroy: weights_close");
  ie_weights_close(&e->w);

  if (e->tok) {
    ie_log_info("ie_engine_destroy: tokenizer_close");
    ie_tok_gptoss_close(e->tok);
  }
  if (e->dev) {
    ie_log_info("ie_engine_destroy: device_destroy");
    ie_device_destroy(e->dev);
  }

  free(e);
  ie_log_info("ie_engine_destroy: done");
}

ie_status_t ie_engine_generate(const ie_engine_t *e,
                               const char *prompt,
                               size_t max_new,
                               int *out_tokens,
                               size_t *out_n_tokens) {
  return ie_engine_generate_ex(e, prompt, max_new, out_tokens, out_n_tokens, NULL);
}

ie_status_t ie_engine_generate_ex(const ie_engine_t *e,
                                  const char *prompt,
                                  size_t max_new,
                                  int *out_tokens,
                                  size_t *out_n_tokens,
                                  ie_generate_stats_t *out_stats) {
  if (!e || !prompt || !out_n_tokens) {
    ie_log_error("ie_engine_generate_ex: bad args (e=%p prompt=%p out_n_tokens=%p)",
                 (const void *)e,
                 (const void *)prompt,
                 (void *)out_n_tokens);
    return IE_ERR_BADARG;
  }
  if (max_new > 0 && !out_tokens) {
    ie_log_error("ie_engine_generate_ex: bad args (max_new=%zu out_tokens=NULL)", max_new);
    return IE_ERR_BADARG;
  }

  *out_n_tokens = 0;

  if (out_stats) {
    memset(out_stats, 0, sizeof(*out_stats));
  }

  if (max_new == 0) {
    ie_log_info("ie_engine_generate_ex: max_new=0; nothing to do");
    return IE_OK;
  }

  const double t_total0 = now_s_();

  ie_log_info("ie_engine_generate_ex: begin (engine=%p max_new=%zu)", (const void *)e, max_new);
  log_prompt_preview_(prompt);

  const int log_tokens = env_flag_("IE_API_LOG_TOKENS", 0);
  const int log_every = env_int_("IE_API_LOG_EVERY", 0, 0, 1000000);
  const int dbg_decode = env_flag_("IE_API_DEBUG_DECODE", 0);
  const int dbg_decode_every = env_int_("IE_API_DEBUG_DECODE_EVERY", 0, 0, 1000000);

  const int dbg_topk = env_int_("IE_DEBUG_TOPK", 0, 0, 64);
  const int dbg_topk_every = env_int_("IE_DEBUG_TOPK_EVERY", 0, 0, 1000000);

  uint32_t need = 0;
  int rc = ie_tok_gptoss_encode(e->tok, prompt, NULL, &need);
  if (rc != 0) {
    ie_log_error("ie_engine_generate_ex: tokenizer encode size-query failed (rc=%d)", rc);
    return IE_ERR_MODEL;
  }

  ie_log_info("ie_engine_generate_ex: tokenizer size-query ok (need_tokens=%" PRIu32 ")", need);

  uint32_t n_prompt = (need == 0) ? 1u : need;
  uint32_t *prompt_ids = (uint32_t *)malloc((size_t)n_prompt * sizeof(uint32_t));
  if (!prompt_ids) {
    ie_log_error("ie_engine_generate_ex: OOM allocating prompt_ids (n_prompt=%" PRIu32 ")", n_prompt);
    return IE_ERR_OOM;
  }

  if (need == 0) {
    prompt_ids[0] = 0;
    n_prompt = 1;
    ie_log_warn("ie_engine_generate_ex: prompt encoded to empty; forcing token_id=0");
  } else {
    uint32_t inout = n_prompt;
    rc = ie_tok_gptoss_encode(e->tok, prompt, prompt_ids, &inout);
    if (rc != 0) {
      ie_log_error("ie_engine_generate_ex: tokenizer encode failed (rc=%d capacity=%" PRIu32 ")", rc, n_prompt);
      free(prompt_ids);
      return IE_ERR_MODEL;
    }
    n_prompt = inout;
    if (n_prompt == 0) {
      prompt_ids[0] = 0;
      n_prompt = 1;
      ie_log_warn("ie_engine_generate_ex: encode returned 0 tokens; forcing token_id=0");
    }
  }

  ie_log_info("ie_engine_generate_ex: encoded prompt tokens=%" PRIu32, n_prompt);

  size_t trace_prompt_index = 0;
  const int trace_prompt = trace_ids_prompt_index_(&trace_prompt_index);
  if (trace_prompt) {
    trace_ids_prompt_(trace_prompt_index, prompt, prompt_ids, n_prompt);
  }

  if (log_tokens) {
    const uint32_t cap = (n_prompt < 32u) ? n_prompt : 32u;
    char line[512];
    size_t pos = 0;

    pos += (size_t)snprintf(line + pos, sizeof(line) - pos, "prompt_ids[0..%u):", (unsigned)cap);
    for (uint32_t i = 0; i < cap && pos + 16 < sizeof(line); ++i) {
      pos += (size_t)snprintf(line + pos, sizeof(line) - pos, " %u", (unsigned)prompt_ids[i]);
    }
    if (cap < n_prompt && pos + 8 < sizeof(line)) {
      (void)snprintf(line + pos, sizeof(line) - pos, " ...");
    }
    ie_log_info("%s", line);
  }

  if (e->hp.n_layers == 0) {
    ie_log_error("ie_engine_generate_ex: invalid hparams (n_layers=0)");
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  const size_t n_layers = (size_t)e->hp.n_layers;
  ie_kv_cache *kv_layers = (ie_kv_cache *)calloc(n_layers, sizeof(*kv_layers));
  if (!kv_layers) {
    ie_log_error("ie_engine_generate_ex: OOM allocating kv_layers (n_layers=%zu)", n_layers);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  ie_kv_opts kv_opts;
  memset(&kv_opts, 0, sizeof(kv_opts));
  kv_opts.storage = IE_KV_STORAGE_F32;
  kv_opts.heads = (int32_t)e->hp.n_kv_heads;
  kv_opts.head_dim = (int32_t)e->hp.d_head;

  int32_t want_seq = (int32_t)n_prompt + (int32_t)max_new + 8;
  if (want_seq < 16) want_seq = 16;

  int32_t hp_seq = (int32_t)e->hp.max_seq;
  if (hp_seq <= 0) hp_seq = want_seq;

  kv_opts.max_seq = (want_seq < hp_seq) ? want_seq : hp_seq;

  ie_log_info("ie_engine_generate_ex: kv_opts heads=%" PRIi32 " head_dim=%" PRIi32 " want_seq=%" PRIi32 " hp_seq=%" PRIi32 " max_seq=%" PRIi32,
              kv_opts.heads,
              kv_opts.head_dim,
              want_seq,
              hp_seq,
              kv_opts.max_seq);

  if (kv_opts.heads <= 0 || kv_opts.head_dim <= 0 || kv_opts.max_seq <= 0) {
    ie_log_error("ie_engine_generate_ex: invalid kv_opts (heads=%" PRIi32 " head_dim=%" PRIi32 " max_seq=%" PRIi32 ")",
                 kv_opts.heads,
                 kv_opts.head_dim,
                 kv_opts.max_seq);
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_MODEL;
  }

  if (ie_kv_init_layers(kv_layers, (int)e->hp.n_layers, &kv_opts) != 0) {
    ie_log_error("ie_engine_generate_ex: kv_init_layers failed (n_layers=%" PRIu32 ")", e->hp.n_layers);
    free(kv_layers);
    free(prompt_ids);
    return IE_ERR_OOM;
  }

  const double t_prefill0 = now_s_();
  ie_log_info("ie_engine_generate_ex: prefill begin (n_prompt=%" PRIu32 " logits_len=%zu)", n_prompt, e->logits_len);

  rc = ie_gptoss_infer_prefill(e->infer, kv_layers, prompt_ids, n_prompt, e->logits);
  const double t_prefill1 = now_s_();
  if (rc != 0) {
    const ie_status_t st = status_from_rc_(rc);
    ie_log_error("ie_engine_generate_ex: prefill failed (rc=%d -> %s)", rc, status_str_(st));
    ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
    free(kv_layers);
    free(prompt_ids);
    return st;
  }

  ie_log_info("ie_engine_generate_ex: prefill ok (time=%.6fs)", (t_prefill1 > t_prefill0) ? (t_prefill1 - t_prefill0) : 0.0);

  if (dbg_topk > 0) {
    debug_dump_topk_(e->tok, e->logits, e->logits_len, dbg_topk, "prefill");
  }

  const double t_decode0 = now_s_();

  size_t produced = 0;
  double ttft_s = 0.0;
  const int use_cuda_greedy_device =
      (e->sample_cfg.kind == IE_SAMPLE_GREEDY) && (getenv("IE_CUDA_GREEDY_DEVICE") != NULL);

  for (; produced < max_new; ++produced) {
    uint32_t next = 0;

    if (use_cuda_greedy_device) {
      rc = ie_gptoss_infer_argmax_device(e->infer, &next);
      if (rc != 0) {
        ie_log_error("ie_engine_generate_ex: argmax_device failed at step=%zu (rc=%d).", produced, rc);
        ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
        free(kv_layers);
        free(prompt_ids);
        return IE_ERR_INTERNAL;
      }
    } else {
      rc = ie_sample_next(e->logits,
                          e->logits_len,
                          &e->sample_cfg,
                          (ie_rng_t *)&e->rng,
                          e->scratch_idx,
                          e->scratch_prob,
                          e->scratch_cap,
                          &next);
      if (rc != 0) {
        ie_log_error("ie_engine_generate_ex: sample_next failed at step=%zu (rc=%d). Stopping early.", produced, rc);
        break;
      }
    }

    out_tokens[produced] = (int)next;

    if (trace_prompt) {
      trace_ids_step_(trace_prompt_index, produced, next);
    }

    if (log_every > 0 && (produced % (size_t)log_every) == 0) {
      ie_log_info("ie_engine_generate_ex: step=%zu next_token=%" PRIu32, produced, next);
    }

    rc = ie_gptoss_infer_step(e->infer, kv_layers, next, e->logits);
    if (rc != 0) {
      const ie_status_t st = status_from_rc_(rc);
      ie_log_error("ie_engine_generate_ex: infer_step failed at step=%zu token=%" PRIu32 " (rc=%d -> %s)",
                   produced,
                   next,
                   rc,
                   status_str_(st));
      ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
      free(kv_layers);
      free(prompt_ids);
      return st;
    }


    if (produced == 0) {
      const double t_now = now_s_();
      ttft_s = (t_now > t_total0) ? (t_now - t_total0) : 0.0;
    }

    if (dbg_topk > 0 && dbg_topk_every > 0 && (produced % (size_t)dbg_topk_every) == 0) {
      char tag[64];
      (void)snprintf(tag, sizeof(tag), "step_%zu", produced);
      debug_dump_topk_(e->tok, e->logits, e->logits_len, dbg_topk, tag);
    }

    if (dbg_decode && dbg_decode_every > 0 && (produced % (size_t)dbg_decode_every) == 0) {
      size_t need_bytes = 0;
      const uint32_t cnt = (uint32_t)(produced + 1);
      const int drc = ie_tok_gptoss_decode(e->tok, (const uint32_t *)out_tokens, cnt, NULL, &need_bytes);
      if (drc == 0 && need_bytes > 0 && need_bytes < (size_t)(1u << 22)) {
        char *txt = (char *)malloc(need_bytes);
        if (txt) {
          size_t inout = need_bytes;
          const int drc2 = ie_tok_gptoss_decode(e->tok, (const uint32_t *)out_tokens, cnt, txt, &inout);
          if (drc2 == 0) {
            txt[need_bytes - 1] = '\0';
            ie_log_info("ie_engine_generate_ex: decoded[%u] bytes=%zu preview=\"%s%s\"",
                        cnt,
                        inout,
                        txt,
                        (inout > 120 ? "..." : ""));
          } else {
            ie_log_warn("ie_engine_generate_ex: decode failed (rc=%d) at step=%zu", drc2, produced);
          }
          free(txt);
        }
      } else {
        ie_log_warn("ie_engine_generate_ex: decode size-query failed (rc=%d need_bytes=%zu) at step=%zu",
                    drc,
                    need_bytes,
                    produced);
      }
    }
  }

  const double t_decode1 = now_s_();

  *out_n_tokens = produced;

  ie_log_info("ie_engine_generate_ex: done (produced=%zu requested=%zu)", produced, max_new);

  ie_kv_free_layers(kv_layers, (int)e->hp.n_layers);
  free(kv_layers);
  free(prompt_ids);

  const double t_total1 = now_s_();

  if (out_stats) {
    out_stats->wall_time_s = (t_total1 > t_total0) ? (t_total1 - t_total0) : 0.0;
    out_stats->prefill_time_s = (t_prefill1 > t_prefill0) ? (t_prefill1 - t_prefill0) : 0.0;
    out_stats->decode_time_s = (t_decode1 > t_decode0) ? (t_decode1 - t_decode0) : 0.0;
    out_stats->ttft_s = ttft_s;

    out_stats->tps_decode = (out_stats->decode_time_s > 0.0) ? ((double)produced / out_stats->decode_time_s) : 0.0;
    out_stats->tps_total = (out_stats->wall_time_s > 0.0) ? ((double)produced / out_stats->wall_time_s) : 0.0;
  }

  return IE_OK;
}
