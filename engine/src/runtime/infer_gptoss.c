/* ============================================================================
 * File: engine/src/runtime/infer_gptoss.c
 * ============================================================================
 */
/**
 * @file infer_gptoss.c
 * @brief GPT-OSS inference entrypoints with a CPU-first forward pass.
 *
 * @details
 * This implementation is intentionally "tensor_map-minimal":
 * - It does NOT require tensor_map.json to carry dtype/shape metadata.
 * - It only requires (name, offset, size_bytes).
 *
 * Safety:
 * - We validate tensor sizes against hparams before using them.
 * - We infer MoE n_experts from router weight size.
 *
 * Notes:
 * - This is correctness-first and CPU-only for BF16/Q4 paths.
 * - Quant assumptions for expert blocks follow a Q4_0-like layout:
 *   - cols multiple of 32
 *   - blocks: 16 bytes per 32 weights (2x4-bit per byte)
 *   - scales: BF16 per block, or FP8(E4M3) packed as u8 per block
 *
 * Logging:
 * - This file prints detailed diagnostics to stderr on failure paths.
 * - Set IE_LOG_LEVEL to control verbosity:
 *     0=quiet, 1=error, 2=warn, 3=info, 4=debug, 5=trace
 * - Hot-path logging is avoided; errors include rich context (tensor name,
 *   expected/actual sizes, layer index, and chosen candidate names).
 */

#define _POSIX_C_SOURCE 200809L

#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ie_device.h"
#include "ie_infer.h"
#include "ie_kernels.h"
#include "ie_kv_cache.h"
#include "ie_kv_instrumentation.h"
#include "tensor_map.h"

/* ------------------------------------------------------------------------- */
/* Configuration defaults                                                     */
/* ------------------------------------------------------------------------- */

/** @brief Default RMSNorm epsilon (matches common HF defaults). */
#define IE_GPTOSS_RMS_EPS_DEFAULT (1e-5f)
/** @brief Default RoPE theta (matches common HF defaults). */
#define IE_GPTOSS_ROPE_THETA_DEFAULT (10000.0f)
/** @brief Default MoE routing top-k (most MoE LLMs use 4). */
#define IE_GPTOSS_MOE_TOPK_DEFAULT (4u)

/* ------------------------------------------------------------------------- */
/* Logging                                                                    */
/* ------------------------------------------------------------------------- */

/**
 * @brief Internal log levels (higher means more verbose).
 */
enum {
  IE_LL_QUIET = 0,
  IE_LL_ERROR = 1,
  IE_LL_WARN  = 2,
  IE_LL_INFO  = 3,
  IE_LL_DEBUG = 4,
  IE_LL_TRACE = 5
};

/** @brief Cached log level parsed from IE_LOG_LEVEL (negative means "unset"). */
static int ie_log_level_cached = -1;

/**
 * @brief Read IE_LOG_LEVEL once and cache the result.
 * @return Cached log level in [0..5].
 */
static int ie_log_level(void) {
  if (ie_log_level_cached >= 0) return ie_log_level_cached;

  const char *s = getenv("IE_LOG_LEVEL");
  int lvl = IE_LL_WARN;
  if (s && *s) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end != s) {
      if (v < 0) v = 0;
      if (v > 5) v = 5;
      lvl = (int)v;
    }
  }
  ie_log_level_cached = lvl;
  return lvl;
}

/**
 * @brief Print a formatted log line if the level is enabled.
 * @param lvl Message level.
 * @param file Source file.
 * @param line Source line.
 * @param fmt printf-style format string.
 */
static void ie_logf(int lvl, const char *file, int line, const char *fmt, ...) {
  if (ie_log_level() < lvl) return;

  const char *tag = "LOG";
  if (lvl == IE_LL_ERROR) tag = "ERR";
  else if (lvl == IE_LL_WARN) tag = "WRN";
  else if (lvl == IE_LL_INFO) tag = "INF";
  else if (lvl == IE_LL_DEBUG) tag = "DBG";
  else if (lvl == IE_LL_TRACE) tag = "TRC";

  fprintf(stderr, "[%s] %s:%d: ", tag, file, line);

  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fputc('\n', stderr);
}

/** @brief Emit an ERROR-level log line. */
#define IE_LOGE(...) ie_logf(IE_LL_ERROR, __FILE__, __LINE__, __VA_ARGS__)
/** @brief Emit a WARN-level log line. */
#define IE_LOGW(...) ie_logf(IE_LL_WARN,  __FILE__, __LINE__, __VA_ARGS__)
/** @brief Emit an INFO-level log line. */
#define IE_LOGI(...) ie_logf(IE_LL_INFO,  __FILE__, __LINE__, __VA_ARGS__)
/** @brief Emit a DEBUG-level log line. */
#define IE_LOGD(...) ie_logf(IE_LL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
/** @brief Emit a TRACE-level log line. */
#define IE_LOGT(...) ie_logf(IE_LL_TRACE, __FILE__, __LINE__, __VA_ARGS__)

/* ------------------------------------------------------------------------- */
/* Weak-reference helpers for kernels that may not exist yet                  */
/* ------------------------------------------------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
/** @brief Weak symbol attribute for optional kernel overrides. */
#define IE_WEAK __attribute__((weak))
#else
/** @brief Weak symbol attribute fallback (no-op). */
#define IE_WEAK
#endif

/* ------------------------------------------------------------------------- */
/* Forward declarations for internal types                                    */
/* ------------------------------------------------------------------------- */

/** @brief Forward declaration for the internal inference implementation. */
struct ie_gptoss_infer_impl;

/**
 * @brief Destroy an inference context.
 * @param ctx Context handle (may be NULL).
 */
void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx);

/* ------------------------------------------------------------------------- */
/* Weight structs                                                             */
/* ------------------------------------------------------------------------- */

/**
 * @brief Per-layer MoE weight views (router + expert matrices).
 *
 * Expert matrices are assumed to be stored in "expert-major" order:
 * contiguous blocks/scales/bias for each expert, with strides recorded here.
 */
typedef struct ie_moe_w {
  /** @brief Number of experts for this layer. */
  uint32_t n_experts;

  /** @brief Router weight (BF16), shape [n_experts, d_model]. */
  const uint16_t *router_w;

  /** @brief Router bias (BF16), shape [n_experts], optional. */
  const uint16_t *router_b;

  /** @brief Whether router weight is stored transposed (currently unused). */
  int router_transposed;

  /** @brief gate_up Q4 blocks for all experts. */
  const uint8_t *gate_up_blocks;

  /** @brief gate_up scales for all experts (BF16 or FP8 stored as u8). */
  const uint8_t *gate_up_scales;

  /** @brief Bytes per gate_up scale element (2 = BF16, 1 = FP8 stored as u8). */
  uint8_t gate_up_scale_bytes;

  /** @brief gate_up BF16 bias for all experts, optional. */
  const uint16_t *gate_up_bias;

  /** @brief down Q4 blocks for all experts. */
  const uint8_t *down_blocks;

  /** @brief down scales for all experts (BF16 or FP8 stored as u8). */
  const uint8_t *down_scales;

  /** @brief Bytes per down scale element (2 = BF16, 1 = FP8 stored as u8). */
  uint8_t down_scale_bytes;

  /** @brief down BF16 bias for all experts, optional. */
  const uint16_t *down_bias;

  /** @brief Stride in bytes from one expert's gate_up blocks to the next. */
  size_t gate_up_blocks_stride;

  /** @brief Stride in bytes from one expert's gate_up scales to the next. */
  size_t gate_up_scales_stride;

  /** @brief Stride in bytes from one expert's gate_up bias to the next (0 if absent). */
  size_t gate_up_bias_stride;

  /** @brief Stride in bytes from one expert's down blocks to the next. */
  size_t down_blocks_stride;

  /** @brief Stride in bytes from one expert's down scales to the next. */
  size_t down_scales_stride;

  /** @brief Stride in bytes from one expert's down bias to the next (0 if absent). */
  size_t down_bias_stride;
} ie_moe_w_t;

/**
 * @brief Per-layer weight views for attention + MoE.
 *
 * All attention weights are BF16 (HF-style linear layout [out, in]).
 * MoE expert weights are Q4_0-like blocks/scales with optional BF16 bias.
 */
typedef struct ie_gptoss_layer_w {
  /** @brief RMSNorm #1 weight (BF16), shape [d_model]. */
  const uint16_t *ln1_w;

  /** @brief RMSNorm #2 weight (BF16), shape [d_model]. */
  const uint16_t *ln2_w;

  /** @brief Q projection weight (BF16), shape [q_dim, d_model]. */
  const uint16_t *q_w;

  /** @brief Q projection bias (BF16), shape [q_dim], optional. */
  const uint16_t *q_b;

  /** @brief K projection weight (BF16), shape [kv_dim, d_model]. */
  const uint16_t *k_w;

  /** @brief K projection bias (BF16), shape [kv_dim], optional. */
  const uint16_t *k_b;

  /** @brief V projection weight (BF16), shape [kv_dim, d_model]. */
  const uint16_t *v_w;

  /** @brief V projection bias (BF16), shape [kv_dim], optional. */
  const uint16_t *v_b;

  /** @brief O projection weight (BF16), shape [d_model, q_dim]. */
  const uint16_t *o_w;

  /** @brief O projection bias (BF16), shape [d_model], optional. */
  const uint16_t *o_b;

  /** @brief MoE weights for this layer. */
  ie_moe_w_t moe;
} ie_gptoss_layer_w_t;

/**
 * @brief Internal GPT-OSS inference context.
 *
 * Owns:
 * - tensor_map loaded from model directory
 * - mmap view of model.ie.bin
 * - resolved pointers to all weights
 * - temporary activation buffers (FP32)
 */
struct ie_gptoss_infer_impl {
  /** @brief Device handle (currently unused for CPU-only path). */
  ie_device_t *dev;

  /** @brief Weights descriptor (paths, loader-provided metadata). */
  const ie_weights_t *weights;

  /** @brief Hyperparameters (dimensions, layer counts, etc.). */
  const ie_gptoss_hparams_t *hp;

  /** @brief Tensor map (name -> offset/size_bytes). */
  tensor_map_t tmap;

  /** @brief Path used to load tensor_map.json (debug). */
  char *tmap_path_used;

  /** @brief Base pointer of read-only mmap for model.ie.bin. */
  uint8_t *bin_base;

  /** @brief Size in bytes of mapped model.ie.bin. */
  size_t bin_size;

  /** @brief Current generation position (token index in sequence). */
  uint32_t pos;

  /** @brief Embedding table (BF16), shape [vocab, d_model]. */
  const uint16_t *w_embed_bf16;

  /** @brief Final norm weight (BF16), shape [d_model]. */
  const uint16_t *w_norm_bf16;

  /** @brief LM head weight (BF16), shape [vocab, d_model]. */
  const uint16_t *w_lm_bf16;

  /** @brief Per-layer resolved weight pointers. */
  ie_gptoss_layer_w_t *layers;

  /** @brief Activation: token embedding / residual stream (FP32), shape [d_model]. */
  float *x;

  /** @brief Activation scratch (FP32), shape [d_model]. */
  float *x1;

  /** @brief Activation scratch / residual add buffer (FP32), shape [d_model]. */
  float *x2;

  /** @brief Q buffer (FP32), shape [n_heads * d_head]. */
  float *q;

  /** @brief K buffer (FP32), shape [n_kv_heads * d_head]. */
  float *k;

  /** @brief V buffer (FP32), shape [n_kv_heads * d_head]. */
  float *v;

  /** @brief Attention output (FP32), shape [n_heads * d_head]. */
  float *attn_out;

  /** @brief MLP gate buffer (FP32), shape [d_ff]. */
  float *mlp_gate;

  /**
   * @brief MLP gate_up projection output (FP32), shape [2 * d_ff].
   *
   * The first half is the "gate" pre-activation; the second half is the "up" pre-activation.
   */
  float *mlp_up;

  /** @brief Softmax scores scratch (FP32), shape [max_seq]. */
  float *scores;

  /** @brief Router logits scratch (FP32), shape [max_experts]. */
  float *router_logits;

  /** @brief RMSNorm epsilon. */
  float rms_eps;

  /** @brief RoPE theta (base). */
  float rope_theta;

  /**
   * @brief RoPE scaling multiplier applied to rope_theta at runtime.
   *
   * If your HF config uses rope_scaling, load the factor into this field.
   * When no scaling is present, this stays at 1.0.
   */
  float rope_scale;

  /** @brief Maximum experts across all layers (for scratch allocation). */
  uint32_t max_experts;

  /**
   * @brief MoE top-k routing selection (clamped to [1..n_experts]).
   *
   * Default is 4 (IE_GPTOSS_MOE_TOPK_DEFAULT).
   */
  uint32_t moe_topk;
};

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Return the minimum of two size_t values.
 * @param a First value.
 * @param b Second value.
 * @return min(a, b).
 */
static size_t ie_min_size(size_t a, size_t b) { return (a < b) ? a : b; }

/**
 * @brief Return the minimum of two uint32_t values.
 * @param a First value.
 * @param b Second value.
 * @return min(a, b).
 */
static uint32_t ie_u32_min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

/**
 * @brief Allocate and return the directory portion of a path.
 *
 * If no slash is present, returns ".". If the path is rooted and the only slash
 * is the first character, returns "/".
 *
 * @param path Input path.
 * @return Newly allocated directory string (caller frees), or NULL on OOM.
 */
static char *ie_dirname_alloc(const char *path) {
  if (!path) return NULL;

  const char *slash = strrchr(path, '/');
  if (!slash) {
    char *out = (char *)malloc(2);
    if (!out) return NULL;
    out[0] = '.';
    out[1] = '\0';
    return out;
  }

  const size_t len = (size_t)(slash - path);
  if (len == 0) {
    char *out = (char *)malloc(2);
    if (!out) return NULL;
    out[0] = '/';
    out[1] = '\0';
    return out;
  }

  char *out = (char *)malloc(len + 1);
  if (!out) return NULL;
  memcpy(out, path, len);
  out[len] = '\0';
  return out;
}

/**
 * @brief Join two path components with exactly one slash.
 * @param a Left component.
 * @param b Right component.
 * @return Newly allocated joined path (caller frees), or NULL on OOM.
 */
static char *ie_path_join_alloc(const char *a, const char *b) {
  if (!a || !b) return NULL;
  const size_t la = strlen(a);
  const size_t lb = strlen(b);
  const int need_slash = (la > 0 && a[la - 1] != '/');

  char *out = (char *)malloc(la + (size_t)need_slash + lb + 1);
  if (!out) return NULL;

  memcpy(out, a, la);
  size_t off = la;
  if (need_slash) out[off++] = '/';
  memcpy(out + off, b, lb);
  out[off + lb] = '\0';
  return out;
}

/**
 * @brief Memory-map a file read-only.
 * @param path File path.
 * @param out_base Receives mapped base address.
 * @param out_size Receives mapped size in bytes.
 * @return 0 on success, negative error code on failure.
 */
static int ie_mmap_ro(const char *path, uint8_t **out_base, size_t *out_size) {
  if (!path || !out_base || !out_size) return -1;
  *out_base = NULL;
  *out_size = 0;

  const int fd = open(path, O_RDONLY);
  if (fd < 0) return -2;

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return -3;
  }
  if (st.st_size <= 0) {
    close(fd);
    return -4;
  }

  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return -5;

  *out_base = (uint8_t *)p;
  *out_size = (size_t)st.st_size;
  return 0;
}

/**
 * @brief Unmap a previously mapped read-only file.
 * @param base Mapped base address.
 * @param size Mapped size in bytes.
 */
static void ie_munmap_ro(uint8_t *base, size_t size) {
  if (!base || size == 0) return;
  (void)munmap((void *)base, size);
}

/**
 * @brief Parse an environment variable as uint32, with bounds and default.
 * @param name Environment variable name.
 * @param def Default if unset or invalid.
 * @param lo Inclusive lower bound.
 * @param hi Inclusive upper bound.
 * @return Parsed and clamped value.
 */
static uint32_t ie_env_u32_clamped(const char *name, uint32_t def, uint32_t lo, uint32_t hi) {
  const char *s = getenv(name);
  if (!s || !*s) return def;

  char *end = NULL;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s) return def;

  if (v < (unsigned long)lo) v = (unsigned long)lo;
  if (v > (unsigned long)hi) v = (unsigned long)hi;
  return (uint32_t)v;
}

/**
 * @brief Parse an environment variable as float, with bounds and default.
 * @param name Environment variable name.
 * @param def Default if unset or invalid.
 * @param lo Inclusive lower bound.
 * @param hi Inclusive upper bound.
 * @return Parsed and clamped value.
 */
static float ie_env_f32_clamped(const char *name, float def, float lo, float hi) {
  const char *s = getenv(name);
  if (!s || !*s) return def;

  char *end = NULL;
  float v = strtof(s, &end);
  if (end == s) return def;

  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return v;
}

/* ------------------------------------------------------------------------- */
/* BF16 helpers                                                               */
/* ------------------------------------------------------------------------- */

/**
 * @brief Convert a BF16 value to float32.
 * @param v BF16 bits.
 * @return float32 value.
 */
static inline float ie_bf16_to_f32(uint16_t v) {
  union {
    uint32_t u;
    float f;
  } t;
  t.u = ((uint32_t)v) << 16;
  return t.f;
}

/**
 * @brief SiLU activation function.
 * @param x Input.
 * @return SiLU(x).
 */
static inline float ie_silu_f32(float x) { return x / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------------- */
/* Math primitives                                                            */
/* ------------------------------------------------------------------------- */

/**
 * @brief Dot product of two float32 vectors.
 * @param a Vector A.
 * @param b Vector B.
 * @param n Number of elements.
 * @return Sum_i a[i] * b[i].
 */
static float ie_dot_f32(const float *a, const float *b, size_t n) {
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

/**
 * @brief In-place softmax over an array of float32 values.
 * @param x Array (modified in-place).
 * @param n Number of elements.
 */
static void ie_softmax_inplace_f32(float *x, size_t n) {
  if (!x || n == 0u) return;

  float m = x[0];
  for (size_t i = 1; i < n; ++i)
    if (x[i] > m) m = x[i];

  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    x[i] = expf(x[i] - m);
    sum += x[i];
  }

  const float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
  for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

/* ------------------------------------------------------------------------- */
/* Generic weight view                                                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief A resolved tensor pointer plus its descriptor.
 */
typedef struct ie_tensor_view {
  /** @brief Pointer into the mapped weights file. */
  const void *ptr;
  /** @brief Tensor descriptor entry. */
  const tensor_desc_t *desc;
} ie_tensor_view_t;

/**
 * @brief Find a tensor descriptor by name in a tensor map.
 * @param tmap Tensor map.
 * @param name Tensor name.
 * @return Descriptor pointer or NULL if not found.
 */
static const tensor_desc_t *ie_w_find(const tensor_map_t *tmap, const char *name) {
  if (!tmap || !name) return NULL;
  return tensor_map_find(tmap, name);
}

/**
 * @brief Resolve a tensor pointer by name.
 * @param impl Inference implementation.
 * @param name Tensor name.
 * @param out Output view.
 * @return 0 on success, negative code on failure.
 */
static int ie_w_get_view(const struct ie_gptoss_infer_impl *impl, const char *name,
                         ie_tensor_view_t *out) {
  if (!impl || !name || !out) return -1;
  out->ptr = NULL;
  out->desc = NULL;

  const tensor_desc_t *td = ie_w_find(&impl->tmap, name);
  if (!td) return -2;

  const uint64_t off = td->offset;
  const uint64_t nb = td->size_bytes;
  if ((uint64_t)impl->bin_size < off + nb) return -3;

  out->desc = td;
  out->ptr = (const void *)(impl->bin_base + (size_t)off);
  return 0;
}

/**
 * @brief Resolve the first tensor found among a list of candidate names.
 * @param impl Inference implementation.
 * @param names Candidate name array.
 * @param count Candidate count.
 * @param out Output view.
 * @return 0 on success, negative code on failure.
 */
static int ie_w_get_first_view(const struct ie_gptoss_infer_impl *impl, const char *const *names,
                               size_t count, ie_tensor_view_t *out) {
  if (!impl || !names || !out) return -1;
  for (size_t i = 0; i < count; ++i) {
    const int rc = ie_w_get_view(impl, names[i], out);
    if (rc == 0) {
      char td_buf[256];
      if (out->desc && tensor_desc_to_string(out->desc, td_buf, sizeof(td_buf)) >= 0) {
        IE_LOGD("resolved tensor: %s", td_buf);
      } else {
        IE_LOGD("resolved tensor: name=%s", names[i]);
      }
      return 0;
    }
  }
  out->ptr = NULL;
  out->desc = NULL;
  return -2;
}

/**
 * @brief Resolve a layer tensor using a list of snprintf formats.
 * @param impl Inference implementation.
 * @param layer Layer index.
 * @param fmts Candidate snprintf formats (must include %u for layer).
 * @param count Candidate format count.
 * @param out Output view.
 * @param required If nonzero, missing tensor is an error; otherwise returns success with NULL view.
 * @return 0 on success, negative code on failure.
 */
static int ie_w_get_layer_fmt_view(const struct ie_gptoss_infer_impl *impl, uint32_t layer,
                                   const char *const *fmts, size_t count, ie_tensor_view_t *out,
                                   int required) {
  if (!impl || !fmts || !out) return -1;
  out->ptr = NULL;
  out->desc = NULL;

  char name[256];
  for (size_t i = 0; i < count; ++i) {
    int n = snprintf(name, sizeof(name), fmts[i], layer);
    if (n <= 0 || (size_t)n >= sizeof(name)) continue;
    if (ie_w_get_view(impl, name, out) == 0) {
      char td_buf[256];
      if (out->desc && tensor_desc_to_string(out->desc, td_buf, sizeof(td_buf)) >= 0) {
        IE_LOGD("resolved layer tensor: layer=%u %s", (unsigned)layer, td_buf);
      } else {
        IE_LOGD("resolved layer tensor: layer=%u name=%s", (unsigned)layer, name);
      }
      return 0;
    }
  }

  if (!required) {
    out->ptr = NULL;
    out->desc = NULL;
    return 0;
  }

  IE_LOGE("missing required layer tensor: layer=%u (tried %zu formats)", (unsigned)layer, count);
  for (size_t i = 0; i < count; ++i) {
    IE_LOGE("  candidate fmt[%zu]=%s", i, fmts[i]);
  }
  return -2;
}

/**
 * @brief Require exact byte size for a tensor.
 * @param td Tensor descriptor.
 * @param want_bytes Required byte size.
 * @return 0 if sizes match, negative code otherwise.
 */
static int ie_require_size_eq(const tensor_desc_t *td, uint64_t want_bytes) {
  if (!td) return -1;
  if (td->size_bytes == want_bytes) return 0;

  char td_buf[256];
  if (tensor_desc_to_string(td, td_buf, sizeof(td_buf)) >= 0) {
    IE_LOGE("tensor size mismatch: want_bytes=%llu got: %s",
            (unsigned long long)want_bytes, td_buf);
  } else {
    IE_LOGE("tensor size mismatch: want_bytes=%llu got_bytes=%llu name=%s",
            (unsigned long long)want_bytes,
            (unsigned long long)td->size_bytes,
            (td->name ? td->name : "<null>"));
  }
  return -2;
}

/* ------------------------------------------------------------------------- */
/* RMSNorm + RoPE                                                             */
/* ------------------------------------------------------------------------- */

IE_WEAK int ie_rmsnorm_cpu_f32(const float *x, const float *w, size_t n, float eps, float *y) {
  if (!x || !w || !y || n == 0u) return -1;

  float ss = 0.0f;
  for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];

  const float inv = 1.0f / sqrtf(ss / (float)n + eps);

  for (size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
  return 0;
}

static int ie_rmsnorm_cpu_f32_bf16w(const float *x, const uint16_t *w_bf16, size_t n, float eps,
                                   float *y) {
  if (!x || !w_bf16 || !y || n == 0u) return -1;

  float ss = 0.0f;
  for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];
  const float inv = 1.0f / sqrtf(ss / (float)n + eps);

  for (size_t i = 0; i < n; ++i) {
    const float wi = ie_bf16_to_f32(w_bf16[i]);
    y[i] = x[i] * inv * wi;
  }
  return 0;
}

IE_WEAK int ie_rope_apply_f32(float *q, float *k, size_t heads, size_t head_dim, uint32_t pos,
                             float theta) {
  if (head_dim == 0u || (head_dim & 1u) != 0u) return -1;
  if (!q && !k) return 0;

  const size_t half = head_dim / 2u;
  const float log_theta = logf(theta);

  for (size_t h = 0; h < heads; ++h) {
    float *vq = q ? (q + h * head_dim) : NULL;
    float *vk = k ? (k + h * head_dim) : NULL;

    for (size_t i = 0; i < half; ++i) {
      const float exponent = -2.0f * (float)i / (float)head_dim;
      const float inv_freq = expf(log_theta * exponent);
      const float ang = (float)pos * inv_freq;
      const float c = cosf(ang);
      const float s = sinf(ang);

      const size_t i0 = 2u * i;
      const size_t i1 = i0 + 1u;

      if (vq) {
        const float x0 = vq[i0];
        const float x1 = vq[i1];
        vq[i0] = x0 * c - x1 * s;
        vq[i1] = x0 * s + x1 * c;
      }

      if (vk) {
        const float x0 = vk[i0];
        const float x1 = vk[i1];
        vk[i0] = x0 * c - x1 * s;
        vk[i1] = x0 * s + x1 * c;
      }
    }
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* GEMV: BF16 matrix x FP32 vector -> FP32 output                             */
/* ------------------------------------------------------------------------- */

static int ie_gemv_bf16_f32(const uint16_t *W_bf16, const float *x, float *y, size_t rows,
                            size_t cols, const uint16_t *bias_bf16) {
  if (!W_bf16 || !x || !y || rows == 0u || cols == 0u) return -1;

  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const size_t base = r * cols;
    for (size_t c = 0; c < cols; ++c) {
      const float w = ie_bf16_to_f32(W_bf16[base + c]);
      acc += w * x[c];
    }
    if (bias_bf16) acc += ie_bf16_to_f32(bias_bf16[r]);
    y[r] = acc;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* FP8 scale decode (E4M3, finite-only)                                       */
/* ------------------------------------------------------------------------- */

static inline float ie_fp8_e4m3_to_f32(uint8_t v) {
  if (v == 0u) return 0.0f;
  const uint8_t sign = (v >> 7) & 0x1u;
  const uint8_t exp  = (v >> 3) & 0xFu;
  const uint8_t man  = (v & 0x7u);

  if (exp == 0u) {
    return sign ? -0.0f : 0.0f;
  }

  const int bias = 7;
  const int e = (int)exp - bias;

  const float frac = (float)man / 8.0f;
  const float val  = (1.0f + frac) * ldexpf(1.0f, e);
  return sign ? -val : val;
}

/* ------------------------------------------------------------------------- */
/* Q4_0-like blocks: row dot                                                  */
/* ------------------------------------------------------------------------- */

static float ie_q4_0_row_dot_f32(const uint8_t *row_blocks, const uint8_t *row_scales,
                                uint8_t scale_bytes, const float *x, size_t cols) {
  const size_t qk = 32u;
  const size_t qs_bytes = qk / 2u;
  const size_t nb = cols / qk;

  float acc = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    float d;
    if (scale_bytes == 2u) {
      const uint16_t *s16 = (const uint16_t *)(const void *)row_scales;
      d = ie_bf16_to_f32(s16[b]);
    } else {
      d = ie_fp8_e4m3_to_f32(row_scales[b]);
    }
    const uint8_t *qs = row_blocks + b * qs_bytes;

    const size_t x0 = b * qk;
    for (size_t i = 0; i < qs_bytes; ++i) {
      const uint8_t packed = qs[i];
      const int v0 = (int)(packed & 0x0F) - 8;
      const int v1 = (int)(packed >> 4) - 8;

      acc += ((float)v0 * d) * x[x0 + 2u * i + 0u];
      acc += ((float)v1 * d) * x[x0 + 2u * i + 1u];
    }
  }

  return acc;
}

static int ie_gemv_q4_0_f32(const uint8_t *W_blocks, const uint8_t *W_scales,
                           uint8_t scale_bytes, const float *x,
                           float *y, size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (!W_blocks || !W_scales || !x || !y || rows == 0u || cols == 0u) return -1;
  if (!(scale_bytes == 1u || scale_bytes == 2u)) return -3;
  if ((cols % 32u) != 0u) return -2;

  const size_t qk = 32u;
  const size_t qs_bytes = qk / 2u;
  const size_t nb = cols / qk;

  const size_t blocks_per_row_bytes = nb * qs_bytes;
  const size_t scales_per_row = nb;

  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *rb = W_blocks + r * blocks_per_row_bytes;
    const uint8_t *rs = W_scales + (size_t)r * scales_per_row * (size_t)scale_bytes;
    float acc = ie_q4_0_row_dot_f32(rb, rs, scale_bytes, x, cols);
    if (bias_bf16) acc += ie_bf16_to_f32(bias_bf16[r]);
    y[r] = acc;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Attention                                                                  */
/* ------------------------------------------------------------------------- */

IE_WEAK int ie_attn_cpu_causal_f32(const float *Q, const float *K, const float *V,
                                  size_t seq_len, size_t n_heads, size_t head_dim,
                                  float inv_sqrt_d, float *out, float *scores) {
  if (!Q || !K || !V || !out || !scores) return -1;
  if (seq_len == 0u || n_heads == 0u || head_dim == 0u) return -2;

  for (size_t h = 0; h < n_heads; ++h) {
    const float *Qh = Q + h * head_dim;

    for (size_t t = 0; t < seq_len; ++t) {
      const size_t base = (t * n_heads + h) * head_dim;
      const float *Kh = K + base;
      scores[t] = ie_dot_f32(Qh, Kh, head_dim) * inv_sqrt_d;
    }

    ie_softmax_inplace_f32(scores, seq_len);

    float *Oh = out + h * head_dim;
    for (size_t d = 0; d < head_dim; ++d) Oh[d] = 0.0f;

    for (size_t t = 0; t < seq_len; ++t) {
      const float w = scores[t];
      const size_t base = (t * n_heads + h) * head_dim;
      const float *Vh = V + base;
      for (size_t d = 0; d < head_dim; ++d) Oh[d] += w * Vh[d];
    }
  }

  return 0;
}

static void ie_attn_causal_gqa_f32(const float *Q, const float *K, const float *V, uint32_t seq_len,
                                  uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
                                  float inv_sqrt_d, float *out, float *scores) {
  const uint32_t group_size = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 0;

  for (uint32_t hq = 0; hq < n_heads; ++hq) {
    const uint32_t hk = (group_size > 0) ? (hq / group_size) : 0;
    const float *Qh = Q + (size_t)hq * (size_t)head_dim;

    for (uint32_t t = 0; t < seq_len; ++t) {
      const size_t base =
          ((size_t)t * (size_t)n_kv_heads + (size_t)hk) * (size_t)head_dim;
      const float *Kh = K + base;
      scores[t] = ie_dot_f32(Qh, Kh, (size_t)head_dim) * inv_sqrt_d;
    }

    ie_softmax_inplace_f32(scores, (size_t)seq_len);

    float *Oh = out + (size_t)hq * (size_t)head_dim;
    for (uint32_t d = 0; d < head_dim; ++d) Oh[d] = 0.0f;

    for (uint32_t t = 0; t < seq_len; ++t) {
      const float w = scores[t];
      const size_t base =
          ((size_t)t * (size_t)n_kv_heads + (size_t)hk) * (size_t)head_dim;
      const float *Vh = V + base;
      for (uint32_t d = 0; d < head_dim; ++d) Oh[d] += w * Vh[d];
    }
  }
}

/* ------------------------------------------------------------------------- */
/* MoE size helpers                                                           */
/* ------------------------------------------------------------------------- */

static int ie_infer_n_experts_from_router(const tensor_desc_t *td_router_w, uint32_t d_model,
                                         uint32_t *out_n_experts) {
  if (!td_router_w || !out_n_experts || d_model == 0u) return -1;

  const uint64_t bytes = td_router_w->size_bytes;
  const uint64_t denom = (uint64_t)d_model * 2u;
  if (denom == 0u) return -2;
  if ((bytes % denom) != 0u) return -3;

  const uint64_t n = bytes / denom;
  if (n == 0u || n > 65535u) return -4;

  *out_n_experts = (uint32_t)n;
  return 0;
}

static int ie_moe_expected_bytes(uint32_t n_experts, uint32_t rows, uint32_t cols,
                                uint8_t scale_bytes,
                                uint64_t *out_blocks_bytes, uint64_t *out_scales_bytes) {
  if (!out_blocks_bytes || !out_scales_bytes) return -1;
  if (n_experts == 0u || rows == 0u || cols == 0u) return -2;
  if ((cols % 32u) != 0u) return -3;
  if (!(scale_bytes == 1u || scale_bytes == 2u)) return -4;

  const uint64_t qk = 32u;
  const uint64_t qs_bytes = 16u;
  const uint64_t nb = (uint64_t)cols / qk;

  const uint64_t blocks_per_row_bytes = nb * qs_bytes;
  const uint64_t scales_per_row_bytes = nb * (uint64_t)scale_bytes;

  const uint64_t one_blocks = (uint64_t)rows * blocks_per_row_bytes;
  const uint64_t one_scales = (uint64_t)rows * scales_per_row_bytes;

  *out_blocks_bytes = one_blocks * (uint64_t)n_experts;
  *out_scales_bytes = one_scales * (uint64_t)n_experts;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* Resolve MoE tensors for one layer                                          */
/* ------------------------------------------------------------------------- */

static int ie_resolve_moe_layer(struct ie_gptoss_infer_impl *impl, uint32_t l,
                               ie_gptoss_layer_w_t *LW) {
  if (!impl || !LW || !impl->hp) return -1;

  const uint32_t d_model = impl->hp->d_model;
  const uint32_t d_ff = impl->hp->d_ff;

  ie_tensor_view_t v_router_w, v_router_b;
  ie_tensor_view_t v_gu_blocks, v_gu_scales, v_gu_bias;
  ie_tensor_view_t v_dn_blocks, v_dn_scales, v_dn_bias;

  const char *const router_w_fmts[] = {"model.layers.%u.mlp.router.weight"};
  const char *const router_b_fmts[] = {"model.layers.%u.mlp.router.bias"};

  const char *const gu_blocks_fmts[] = {"model.layers.%u.mlp.experts.gate_up_proj_blocks"};
  const char *const gu_scales_fmts[] = {"model.layers.%u.mlp.experts.gate_up_proj_scales"};
  const char *const gu_bias_fmts[] = {"model.layers.%u.mlp.experts.gate_up_proj_bias"};

  const char *const dn_blocks_fmts[] = {"model.layers.%u.mlp.experts.down_proj_blocks"};
  const char *const dn_scales_fmts[] = {"model.layers.%u.mlp.experts.down_proj_scales"};
  const char *const dn_bias_fmts[] = {"model.layers.%u.mlp.experts.down_proj_bias"};

  if (ie_w_get_layer_fmt_view(impl, l, router_w_fmts, 1, &v_router_w, 1) != 0) {
    IE_LOGE("moe: missing router.weight layer=%u", (unsigned)l);
    return -2;
  }
  (void)ie_w_get_layer_fmt_view(impl, l, router_b_fmts, 1, &v_router_b, 0);

  if (!v_router_w.desc) return -3;

  uint32_t n_experts = 0;
  if (ie_infer_n_experts_from_router(v_router_w.desc, d_model, &n_experts) != 0) {
    char td_buf[256];
    (void)tensor_desc_to_string(v_router_w.desc, td_buf, sizeof(td_buf));
    IE_LOGE("moe: failed to infer n_experts from router.weight layer=%u d_model=%u td=%s",
            (unsigned)l, (unsigned)d_model, td_buf);
    return -4;
  }

  if (v_router_b.desc) {
    if (ie_require_size_eq(v_router_b.desc, (uint64_t)n_experts * 2u) != 0) {
      IE_LOGE("moe: router.bias size mismatch layer=%u n_experts=%u", (unsigned)l, (unsigned)n_experts);
      return -5;
    }
  }

  if (ie_w_get_layer_fmt_view(impl, l, gu_blocks_fmts, 1, &v_gu_blocks, 1) != 0) return -6;
  if (ie_w_get_layer_fmt_view(impl, l, gu_scales_fmts, 1, &v_gu_scales, 1) != 0) return -7;
  (void)ie_w_get_layer_fmt_view(impl, l, gu_bias_fmts, 1, &v_gu_bias, 0);

  if (ie_w_get_layer_fmt_view(impl, l, dn_blocks_fmts, 1, &v_dn_blocks, 1) != 0) return -8;
  if (ie_w_get_layer_fmt_view(impl, l, dn_scales_fmts, 1, &v_dn_scales, 1) != 0) return -9;
  (void)ie_w_get_layer_fmt_view(impl, l, dn_bias_fmts, 1, &v_dn_bias, 0);

  if (!v_gu_blocks.desc || !v_gu_scales.desc || !v_dn_blocks.desc || !v_dn_scales.desc) {
    IE_LOGE("moe: missing required expert tensors layer=%u", (unsigned)l);
    return -10;
  }

  uint8_t gu_scale_bytes = 0u;
  if (v_gu_scales.desc->dtype == TENSOR_DTYPE_BF16) gu_scale_bytes = 2u;
  else if (v_gu_scales.desc->dtype == TENSOR_DTYPE_U8) gu_scale_bytes = 1u;

  if (gu_scale_bytes == 0u) {
    uint64_t tmp_blocks = 0, tmp_scales_1 = 0, tmp_scales_2 = 0;
    if (ie_moe_expected_bytes(n_experts, 2u * d_ff, d_model, 1u, &tmp_blocks, &tmp_scales_1) != 0) return -11;
    if (ie_moe_expected_bytes(n_experts, 2u * d_ff, d_model, 2u, &tmp_blocks, &tmp_scales_2) != 0) return -11;

    if (v_gu_scales.desc->size_bytes == tmp_scales_1) gu_scale_bytes = 1u;
    else if (v_gu_scales.desc->size_bytes == tmp_scales_2) gu_scale_bytes = 2u;
    else {
      IE_LOGE("moe: cannot infer gate_up scale_bytes layer=%u size_bytes=%llu (want %llu or %llu)",
              (unsigned)l,
              (unsigned long long)v_gu_scales.desc->size_bytes,
              (unsigned long long)tmp_scales_1,
              (unsigned long long)tmp_scales_2);
      return -11;
    }
  }

  uint8_t dn_scale_bytes = 0u;
  if (v_dn_scales.desc->dtype == TENSOR_DTYPE_BF16) dn_scale_bytes = 2u;
  else if (v_dn_scales.desc->dtype == TENSOR_DTYPE_U8) dn_scale_bytes = 1u;

  if (dn_scale_bytes == 0u) {
    uint64_t tmp_blocks = 0, tmp_scales_1 = 0, tmp_scales_2 = 0;
    if (ie_moe_expected_bytes(n_experts, d_model, d_ff, 1u, &tmp_blocks, &tmp_scales_1) != 0) return -12;
    if (ie_moe_expected_bytes(n_experts, d_model, d_ff, 2u, &tmp_blocks, &tmp_scales_2) != 0) return -12;

    if (v_dn_scales.desc->size_bytes == tmp_scales_1) dn_scale_bytes = 1u;
    else if (v_dn_scales.desc->size_bytes == tmp_scales_2) dn_scale_bytes = 2u;
    else {
      IE_LOGE("moe: cannot infer down scale_bytes layer=%u size_bytes=%llu (want %llu or %llu)",
              (unsigned)l,
              (unsigned long long)v_dn_scales.desc->size_bytes,
              (unsigned long long)tmp_scales_1,
              (unsigned long long)tmp_scales_2);
      return -12;
    }
  }

  uint64_t want_gu_blocks = 0, want_gu_scales = 0;
  if (ie_moe_expected_bytes(n_experts, 2u * d_ff, d_model, gu_scale_bytes, &want_gu_blocks, &want_gu_scales) != 0)
    return -11;

  uint64_t want_dn_blocks = 0, want_dn_scales = 0;
  if (ie_moe_expected_bytes(n_experts, d_model, d_ff, dn_scale_bytes, &want_dn_blocks, &want_dn_scales) != 0)
    return -12;

  if (ie_require_size_eq(v_gu_blocks.desc, want_gu_blocks) != 0) return -13;
  if (ie_require_size_eq(v_gu_scales.desc, want_gu_scales) != 0) return -14;
  if (ie_require_size_eq(v_dn_blocks.desc, want_dn_blocks) != 0) return -15;
  if (ie_require_size_eq(v_dn_scales.desc, want_dn_scales) != 0) return -16;

  if (v_gu_bias.desc) {
    if (ie_require_size_eq(v_gu_bias.desc,
                           (uint64_t)n_experts * (uint64_t)(2u * d_ff) * 2u) != 0)
      return -17;
  }
  if (v_dn_bias.desc) {
    if (ie_require_size_eq(v_dn_bias.desc, (uint64_t)n_experts * (uint64_t)d_model * 2u) != 0)
      return -18;
  }

  LW->moe.n_experts = n_experts;
  LW->moe.router_w = (const uint16_t *)v_router_w.ptr;
  LW->moe.router_b = (const uint16_t *)v_router_b.ptr;
  LW->moe.router_transposed = 0;

  LW->moe.gate_up_blocks = (const uint8_t *)v_gu_blocks.ptr;
  LW->moe.gate_up_scales = (const uint8_t *)v_gu_scales.ptr;
  LW->moe.gate_up_scale_bytes = gu_scale_bytes;
  LW->moe.gate_up_bias = (const uint16_t *)v_gu_bias.ptr;

  LW->moe.down_blocks = (const uint8_t *)v_dn_blocks.ptr;
  LW->moe.down_scales = (const uint8_t *)v_dn_scales.ptr;
  LW->moe.down_scale_bytes = dn_scale_bytes;
  LW->moe.down_bias = (const uint16_t *)v_dn_bias.ptr;

  LW->moe.gate_up_blocks_stride = (size_t)(v_gu_blocks.desc->size_bytes / n_experts);
  LW->moe.gate_up_scales_stride = (size_t)(v_gu_scales.desc->size_bytes / n_experts);
  LW->moe.gate_up_bias_stride = v_gu_bias.desc ? (size_t)(v_gu_bias.desc->size_bytes / n_experts) : 0u;

  LW->moe.down_blocks_stride = (size_t)(v_dn_blocks.desc->size_bytes / n_experts);
  LW->moe.down_scales_stride = (size_t)(v_dn_scales.desc->size_bytes / n_experts);
  LW->moe.down_bias_stride = v_dn_bias.desc ? (size_t)(v_dn_bias.desc->size_bytes / n_experts) : 0u;

  if (impl->max_experts < n_experts) impl->max_experts = n_experts;

  IE_LOGI("moe: layer=%u n_experts=%u gu_scale_bytes=%u dn_scale_bytes=%u",
          (unsigned)l, (unsigned)n_experts, (unsigned)gu_scale_bytes, (unsigned)dn_scale_bytes);

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Forward pass                                                               */
/* ------------------------------------------------------------------------- */

static int ie_forward_one_token(struct ie_gptoss_infer_impl *impl, ie_kv_cache *kv_layers,
                                uint32_t token_id, uint32_t pos, float *out_logits) {
  if (!impl || !kv_layers || !out_logits) return -1;

  const ie_gptoss_hparams_t *hp = impl->hp;
  if (!hp) return -2;

  const uint32_t d_model = hp->d_model;
  const uint32_t d_head = hp->d_head;
  const uint32_t n_heads = hp->n_heads;
  const uint32_t n_kv_heads = hp->n_kv_heads;
  const uint32_t vocab = hp->vocab_size;
  const uint32_t n_layers = hp->n_layers;

  if (!impl->w_embed_bf16 || !impl->w_norm_bf16 || !impl->w_lm_bf16) return -3;
  if (token_id >= vocab) return -4;
  if (pos >= hp->max_seq) return -5;
  if ((d_head & 1u) != 0u) return -6;

  {
    const uint16_t *emb = impl->w_embed_bf16 + (size_t)token_id * (size_t)d_model;
    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] = ie_bf16_to_f32(emb[i]);
  }

  const float inv_sqrt_d = 1.0f / sqrtf((float)d_head);
  const size_t q_dim = (size_t)n_heads * (size_t)d_head;
  const size_t kv_dim = (size_t)n_kv_heads * (size_t)d_head;
  const size_t d_ff = (size_t)hp->d_ff;

  const float rope_theta_eff = impl->rope_theta * impl->rope_scale;

  for (uint32_t l = 0; l < n_layers; ++l) {
    const ie_gptoss_layer_w_t *W = &impl->layers[l];
    ie_kv_cache *kv = &kv_layers[l];

    if (kv->storage != IE_KV_STORAGE_F32) {
      IE_LOGE("kv storage mismatch: layer=%u storage=%d (want %d)",
              (unsigned)l, (int)kv->storage, (int)IE_KV_STORAGE_F32);
      return -7;
    }
    if ((uint32_t)kv->heads != n_kv_heads) {
      IE_LOGE("kv heads mismatch: layer=%u heads=%zu want=%u",
              (unsigned)l, kv->heads, (unsigned)n_kv_heads);
      return -8;
    }
    if ((uint32_t)kv->head_dim != d_head) {
      IE_LOGE("kv head_dim mismatch: layer=%u head_dim=%zu want=%u",
              (unsigned)l, kv->head_dim, (unsigned)d_head);
      return -9;
    }

    /* Allow KV max_seq to be smaller than hp->max_seq: normal short-context inference. */
    if ((uint32_t)kv->max_seq == 0u) {
      IE_LOGE("kv max_seq invalid: layer=%u max_seq=%zu", (unsigned)l, kv->max_seq);
      return -10;
    }
    if ((uint32_t)kv->max_seq > hp->max_seq) {
      IE_LOGE("kv max_seq exceeds model max_seq: layer=%u max_seq=%zu hp_seq=%u",
              (unsigned)l, kv->max_seq, (unsigned)hp->max_seq);
      return -10;
    }
    if (pos >= (uint32_t)kv->max_seq) {
      IE_LOGE("kv max_seq too small for pos: layer=%u pos=%u max_seq=%zu",
              (unsigned)l, (unsigned)pos, kv->max_seq);
      return -10;
    }

    if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln1_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
      IE_LOGE("rmsnorm1 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
      return -11;
    }

    if (ie_gemv_bf16_f32(W->q_w, impl->x1, impl->q, q_dim, (size_t)d_model, W->q_b) != 0) return -12;
    if (ie_gemv_bf16_f32(W->k_w, impl->x1, impl->k, kv_dim, (size_t)d_model, W->k_b) != 0) return -13;
    if (ie_gemv_bf16_f32(W->v_w, impl->x1, impl->v, kv_dim, (size_t)d_model, W->v_b) != 0) return -14;

    if (ie_rope_apply_f32(impl->q, NULL, (size_t)n_heads, (size_t)d_head, pos, rope_theta_eff) != 0) return -15;
    if (ie_rope_apply_f32(NULL, impl->k, (size_t)n_kv_heads, (size_t)d_head, pos, rope_theta_eff) != 0) return -16;

    if (ie_kv_store_token_f32(kv, (size_t)pos, impl->k, impl->v) != 0) {
      IE_LOGE("kv store failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
      return -17;
    }

    {
      /* KV reuse instrumentation: per layer, this token attends over (pos) cached tokens. */
      ie_kv_add_hits((uint64_t)pos);
      ie_kv_add_misses(1u);

      const float *Kbase = (const float *)kv->K;
      const float *Vbase = (const float *)kv->V;

      if (n_kv_heads == n_heads) {
        if (ie_attn_cpu_causal_f32(impl->q, Kbase, Vbase, (size_t)(pos + 1u),
                                   (size_t)n_heads, (size_t)d_head, inv_sqrt_d,
                                   impl->attn_out, impl->scores) != 0) {
          IE_LOGE("attn kernel failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -18;
        }
      } else {
        ie_attn_causal_gqa_f32(impl->q, Kbase, Vbase, pos + 1u, n_heads, n_kv_heads, d_head,
                               inv_sqrt_d, impl->attn_out, impl->scores);
      }
    }

    if (ie_gemv_bf16_f32(W->o_w, impl->attn_out, impl->x2, (size_t)d_model, q_dim, W->o_b) != 0) return -19;
    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];

    if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln2_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
      IE_LOGE("rmsnorm2 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
      return -20;
    }

    {
      const ie_moe_w_t *M = &W->moe;
      const uint32_t n_exp = M->n_experts;
      if (n_exp == 0u || !impl->router_logits) return -21;

      if (ie_gemv_bf16_f32(M->router_w, impl->x1, impl->router_logits,
                           (size_t)n_exp, (size_t)d_model, M->router_b) != 0) {
        IE_LOGE("router gemv failed: layer=%u pos=%u n_experts=%u",
                (unsigned)l, (unsigned)pos, (unsigned)n_exp);
        return -22;
      }

      const uint32_t topk = ie_u32_min(impl->moe_topk, n_exp);
      if (topk == 0u) return -23;

      uint32_t idx[IE_GPTOSS_MOE_TOPK_DEFAULT] = {0, 0, 0, 0};
      float val[IE_GPTOSS_MOE_TOPK_DEFAULT] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};

      for (uint32_t e = 0; e < n_exp; ++e) {
        const float v = impl->router_logits[e];
        for (uint32_t k = 0; k < topk; ++k) {
          if (v > val[k]) {
            for (uint32_t j = topk - 1u; j > k; --j) {
              val[j] = val[j - 1u];
              idx[j] = idx[j - 1u];
            }
            val[k] = v;
            idx[k] = e;
            break;
          }
        }
      }

      ie_softmax_inplace_f32(val, (size_t)topk);

      for (uint32_t i = 0; i < d_model; ++i) impl->x2[i] = 0.0f;

      for (uint32_t sel = 0; sel < topk; ++sel) {
        const uint32_t ex = idx[sel];
        const float p = val[sel];
        if (!(p > 0.0f)) continue;

        const uint8_t *gu_b = M->gate_up_blocks + ex * M->gate_up_blocks_stride;
        const uint8_t *gu_s = M->gate_up_scales + ex * M->gate_up_scales_stride;
        const uint16_t *gu_bias =
            M->gate_up_bias ? (const uint16_t *)((const uint8_t *)M->gate_up_bias + ex * M->gate_up_bias_stride)
                            : NULL;

        const uint8_t *dn_b = M->down_blocks + ex * M->down_blocks_stride;
        const uint8_t *dn_s = M->down_scales + ex * M->down_scales_stride;
        const uint16_t *dn_bias =
            M->down_bias ? (const uint16_t *)((const uint8_t *)M->down_bias + ex * M->down_bias_stride) : NULL;

        if (ie_gemv_q4_0_f32(gu_b, gu_s, M->gate_up_scale_bytes,
                             impl->x1, impl->mlp_up,
                             2u * d_ff, (size_t)d_model, gu_bias) != 0) {
          IE_LOGE("moe gate_up gemv failed: layer=%u pos=%u ex=%u",
                  (unsigned)l, (unsigned)pos, (unsigned)ex);
          return -24;
        }

        for (size_t i = 0; i < d_ff; ++i) {
          const float g = ie_silu_f32(impl->mlp_up[i]);
          const float u = impl->mlp_up[d_ff + i];
          impl->mlp_gate[i] = g * u;
        }

        if (ie_gemv_q4_0_f32(dn_b, dn_s, M->down_scale_bytes,
                             impl->mlp_gate, impl->attn_out,
                             (size_t)d_model, d_ff, dn_bias) != 0) {
          IE_LOGE("moe down gemv failed: layer=%u pos=%u ex=%u",
                  (unsigned)l, (unsigned)pos, (unsigned)ex);
          return -25;
        }

        for (uint32_t i = 0; i < d_model; ++i) impl->x2[i] += p * impl->attn_out[i];
      }
    }

    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];
  }

  if (ie_rmsnorm_cpu_f32_bf16w(impl->x, impl->w_norm_bf16, (size_t)d_model, impl->rms_eps, impl->x1) != 0)
    return -26;

  if (ie_gemv_bf16_f32(impl->w_lm_bf16, impl->x1, out_logits, (size_t)vocab, (size_t)d_model, NULL) != 0)
    return -27;

  if (getenv("IE_DEBUG_TOPK")) {
    uint32_t K = ie_env_u32_clamped("IE_DEBUG_TOPK", 10u, 1u, 100u);
    for (uint32_t i = 0; i < K; ++i) {
      float maxv = -INFINITY;
      int maxi = -1;
      for (uint32_t j = 0; j < vocab; ++j) {
        if (out_logits[j] > maxv) {
          maxv = out_logits[j];
          maxi = (int)j;
        }
      }
      if (maxi < 0) break;
      fprintf(stderr, "[DBG] top%u id=%d val=%g\n", (unsigned)i, maxi, (double)maxv);
      out_logits[(uint32_t)maxi] = -INFINITY;
    }
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

int ie_gptoss_infer_create(const ie_device_t *dev_const, const ie_weights_t *w,
                           const ie_gptoss_hparams_t *hp, ie_gptoss_infer_t **out_ctx) {
  if (!out_ctx) return -1;
  *out_ctx = NULL;
  if (!w || !hp) return -2;

  if (w->json_path[0] == '\0') {
    IE_LOGE("create: missing w->json_path");
    return -2;
  }
  if (w->weights_path[0] == '\0') {
    IE_LOGE("create: missing w->weights_path");
    return -2;
  }

  IE_LOGI("create: json_path=%s", w->json_path);
  IE_LOGI("create: weights_path=%s", w->weights_path);

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)calloc(1, sizeof(*impl));
  if (!impl) return -3;

  impl->dev = (ie_device_t *)dev_const;
  impl->weights = w;
  impl->hp = hp;
  impl->pos = 0;

  /*
   * NOTE: ie_gptoss_hparams_t does not carry RoPE/RMS fields in this codebase.
   * Keep runtime parameters internal and configurable via environment variables.
   */
  impl->rms_eps = ie_env_f32_clamped("IE_RMS_EPS", IE_GPTOSS_RMS_EPS_DEFAULT, 1.0e-8f, 1.0e-2f);
  impl->rope_theta = ie_env_f32_clamped("IE_ROPE_THETA", IE_GPTOSS_ROPE_THETA_DEFAULT, 1.0f, 1.0e8f);
  impl->rope_scale = ie_env_f32_clamped("IE_ROPE_SCALE", 1.0f, 0.125f, 128.0f);

  /* Step B: match model routing behavior (top-4 experts). Do not allow overrides here. */
  impl->moe_topk = IE_GPTOSS_MOE_TOPK_DEFAULT;

  impl->max_experts = 0;

  char *model_dir = ie_dirname_alloc(w->json_path);
  if (!model_dir) {
    free(impl);
    return -4;
  }
  IE_LOGD("create: inferred model_dir=%s", model_dir);

  const char *const tmap_rel[] = {
      "tensor_map.json",
      "dedup_out/tensor_map.json",
      "hf/original/tensor_map.json",
  };

  char *tmap_path = NULL;
  int tmap_ok = -1;
  memset(&impl->tmap, 0, sizeof(impl->tmap));
  impl->tmap_path_used = NULL;

  for (size_t i = 0; i < (sizeof(tmap_rel) / sizeof(tmap_rel[0])); ++i) {
    free(tmap_path);
    tmap_path = ie_path_join_alloc(model_dir, tmap_rel[i]);
    if (!tmap_path) continue;

    IE_LOGI("create: trying tensor_map=%s", tmap_path);

    if (tensor_map_load(tmap_path, &impl->tmap) == 0) {
      tmap_ok = 0;
      impl->tmap_path_used = tmap_path;
      tmap_path = NULL;
      IE_LOGI("create: loaded tensor_map=%s (count=%u)",
              impl->tmap_path_used, (unsigned)impl->tmap.count);
      break;
    }

    IE_LOGW("create: failed to load tensor_map=%s", tmap_path);

    tensor_map_free(&impl->tmap);
    memset(&impl->tmap, 0, sizeof(impl->tmap));
  }

  free(tmap_path);

  if (tmap_ok != 0) {
    IE_LOGE("create: failed to load tensor_map.json (model_dir=%s)", model_dir);
    IE_LOGE("create: tried: %s/tensor_map.json", model_dir);
    IE_LOGE("create: tried: %s/dedup_out/tensor_map.json", model_dir);
    IE_LOGE("create: tried: %s/hf/original/tensor_map.json", model_dir);
    free(model_dir);
    free(impl);
    return -5;
  }

  {
    int rc_mm = ie_mmap_ro(w->weights_path, &impl->bin_base, &impl->bin_size);
    if (rc_mm != 0) {
      IE_LOGE("create: failed to mmap weights file: path=%s rc=%d", w->weights_path, rc_mm);
      tensor_map_free(&impl->tmap);
      free(impl->tmap_path_used);
      free(model_dir);
      free(impl);
      return -6;
    }
    IE_LOGI("create: mapped model.ie.bin size_bytes=%llu",
            (unsigned long long)impl->bin_size);
  }

  free(model_dir);

  const uint32_t d_model = hp->d_model;
  const uint32_t vocab = hp->vocab_size;
  const uint32_t n_layers = hp->n_layers;
  const uint32_t d_head = hp->d_head;
  const uint32_t n_heads = hp->n_heads;
  const uint32_t n_kv_heads = hp->n_kv_heads;
  const uint32_t d_ff = hp->d_ff;

  IE_LOGI("create: hp n_layers=%u d_model=%u d_head=%u n_heads=%u n_kv_heads=%u d_ff=%u vocab=%u max_seq=%u",
          (unsigned)n_layers, (unsigned)d_model, (unsigned)d_head, (unsigned)n_heads,
          (unsigned)n_kv_heads, (unsigned)d_ff, (unsigned)vocab, (unsigned)hp->max_seq);

  IE_LOGI("create: rms_eps=%g rope_theta=%g rope_scale=%g moe_topk=%u",
          (double)impl->rms_eps, (double)impl->rope_theta, (double)impl->rope_scale,
          (unsigned)impl->moe_topk);

  {
    ie_tensor_view_t v;
    const char *const embed_candidates[] = {
      "model.embed_tokens.weight",
      "transformer.wte.weight",
      "tok_embeddings.weight"
    };
    if (ie_w_get_first_view(impl, embed_candidates, 3, &v) != 0) {
      IE_LOGE("create: missing embedding tensor (tried common candidates)");
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -7;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)vocab * (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: embedding size check failed: want=%llu",
              (unsigned long long)((uint64_t)vocab * (uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -8;
    }
    impl->w_embed_bf16 = (const uint16_t *)v.ptr;
  }

  {
    ie_tensor_view_t v;
    const char *const norm_candidates[] = {
      "model.norm.weight",
      "transformer.ln_f.weight",
      "norm.weight"
    };
    if (ie_w_get_first_view(impl, norm_candidates, 3, &v) != 0) {
      IE_LOGE("create: missing final norm tensor (tried common candidates)");
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -9;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: final norm size check failed: want=%llu",
              (unsigned long long)((uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -10;
    }
    impl->w_norm_bf16 = (const uint16_t *)v.ptr;
  }

  {
    ie_tensor_view_t v;
    const char *const lm_candidates[] = {
      "lm_head.weight",
      "model.lm_head.weight",
      "transformer.lm_head.weight",
      "model.embed_tokens.weight"
    };
    if (ie_w_get_first_view(impl, lm_candidates, 4, &v) != 0) {
      IE_LOGE("create: missing lm_head tensor (tried common candidates)");
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -11;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)vocab * (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: lm_head size check failed: want=%llu",
              (unsigned long long)((uint64_t)vocab * (uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -12;
    }
    impl->w_lm_bf16 = (const uint16_t *)v.ptr;
  }

  impl->layers = (ie_gptoss_layer_w_t *)calloc((size_t)n_layers, sizeof(*impl->layers));
  if (!impl->layers) {
    IE_LOGE("create: failed to allocate layer table (n_layers=%u)", (unsigned)n_layers);
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -13;
  }

  const uint64_t q_dim = (uint64_t)n_heads * (uint64_t)d_head;
  const uint64_t kv_dim = (uint64_t)n_kv_heads * (uint64_t)d_head;

  for (uint32_t l = 0; l < n_layers; ++l) {
    ie_gptoss_layer_w_t *LW = &impl->layers[l];
    ie_tensor_view_t v;

    IE_LOGD("create: resolving layer=%u", (unsigned)l);

    const char *const ln1_fmts[] = {
      "model.layers.%u.input_layernorm.weight",
      "model.layers.%u.attention_norm.weight"
    };
    if (ie_w_get_layer_fmt_view(impl, l, ln1_fmts, 2, &v, 1) != 0) {
      IE_LOGE("create: missing ln1 for layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -14;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: ln1 size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)((uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -15;
    }
    LW->ln1_w = (const uint16_t *)v.ptr;

    const char *const q_w_fmts[] = {"model.layers.%u.self_attn.q_proj.weight"};
    const char *const q_b_fmts[] = {"model.layers.%u.self_attn.q_proj.bias"};
    if (ie_w_get_layer_fmt_view(impl, l, q_w_fmts, 1, &v, 1) != 0) {
      IE_LOGE("create: missing q_proj.weight layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -16;
    }
    if (!v.desc || ie_require_size_eq(v.desc, q_dim * (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: q_proj.weight size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)(q_dim * (uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -17;
    }
    LW->q_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, q_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, q_dim * 2u) != 0) {
        IE_LOGE("create: q_proj.bias size mismatch layer=%u want=%llu",
                (unsigned)l, (unsigned long long)(q_dim * 2u));
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -18;
      }
      LW->q_b = (const uint16_t *)v.ptr;
    } else {
      LW->q_b = NULL;
    }

    const char *const k_w_fmts[] = {"model.layers.%u.self_attn.k_proj.weight"};
    const char *const k_b_fmts[] = {"model.layers.%u.self_attn.k_proj.bias"};
    if (ie_w_get_layer_fmt_view(impl, l, k_w_fmts, 1, &v, 1) != 0) {
      IE_LOGE("create: missing k_proj.weight layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -19;
    }
    if (!v.desc || ie_require_size_eq(v.desc, kv_dim * (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: k_proj.weight size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)(kv_dim * (uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -20;
    }
    LW->k_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, k_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, kv_dim * 2u) != 0) {
        IE_LOGE("create: k_proj.bias size mismatch layer=%u want=%llu",
                (unsigned)l, (unsigned long long)(kv_dim * 2u));
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -21;
      }
      LW->k_b = (const uint16_t *)v.ptr;
    } else {
      LW->k_b = NULL;
    }

    const char *const v_w_fmts[] = {"model.layers.%u.self_attn.v_proj.weight"};
    const char *const v_b_fmts[] = {"model.layers.%u.self_attn.v_proj.bias"};
    if (ie_w_get_layer_fmt_view(impl, l, v_w_fmts, 1, &v, 1) != 0) {
      IE_LOGE("create: missing v_proj.weight layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -22;
    }
    if (!v.desc || ie_require_size_eq(v.desc, kv_dim * (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: v_proj.weight size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)(kv_dim * (uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -23;
    }
    LW->v_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, v_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, kv_dim * 2u) != 0) {
        IE_LOGE("create: v_proj.bias size mismatch layer=%u want=%llu",
                (unsigned)l, (unsigned long long)(kv_dim * 2u));
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -24;
      }
      LW->v_b = (const uint16_t *)v.ptr;
    } else {
      LW->v_b = NULL;
    }

    const char *const o_w_fmts[] = {"model.layers.%u.self_attn.o_proj.weight"};
    const char *const o_b_fmts[] = {"model.layers.%u.self_attn.o_proj.bias"};
    if (ie_w_get_layer_fmt_view(impl, l, o_w_fmts, 1, &v, 1) != 0) {
      IE_LOGE("create: missing o_proj.weight layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -25;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * q_dim * 2u) != 0) {
      IE_LOGE("create: o_proj.weight size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)((uint64_t)d_model * q_dim * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -26;
    }
    LW->o_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, o_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
        IE_LOGE("create: o_proj.bias size mismatch layer=%u want=%llu",
                (unsigned)l, (unsigned long long)((uint64_t)d_model * 2u));
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
      LW->o_b = (const uint16_t *)v.ptr;
    } else {
      LW->o_b = NULL;
    }

    const char *const ln2_fmts[] = {
      "model.layers.%u.post_attention_layernorm.weight",
      "model.layers.%u.ffn_norm.weight"
    };
    if (ie_w_get_layer_fmt_view(impl, l, ln2_fmts, 2, &v, 1) != 0) {
      IE_LOGE("create: missing ln2 for layer=%u", (unsigned)l);
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -28;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      IE_LOGE("create: ln2 size mismatch layer=%u want=%llu",
              (unsigned)l, (unsigned long long)((uint64_t)d_model * 2u));
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -29;
    }
    LW->ln2_w = (const uint16_t *)v.ptr;

    {
      const int rc_moe = ie_resolve_moe_layer(impl, l, LW);
      if (rc_moe != 0) {
        IE_LOGE("create: moe resolve failed: layer=%u rc=%d d_model=%u d_ff=%u",
                (unsigned)l, rc_moe, (unsigned)d_model, (unsigned)d_ff);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -30;
      }
    }
  }

  const size_t d_model_sz = (size_t)d_model;
  const size_t q_dim_sz = (size_t)n_heads * (size_t)d_head;
  const size_t kv_dim_sz = (size_t)n_kv_heads * (size_t)d_head;
  const size_t d_ff_sz = (size_t)d_ff;

  impl->x = (float *)malloc(d_model_sz * sizeof(float));
  impl->x1 = (float *)malloc(d_model_sz * sizeof(float));
  impl->x2 = (float *)malloc(d_model_sz * sizeof(float));
  impl->q = (float *)malloc(q_dim_sz * sizeof(float));
  impl->k = (float *)malloc(kv_dim_sz * sizeof(float));
  impl->v = (float *)malloc(kv_dim_sz * sizeof(float));
  impl->attn_out = (float *)malloc(q_dim_sz * sizeof(float));
  impl->mlp_gate = (float *)malloc(d_ff_sz * sizeof(float));
  impl->mlp_up = (float *)malloc((2u * d_ff_sz) * sizeof(float));
  impl->scores = (float *)malloc((size_t)hp->max_seq * sizeof(float));
  impl->router_logits = (impl->max_experts > 0u) ? (float *)malloc((size_t)impl->max_experts * sizeof(float)) : NULL;

  if (!impl->x || !impl->x1 || !impl->x2 || !impl->q || !impl->k || !impl->v || !impl->attn_out ||
      !impl->mlp_gate || !impl->mlp_up || !impl->scores || (impl->max_experts > 0u && !impl->router_logits)) {
    IE_LOGE("create: activation allocation failed (max_experts=%u)", (unsigned)impl->max_experts);
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -31;
  }

  IE_LOGI("create: success");
  *out_ctx = (ie_gptoss_infer_t *)impl;
  return 0;
}

void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx) {
  if (!ctx) return;
  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  if (impl->bin_base) ie_munmap_ro(impl->bin_base, impl->bin_size);
  tensor_map_free(&impl->tmap);

  free(impl->tmap_path_used);
  free(impl->layers);

  free(impl->x);
  free(impl->x1);
  free(impl->x2);
  free(impl->q);
  free(impl->k);
  free(impl->v);
  free(impl->attn_out);
  free(impl->mlp_gate);
  free(impl->mlp_up);
  free(impl->scores);
  free(impl->router_logits);

  free(impl);
}

int ie_gptoss_infer_prefill(ie_gptoss_infer_t *ctx, ie_kv_cache *kv, const uint32_t *prompt,
                           uint32_t n_prompt, float *out_logits) {
  if (!ctx || !kv || (!prompt && n_prompt > 0) || !out_logits) return -1;

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  impl->pos = 0;
  if (n_prompt == 0u) return -2;

  const uint32_t limit = (uint32_t)ie_min_size((size_t)n_prompt, (size_t)impl->hp->max_seq);
  if (limit != n_prompt) {
    IE_LOGW("prefill: prompt truncated from %u to %u due to max_seq=%u",
            (unsigned)n_prompt, (unsigned)limit, (unsigned)impl->hp->max_seq);
  }

  for (uint32_t i = 0; i < limit; ++i) {
    const int rc = ie_forward_one_token(impl, kv, prompt[i], impl->pos, out_logits);
    if (rc != 0) {
      IE_LOGE("prefill: failed at i=%u token=%u pos=%u rc=%d",
              (unsigned)i, (unsigned)prompt[i], (unsigned)impl->pos, rc);
      return rc;
    }
    impl->pos += 1u;
  }

  return 0;
}

int ie_gptoss_infer_step(ie_gptoss_infer_t *ctx, ie_kv_cache *kv, uint32_t token_id,
                         float *out_logits) {
  if (!ctx || !kv || !out_logits) return -1;

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  const uint32_t pos = impl->pos;
  const int rc = ie_forward_one_token(impl, kv, token_id, pos, out_logits);
  if (rc != 0) {
    IE_LOGE("step: failed token=%u pos=%u rc=%d", (unsigned)token_id, (unsigned)pos, rc);
    return rc;
  }

  impl->pos = pos + 1u;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* Cursor control                                                             */
/* ------------------------------------------------------------------------- */

void ie_gptoss_infer_reset(ie_gptoss_infer_t *ctx) {
  if (!ctx) return;
  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;
  impl->pos = 0;
}

int ie_gptoss_infer_seek(ie_gptoss_infer_t *ctx, uint32_t pos) {
  if (!ctx) return -1;
  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;
  if (!impl->hp) return -2;

  if (pos > impl->hp->max_seq) {
    IE_LOGE("seek: invalid pos=%u (max_seq=%u)", (unsigned)pos, (unsigned)impl->hp->max_seq);
    return -3;
  }
  if (pos == impl->hp->max_seq) {
    IE_LOGW("seek: pos==max_seq (%u); next step/prefill will fail if it writes at this position",
            (unsigned)pos);
  }

  impl->pos = pos;
  return 0;
}

uint32_t ie_gptoss_infer_pos(const ie_gptoss_infer_t *ctx) {
  if (!ctx) return 0u;
  const struct ie_gptoss_infer_impl *impl = (const struct ie_gptoss_infer_impl *)ctx;
  return impl->pos;
}

/* Backward-compatible aliases (if older call-sites exist). */

uint32_t ie_gptoss_infer_get_pos(const ie_gptoss_infer_t *ctx) {
  return ie_gptoss_infer_pos(ctx);
}

void ie_gptoss_infer_set_pos(ie_gptoss_infer_t *ctx, uint32_t pos) {
  (void)ie_gptoss_infer_seek(ctx, pos);
}
