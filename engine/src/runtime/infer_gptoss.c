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
#include <limits.h>
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
#if defined(IE_CUDA_AVAILABLE)
#include "ie_attn_cuda.h"
#include "ie_elem_cuda.h"
#include "ie_device_cuda.h"
#include "ie_rmsnorm_cuda.h"
#endif
#include "ie_infer.h"
#include "ie_kernels.h"
#include "ie_kv_cache.h"
#include "ie_kv_instrumentation.h"
#include "ie_rope.h"
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

  /** @brief Router weight (FP32), shape [n_experts, d_model], optional. */
  float *router_w_f32;

  /** @brief Router bias (FP32), shape [n_experts], optional. */
  float *router_b_f32;

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

  /** @brief RMSNorm #1 weight (FP32), shape [d_model], optional. */
  float *ln1_w_f32;

  /** @brief RMSNorm #2 weight (BF16), shape [d_model]. */
  const uint16_t *ln2_w;

  /** @brief RMSNorm #2 weight (FP32), shape [d_model], optional. */
  float *ln2_w_f32;

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

  /** @brief Q projection weight blocks (Q4_0), optional. */
  const uint8_t *q_q4_blocks;

  /** @brief Q projection scales (Q4_0), optional. */
  const uint8_t *q_q4_scales;

  /** @brief Q projection scale bytes (1 or 2). */
  uint8_t q_q4_scale_bytes;

  /** @brief K projection weight blocks (Q4_0), optional. */
  const uint8_t *k_q4_blocks;

  /** @brief K projection scales (Q4_0), optional. */
  const uint8_t *k_q4_scales;

  /** @brief K projection scale bytes (1 or 2). */
  uint8_t k_q4_scale_bytes;

  /** @brief V projection weight blocks (Q4_0), optional. */
  const uint8_t *v_q4_blocks;

  /** @brief V projection scales (Q4_0), optional. */
  const uint8_t *v_q4_scales;

  /** @brief V projection scale bytes (1 or 2). */
  uint8_t v_q4_scale_bytes;

  /** @brief O projection weight blocks (Q4_0), optional. */
  const uint8_t *o_q4_blocks;

  /** @brief O projection scales (Q4_0), optional. */
  const uint8_t *o_q4_scales;

  /** @brief O projection scale bytes (1 or 2). */
  uint8_t o_q4_scale_bytes;

  /** @brief Q projection weight (FP32), shape [q_dim, d_model], optional. */
  float *q_w_f32;

  /** @brief Q projection bias (FP32), shape [q_dim], optional. */
  float *q_b_f32;

  /** @brief K projection weight (FP32), shape [kv_dim, d_model], optional. */
  float *k_w_f32;

  /** @brief K projection bias (FP32), shape [kv_dim], optional. */
  float *k_b_f32;

  /** @brief V projection weight (FP32), shape [kv_dim, d_model], optional. */
  float *v_w_f32;

  /** @brief V projection bias (FP32), shape [kv_dim], optional. */
  float *v_b_f32;

  /** @brief O projection weight (FP32), shape [d_model, q_dim], optional. */
  float *o_w_f32;

  /** @brief O projection bias (FP32), shape [d_model], optional. */
  float *o_b_f32;

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

  /** @brief Final norm weight (FP32), shape [d_model], optional. */
  float *w_norm_f32;

  /** @brief LM head weight (BF16), shape [vocab, d_model]. */
  const uint16_t *w_lm_bf16;

  /** @brief LM head Q4 blocks (optional), shape [vocab, d_model]. */
  const uint8_t *w_lm_q4_blocks;

  /** @brief LM head Q4 scales (optional), shape [vocab, d_model]. */
  const uint8_t *w_lm_q4_scales;

  /** @brief Bytes per LM head scale element (2 = BF16, 1 = FP8 stored as u8). */
  uint8_t w_lm_q4_scale_bytes;

  /** @brief LM head weight (FP32), shape [vocab, d_model], optional. */
  float *w_lm_f32;

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

  /** @brief Enable CUDA attention path (device + env controlled). */
  int use_cuda_attn;

  /** @brief Enable full CUDA path (device activations). */
  int use_cuda_full;

  /** @brief Device Q buffer for CUDA attention. */
  float *d_q;

  /** @brief Device K buffer for current token. */
  float *d_k;

  /** @brief Device V buffer for current token. */
  float *d_v;

  /** @brief Device attention output buffer. */
  float *d_attn_out;

  /** @brief Device activations. */
  float *d_x;
  float *d_x1;
  float *d_x2;

  /** @brief Device MLP buffers. */
  float *d_mlp_gate;
  float *d_mlp_up;

  /** @brief Device RMSNorm weights (per-layer). */
  float **d_ln1_w;
  float **d_ln2_w;

  /** @brief Device final RMSNorm weight. */
  float *d_norm_w;

  /** @brief Device logits buffer. */
  float *d_logits;

  /** @brief Per-layer device KV cache (K). */
  float **d_kv_K;

  /** @brief Per-layer device KV cache (V). */
  float **d_kv_V;

  /** @brief CUDA attention calls (successful). */
  uint64_t cuda_attn_calls;

  /** @brief CUDA attention fallbacks (errors). */
  uint64_t cuda_attn_fallbacks;

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

  /** @brief Whether to use FP32 cached attention weights. */
  int use_f32_attn;

  /** @brief Whether to use FP32 cached router weights. */
  int use_f32_router;

  /** @brief Whether to use FP32 cached LM head weights. */
  int use_f32_lm_head;

  /** @brief Whether to use FP32 cached RMSNorm weights. */
  int use_f32_rmsnorm;
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

/**
 * @brief Parse an environment variable as boolean.
 * @param name Environment variable name.
 * @return 1 if enabled, 0 otherwise.
 */
static int ie_env_flag(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0;
  if (s[0] == '0') return 0;
  if (s[0] == 'f' || s[0] == 'F') return 0;
  if (s[0] == 'n' || s[0] == 'N') return 0;
  return 1;
}

/**
 * @brief Allocate and convert BF16 weights to FP32.
 * @param src BF16 source buffer.
 * @param n Number of elements.
 * @param out Receives allocated FP32 buffer (or NULL if src/n is empty).
 * @return 0 on success, negative on failure.
 */
static int ie_alloc_f32_from_bf16(const uint16_t *src, size_t n, float **out) {
  if (!out) return -1;
  *out = NULL;
  if (!src || n == 0) return 0;
  float *buf = (float *)malloc(n * sizeof(float));
  if (!buf) return -2;
  ie_vec_bf16_to_f32(src, buf, n);
  *out = buf;
  return 0;
}

static void *ie_aligned_alloc_bytes(size_t align, size_t nbytes) {
  void *p = NULL;
  if (align < sizeof(void *)) align = sizeof(void *);
  if (posix_memalign(&p, align, nbytes) != 0) {
    return NULL;
  }
  return p;
}

static void *ie_aligned_or_malloc(size_t align, size_t nbytes) {
  void *p = ie_aligned_alloc_bytes(align, nbytes);
  if (p) return p;
  return malloc(nbytes);
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

static int ie_debug_nan_enabled_(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *s = getenv("IE_DEBUG_NAN");
    cached = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
  }
  return cached;
}

static int ie_trace_bf16_enabled_(void) {
  static int cached = -1;
  if (cached < 0) {
    const char *s = getenv("IE_TRACE_BF16");
    cached = (s && s[0] && strcmp(s, "0") != 0) ? 1 : 0;
  }
  return cached;
}

static int ie_check_finite_f32_(const float *v, size_t n, const char *tag,
                                uint32_t layer, uint32_t pos) {
  if (!v || !tag || n == 0) return 0;
  for (size_t i = 0; i < n; ++i) {
    if (!isfinite(v[i])) {
      IE_LOGE("nan: %s layer=%u pos=%u idx=%zu val=%g",
              tag, (unsigned)layer, (unsigned)pos, i, (double)v[i]);
      return -1;
    }
  }
  return 0;
}

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

static int ie_gemv_q4_0_f32_dispatch(struct ie_gptoss_infer_impl *impl,
                                     const uint8_t *w_q4, const uint8_t *w_scales,
                                     size_t scale_bytes, const float *x, float *y,
                                     size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (impl && impl->dev) {
    return ie_device_gemv_q4_0_f32(impl->dev, w_q4, w_scales, scale_bytes,
                                   x, y, rows, cols, bias_bf16);
  }
  return ie_gemv_q4_0_f32(w_q4, w_scales, scale_bytes, x, y, rows, cols, bias_bf16);
}

static int ie_gemv_q4_0_f32_dispatch_ex(struct ie_gptoss_infer_impl *impl,
                                       const uint8_t *w_q4, const uint8_t *w_scales,
                                       size_t scale_bytes, int scale_fmt,
                                       const float *x, float *y,
                                       size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (impl && impl->dev) {
    return ie_device_gemv_q4_0_f32(impl->dev, w_q4, w_scales, scale_bytes,
                                   x, y, rows, cols, bias_bf16);
  }
  return ie_gemv_q4_0_f32_ex(w_q4, w_scales, scale_bytes, scale_fmt, x, y, rows, cols, bias_bf16);
}

#if defined(IE_CUDA_AVAILABLE)
static int ie_gemv_q4_0_f32_device_ex(struct ie_gptoss_infer_impl *impl,
                                     const uint8_t *w_q4, const uint8_t *w_scales,
                                     size_t scale_bytes, int scale_fmt,
                                     const float *dx, float *dy,
                                     size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (!impl || !impl->dev || !w_q4 || !w_scales || !dx || !dy || rows == 0 || cols == 0) return -1;
  if (scale_bytes != 1u && scale_bytes != 2u) return -2;
  if (scale_fmt != 0 && scale_fmt != 1) return -2;

  const size_t blocks_per_row = (cols + 31u) / 32u;
  const size_t row_w_bytes = blocks_per_row * 16u;
  const size_t row_s_bytes = blocks_per_row * (size_t)scale_bytes;
  const size_t W_bytes = rows * row_w_bytes;
  const size_t S_bytes = rows * row_s_bytes;

  const uint8_t *dW = NULL;
  const uint8_t *dS = NULL;
  if (ie_device_q4_map(impl->dev, w_q4, W_bytes, w_scales, S_bytes, &dW, &dS) != 0) {
    /* Fallback: allocate temp device buffers and copy weights/scales. */
    uint8_t *tmpW = (uint8_t *)ie_cuda_malloc(W_bytes);
    uint8_t *tmpS = (uint8_t *)ie_cuda_malloc(S_bytes);
    if (!tmpW || !tmpS) {
      if (tmpW) ie_cuda_free(tmpW);
      if (tmpS) ie_cuda_free(tmpS);
      return -3;
    }
    if (ie_cuda_memcpy(tmpW, w_q4, W_bytes, IE_CUDA_COPY_H2D) != 0 ||
        ie_cuda_memcpy(tmpS, w_scales, S_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_cuda_free(tmpW);
      ie_cuda_free(tmpS);
      return -4;
    }
    const uint16_t *dbias = NULL;
    uint16_t *tmpB = NULL;
    if (bias_bf16) {
      const size_t b_bytes = rows * sizeof(uint16_t);
      tmpB = (uint16_t *)ie_cuda_malloc(b_bytes);
      if (!tmpB || ie_cuda_memcpy(tmpB, bias_bf16, b_bytes, IE_CUDA_COPY_H2D) != 0) {
        if (tmpB) ie_cuda_free(tmpB);
        ie_cuda_free(tmpW);
        ie_cuda_free(tmpS);
        return -5;
      }
      dbias = (const uint16_t *)tmpB;
    }
    const int rc = ie_cuda_gemv_q4_0_f32_ex(tmpW, tmpS, scale_bytes, scale_fmt, dx, dy, rows, cols, dbias);
    if (tmpB) ie_cuda_free(tmpB);
    ie_cuda_free(tmpW);
    ie_cuda_free(tmpS);
    return rc == 0 ? 0 : -6;
  }

  const uint16_t *dbias = NULL;
  uint16_t *tmpB = NULL;
  if (bias_bf16) {
    const size_t b_bytes = rows * sizeof(uint16_t);
    tmpB = (uint16_t *)ie_cuda_malloc(b_bytes);
    if (!tmpB || ie_cuda_memcpy(tmpB, bias_bf16, b_bytes, IE_CUDA_COPY_H2D) != 0) {
      if (tmpB) ie_cuda_free(tmpB);
      return -7;
    }
    dbias = (const uint16_t *)tmpB;
  }

  const int rc = ie_cuda_gemv_q4_0_f32_ex(dW, dS, scale_bytes, scale_fmt, dx, dy, rows, cols, dbias);
  if (tmpB) ie_cuda_free(tmpB);
  return rc == 0 ? 0 : -8;
}

static int ie_gemv_q4_0_f32_device(struct ie_gptoss_infer_impl *impl,
                                  const uint8_t *w_q4, const uint8_t *w_scales,
                                  size_t scale_bytes, const float *dx, float *dy,
                                  size_t rows, size_t cols, const uint16_t *bias_bf16) {
  return ie_gemv_q4_0_f32_device_ex(impl, w_q4, w_scales, scale_bytes, 0, dx, dy, rows, cols, bias_bf16);
}

static int ie_gemv_bf16_f32_device(struct ie_gptoss_infer_impl *impl,
                                   const uint16_t *w_bf16, const float *dx, float *dy,
                                   size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (!impl || !impl->dev || !w_bf16 || !dx || !dy || rows == 0 || cols == 0) return -1;

  const size_t W_bytes = rows * cols * sizeof(uint16_t);
  const void *dW = NULL;
  if (ie_device_blob_ptr(impl->dev, w_bf16, W_bytes, &dW) != 0) {
    uint16_t *tmpW = (uint16_t *)ie_cuda_malloc(W_bytes);
    if (!tmpW) return -2;
    if (ie_cuda_memcpy(tmpW, w_bf16, W_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_cuda_free(tmpW);
      return -3;
    }
    const uint16_t *dbias = NULL;
    uint16_t *tmpB = NULL;
    if (bias_bf16) {
      const size_t b_bytes = rows * sizeof(uint16_t);
      tmpB = (uint16_t *)ie_cuda_malloc(b_bytes);
      if (!tmpB || ie_cuda_memcpy(tmpB, bias_bf16, b_bytes, IE_CUDA_COPY_H2D) != 0) {
        if (tmpB) ie_cuda_free(tmpB);
        ie_cuda_free(tmpW);
        return -4;
      }
      dbias = (const uint16_t *)tmpB;
    }
    const int rc = ie_cuda_gemv_bf16_f32(tmpW, dx, dy, rows, cols, dbias);
    if (tmpB) ie_cuda_free(tmpB);
    ie_cuda_free(tmpW);
    return rc == 0 ? 0 : -5;
  }

  const uint16_t *dbias = NULL;
  uint16_t *tmpB = NULL;
  if (bias_bf16) {
    const size_t b_bytes = rows * sizeof(uint16_t);
    tmpB = (uint16_t *)ie_cuda_malloc(b_bytes);
    if (!tmpB || ie_cuda_memcpy(tmpB, bias_bf16, b_bytes, IE_CUDA_COPY_H2D) != 0) {
      if (tmpB) ie_cuda_free(tmpB);
      return -6;
    }
    dbias = (const uint16_t *)tmpB;
  }

  const int rc = ie_cuda_gemv_bf16_f32((const uint16_t *)dW, dx, dy, rows, cols, dbias);
  if (tmpB) ie_cuda_free(tmpB);
  return rc == 0 ? 0 : -7;
}

static int ie_gemv_f32_device(struct ie_gptoss_infer_impl *impl,
                              const float *w_f32, const float *dx, float *dy,
                              size_t rows, size_t cols, const float *bias_f32) {
  if (!impl || !impl->dev || !w_f32 || !dx || !dy || rows == 0 || cols == 0) return -1;

  const size_t W_bytes = rows * cols * sizeof(float);
  const void *dW = NULL;
  if (ie_device_blob_ptr(impl->dev, w_f32, W_bytes, &dW) != 0) {
    float *tmpW = (float *)ie_cuda_malloc(W_bytes);
    if (!tmpW) return -2;
    if (ie_cuda_memcpy(tmpW, w_f32, W_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_cuda_free(tmpW);
      return -3;
    }
    const float *dbias = NULL;
    float *tmpB = NULL;
    if (bias_f32) {
      const size_t b_bytes = rows * sizeof(float);
      tmpB = (float *)ie_cuda_malloc(b_bytes);
      if (!tmpB || ie_cuda_memcpy(tmpB, bias_f32, b_bytes, IE_CUDA_COPY_H2D) != 0) {
        if (tmpB) ie_cuda_free(tmpB);
        ie_cuda_free(tmpW);
        return -4;
      }
      dbias = (const float *)tmpB;
    }
    const int rc = ie_cuda_gemv_f32(tmpW, dx, dy, rows, cols, dbias);
    if (tmpB) ie_cuda_free(tmpB);
    ie_cuda_free(tmpW);
    return rc == 0 ? 0 : -5;
  }

  const float *dbias = NULL;
  float *tmpB = NULL;
  if (bias_f32) {
    const size_t b_bytes = rows * sizeof(float);
    tmpB = (float *)ie_cuda_malloc(b_bytes);
    if (!tmpB || ie_cuda_memcpy(tmpB, bias_f32, b_bytes, IE_CUDA_COPY_H2D) != 0) {
      if (tmpB) ie_cuda_free(tmpB);
      return -6;
    }
    dbias = (const float *)tmpB;
  }

  const int rc = ie_cuda_gemv_f32((const float *)dW, dx, dy, rows, cols, dbias);
  if (tmpB) ie_cuda_free(tmpB);
  return rc == 0 ? 0 : -7;
}

static int ie_cuda_debug_check_finite(struct ie_gptoss_infer_impl *impl,
                                      const float *dptr, size_t n,
                                      const char *tag, uint32_t layer, uint32_t pos) {
  if (!impl || !dptr || n == 0 || !tag) return 0;
  if (!ie_env_flag("IE_CUDA_DEBUG_NAN")) return 0;

  const size_t k = (n < 16u) ? n : 16u;
  float tmp[16];
  if (ie_cuda_memcpy(tmp, dptr, k * sizeof(float), IE_CUDA_COPY_D2H) != 0) {
    IE_LOGE("cuda_full: debug memcpy failed tag=%s layer=%u pos=%u", tag, (unsigned)layer, (unsigned)pos);
    return -1;
  }
  for (size_t i = 0; i < k; ++i) {
    if (!isfinite(tmp[i])) {
      IE_LOGE("cuda_full: non-finite %s[%zu]=%g layer=%u pos=%u", tag, i, (double)tmp[i],
              (unsigned)layer, (unsigned)pos);
      return -1;
    }
  }
  return 0;
}
#endif

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

static void ie_cuda_debug_stats(struct ie_gptoss_infer_impl *impl,
                                const float *dptr, size_t n,
                                const char *tag, uint32_t layer, uint32_t pos) {
  if (!impl || !dptr || n == 0 || !tag) return;
  if (!ie_env_flag("IE_CUDA_DEBUG_STATS")) return;

  size_t n_sample = n;
  if (n_sample > 2048u) n_sample = 2048u;

  float *tmp = (float *)malloc(n_sample * sizeof(float));
  if (!tmp) return;
  if (ie_cuda_memcpy(tmp, dptr, n_sample * sizeof(float), IE_CUDA_COPY_D2H) != 0) {
    free(tmp);
    return;
  }

  float vmin = tmp[0];
  float vmax = tmp[0];
  double sum = 0.0;
  for (size_t i = 0; i < n_sample; ++i) {
    const float v = tmp[i];
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
    sum += (double)v;
  }
  const double mean = sum / (double)n_sample;
  IE_LOGI("cuda_full: stats %s sample=%zu min=%.6g max=%.6g mean=%.6g layer=%u pos=%u",
          tag, (unsigned long)n_sample, (double)vmin, (double)vmax, mean,
          (unsigned)layer, (unsigned)pos);
  free(tmp);
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

static int ie_q4_expected_bytes(uint32_t rows, uint32_t cols, uint8_t scale_bytes,
                                uint64_t *out_blocks_bytes, uint64_t *out_scales_bytes) {
  if (!out_blocks_bytes || !out_scales_bytes) return -1;
  if (rows == 0u || cols == 0u) return -2;
  if ((cols % 32u) != 0u) return -3;
  if (!(scale_bytes == 1u || scale_bytes == 2u)) return -4;

  const uint64_t blocks_per_row_bytes = ((uint64_t)cols / 32u) * 16u;
  const uint64_t scales_per_row_bytes = ((uint64_t)cols / 32u) * (uint64_t)scale_bytes;

  *out_blocks_bytes = (uint64_t)rows * blocks_per_row_bytes;
  *out_scales_bytes = (uint64_t)rows * scales_per_row_bytes;
  return 0;
}

static void ie_add_bias_f32(float *dst, const float *bias, size_t n) {
  if (!dst || !bias) return;
  for (size_t i = 0; i < n; ++i) dst[i] += bias[i];
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

static int ie_resolve_q4_attn_weight(struct ie_gptoss_infer_impl *impl,
                                     uint32_t l,
                                     const char *const blocks_fmts[],
                                     const char *const scales_fmts[],
                                     uint32_t rows,
                                     uint32_t cols,
                                     const char *tag,
                                     const uint8_t **out_blocks,
                                     const uint8_t **out_scales,
                                     uint8_t *out_scale_bytes) {
  if (!impl || !blocks_fmts || !scales_fmts || !out_blocks || !out_scales || !out_scale_bytes) return -1;

  ie_tensor_view_t v_blocks, v_scales;
  const int have_blocks =
      (ie_w_get_layer_fmt_view(impl, l, blocks_fmts, 1, &v_blocks, 0) == 0 && v_blocks.desc);
  const int have_scales =
      (ie_w_get_layer_fmt_view(impl, l, scales_fmts, 1, &v_scales, 0) == 0 && v_scales.desc);

  if (!have_blocks && !have_scales) {
    *out_blocks = NULL;
    *out_scales = NULL;
    *out_scale_bytes = 0u;
    return 0;
  }
  if (!have_blocks || !have_scales) {
    IE_LOGE("create: q4 %s missing blocks/scales layer=%u", tag, (unsigned)l);
    return -2;
  }

  uint8_t scale_bytes = 0u;
  if (v_scales.desc->dtype == TENSOR_DTYPE_BF16) scale_bytes = 2u;
  else if (v_scales.desc->dtype == TENSOR_DTYPE_U8) scale_bytes = 1u;

  if (scale_bytes == 0u) {
    uint64_t tmp_blocks = 0, tmp_scales_1 = 0, tmp_scales_2 = 0;
    if (ie_q4_expected_bytes(rows, cols, 1u, &tmp_blocks, &tmp_scales_1) != 0 ||
        ie_q4_expected_bytes(rows, cols, 2u, &tmp_blocks, &tmp_scales_2) != 0) {
      IE_LOGE("create: q4 %s bad dims layer=%u rows=%u cols=%u",
              tag, (unsigned)l, (unsigned)rows, (unsigned)cols);
      return -3;
    }
    if (v_scales.desc->size_bytes == tmp_scales_1) scale_bytes = 1u;
    else if (v_scales.desc->size_bytes == tmp_scales_2) scale_bytes = 2u;
    else {
      IE_LOGE("create: q4 %s cannot infer scale_bytes layer=%u size_bytes=%llu",
              tag, (unsigned)l, (unsigned long long)v_scales.desc->size_bytes);
      return -4;
    }
  }

  uint64_t want_blocks = 0, want_scales = 0;
  if (ie_q4_expected_bytes(rows, cols, scale_bytes, &want_blocks, &want_scales) != 0) {
    IE_LOGE("create: q4 %s bad dims layer=%u rows=%u cols=%u scale_bytes=%u",
            tag, (unsigned)l, (unsigned)rows, (unsigned)cols, (unsigned)scale_bytes);
    return -5;
  }
  if (ie_require_size_eq(v_blocks.desc, want_blocks) != 0 ||
      ie_require_size_eq(v_scales.desc, want_scales) != 0) {
    IE_LOGE("create: q4 %s size mismatch layer=%u want_blocks=%llu want_scales=%llu",
            tag, (unsigned)l,
            (unsigned long long)want_blocks,
            (unsigned long long)want_scales);
    return -6;
  }

  *out_blocks = (const uint8_t *)v_blocks.ptr;
  *out_scales = (const uint8_t *)v_scales.ptr;
  *out_scale_bytes = scale_bytes;
  return 1;
}

/* ------------------------------------------------------------------------- */
/* Forward pass                                                               */
/* ------------------------------------------------------------------------- */

static int ie_forward_one_token(struct ie_gptoss_infer_impl *impl, ie_kv_cache *kv_layers,
                                uint32_t token_id, uint32_t pos, float *out_logits) {
  static int bf16_trace_done = 0;
  if (!impl || !kv_layers || !out_logits) return -1;

  const ie_gptoss_hparams_t *hp = impl->hp;
  if (!hp) return -2;

  const uint32_t d_model = hp->d_model;
  const uint32_t d_head = hp->d_head;
  const uint32_t n_heads = hp->n_heads;
  const uint32_t n_kv_heads = hp->n_kv_heads;
  const uint32_t vocab = hp->vocab_size;
  const uint32_t n_layers = hp->n_layers;

  if (!impl->w_embed_bf16 || (!impl->w_norm_bf16 && !impl->w_norm_f32) || !impl->w_lm_bf16) return -3;
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

#if defined(IE_CUDA_AVAILABLE)
  if (impl->use_cuda_full) {
    if (!impl->d_x || !impl->d_x1 || !impl->d_x2 || !impl->d_k || !impl->d_v ||
        !impl->d_q || !impl->d_attn_out || !impl->d_mlp_gate || !impl->d_mlp_up ||
        !impl->d_logits || !impl->d_ln1_w || !impl->d_ln2_w || !impl->d_norm_w ||
        !impl->d_kv_K || !impl->d_kv_V) {
      IE_LOGE("cuda_full: missing device buffers");
      return -11;
    }
    if (ie_cuda_memcpy(impl->d_x, impl->x, (size_t)d_model * sizeof(float), IE_CUDA_COPY_H2D) != 0) {
      IE_LOGE("cuda_full: H2D x failed");
      return -11;
    }

    const int dbg_stats = ie_env_flag("IE_CUDA_DEBUG_STATS");
    for (uint32_t l = 0; l < n_layers; ++l) {
      const ie_gptoss_layer_w_t *W = &impl->layers[l];
      ie_kv_cache *kv = &kv_layers[l];

      if (kv->storage != IE_KV_STORAGE_F32) return -7;
      if ((uint32_t)kv->heads != n_kv_heads) return -8;
      if ((uint32_t)kv->head_dim != d_head) return -9;
      if ((uint32_t)kv->max_seq == 0u || pos >= (uint32_t)kv->max_seq) return -10;

      if (ie_rmsnorm_cuda_f32(impl->d_x, impl->d_ln1_w[l], impl->d_x1, 1u, (size_t)d_model,
                              impl->rms_eps) != 0) {
        IE_LOGE("cuda_full: rmsnorm1 failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
        return -11;
      }
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_x1, (size_t)d_model, "x1_ln1", l, pos);
      if (ie_cuda_debug_check_finite(impl, impl->d_x1, (size_t)d_model, "x1_ln1", l, pos) != 0) return -11;

      if (W->q_q4_blocks && W->q_q4_scales) {
        if (ie_gemv_q4_0_f32_device(impl, W->q_q4_blocks, W->q_q4_scales, W->q_q4_scale_bytes,
                                    impl->d_x1, impl->d_q, q_dim, (size_t)d_model, W->q_b) != 0) {
          IE_LOGE("cuda_full: q gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      } else {
        if (ie_gemv_bf16_f32_device(impl, W->q_w, impl->d_x1, impl->d_q,
                                    q_dim, (size_t)d_model, W->q_b) != 0) {
          IE_LOGE("cuda_full: q bf16 gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      }
      if (ie_cuda_debug_check_finite(impl, impl->d_q, q_dim, "q", l, pos) != 0) return -12;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_q, q_dim, "q", l, pos);
      if (W->k_q4_blocks && W->k_q4_scales) {
        if (ie_gemv_q4_0_f32_device(impl, W->k_q4_blocks, W->k_q4_scales, W->k_q4_scale_bytes,
                                    impl->d_x1, impl->d_k, kv_dim, (size_t)d_model, W->k_b) != 0) {
          IE_LOGE("cuda_full: k gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      } else {
        if (ie_gemv_bf16_f32_device(impl, W->k_w, impl->d_x1, impl->d_k,
                                    kv_dim, (size_t)d_model, W->k_b) != 0) {
          IE_LOGE("cuda_full: k bf16 gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      }
      if (ie_cuda_debug_check_finite(impl, impl->d_k, kv_dim, "k", l, pos) != 0) return -12;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_k, kv_dim, "k", l, pos);
      if (W->v_q4_blocks && W->v_q4_scales) {
        if (ie_gemv_q4_0_f32_device(impl, W->v_q4_blocks, W->v_q4_scales, W->v_q4_scale_bytes,
                                    impl->d_x1, impl->d_v, kv_dim, (size_t)d_model, W->v_b) != 0) {
          IE_LOGE("cuda_full: v gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      } else {
        if (ie_gemv_bf16_f32_device(impl, W->v_w, impl->d_x1, impl->d_v,
                                    kv_dim, (size_t)d_model, W->v_b) != 0) {
          IE_LOGE("cuda_full: v bf16 gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -12;
        }
      }
      if (ie_cuda_debug_check_finite(impl, impl->d_v, kv_dim, "v", l, pos) != 0) return -12;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_v, kv_dim, "v", l, pos);

      if (ie_cuda_rope_f32(impl->d_q, NULL, (size_t)n_heads, (size_t)d_head, pos,
                           rope_theta_eff, ie_rope_pos_mul()) != 0 ||
          ie_cuda_rope_f32(NULL, impl->d_k, (size_t)n_kv_heads, (size_t)d_head, pos,
                           rope_theta_eff, ie_rope_pos_mul()) != 0) {
        IE_LOGE("cuda_full: rope failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
        return -15;
      }

      {
        const size_t off = (size_t)pos * kv_dim;
        const size_t kv_bytes = kv_dim * sizeof(float);
        if (ie_cuda_memcpy(impl->d_kv_K[l] + off, impl->d_k, kv_bytes, IE_CUDA_COPY_D2D) != 0 ||
            ie_cuda_memcpy(impl->d_kv_V[l] + off, impl->d_v, kv_bytes, IE_CUDA_COPY_D2D) != 0) {
          IE_LOGE("cuda_full: kv store failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -17;
        }
      }

      {
        int rc_attn = 0;
        if (n_kv_heads == n_heads) {
          rc_attn = ie_attn_cuda_causal_f32(impl->d_q, impl->d_kv_K[l], impl->d_kv_V[l],
                                            (size_t)(pos + 1u), (size_t)n_heads, (size_t)d_head,
                                            inv_sqrt_d, impl->d_attn_out);
        } else {
          rc_attn = ie_attn_cuda_causal_gqa_f32(impl->d_q, impl->d_kv_K[l], impl->d_kv_V[l],
                                                (size_t)(pos + 1u), (size_t)n_heads,
                                                (size_t)n_kv_heads, (size_t)d_head,
                                                inv_sqrt_d, impl->d_attn_out);
        }
        if (rc_attn != 0) {
          IE_LOGE("cuda_full: attn failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -18;
        }
      }
      if (ie_cuda_debug_check_finite(impl, impl->d_attn_out, q_dim, "attn_out", l, pos) != 0) return -18;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_attn_out, q_dim, "attn_out", l, pos);

      if (W->o_q4_blocks && W->o_q4_scales) {
        if (ie_gemv_q4_0_f32_device(impl, W->o_q4_blocks, W->o_q4_scales, W->o_q4_scale_bytes,
                                    impl->d_attn_out, impl->d_x2, (size_t)d_model, q_dim, W->o_b) != 0) {
          IE_LOGE("cuda_full: o gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -19;
        }
      } else {
        if (ie_gemv_bf16_f32_device(impl, W->o_w, impl->d_attn_out, impl->d_x2,
                                    (size_t)d_model, q_dim, W->o_b) != 0) {
          IE_LOGE("cuda_full: o bf16 gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -19;
        }
      }
      if (ie_cuda_add_inplace_f32(impl->d_x, impl->d_x2, (size_t)d_model) != 0) return -19;
      if (ie_cuda_debug_check_finite(impl, impl->d_x, (size_t)d_model, "x_post_attn", l, pos) != 0) return -19;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_x, (size_t)d_model, "x_post_attn", l, pos);

      if (ie_rmsnorm_cuda_f32(impl->d_x, impl->d_ln2_w[l], impl->d_x1, 1u, (size_t)d_model,
                              impl->rms_eps) != 0) {
        IE_LOGE("cuda_full: rmsnorm2 failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
        return -20;
      }
      if (ie_cuda_debug_check_finite(impl, impl->d_x1, (size_t)d_model, "x1_ln2", l, pos) != 0) return -20;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_x1, (size_t)d_model, "x1_ln2", l, pos);

      {
        const ie_moe_w_t *M = &W->moe;
        const uint32_t n_exp = M->n_experts;
        if (n_exp == 0u || !impl->router_logits) return -21;

        if (ie_cuda_memcpy(impl->x1, impl->d_x1, (size_t)d_model * sizeof(float), IE_CUDA_COPY_D2H) != 0) {
          IE_LOGE("cuda_full: D2H x1 failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -21;
        }

        if (M->router_w_f32) {
          ie_gemv_f32(M->router_w_f32, impl->x1, impl->router_logits,
                      (size_t)n_exp, (size_t)d_model, M->router_b_f32, 1);
        } else if (ie_gemv_bf16_f32(M->router_w, impl->x1, impl->router_logits,
                                    (size_t)n_exp, (size_t)d_model, M->router_b) != 0) {
          IE_LOGE("cuda_full: router gemv failed layer=%u pos=%u", (unsigned)l, (unsigned)pos);
          return -22;
        }

        const uint32_t topk = ie_u32_min(impl->moe_topk, n_exp);
        if (topk == 0u) return -23;

        uint32_t idx[IE_GPTOSS_MOE_TOPK_DEFAULT] = {0, 0, 0, 0};
        float val[IE_GPTOSS_MOE_TOPK_DEFAULT] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
        const int gu_scale_fmt = (M->gate_up_scale_bytes == 1u) ? 1 : 0;
        const int dn_scale_fmt = (M->down_scale_bytes == 1u) ? 1 : 0;

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

        if (ie_cuda_zero_f32(impl->d_x2, (size_t)d_model) != 0) return -24;

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

          if (ie_gemv_q4_0_f32_device_ex(impl, gu_b, gu_s, M->gate_up_scale_bytes, gu_scale_fmt,
                                         impl->d_x1, impl->d_mlp_up,
                                         2u * d_ff, (size_t)d_model, gu_bias) != 0) {
            IE_LOGE("cuda_full: moe gate_up failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -24;
          }
          if (ie_cuda_fix_nonfinite_f32(impl->d_mlp_up, 2u * d_ff) != 0) {
            IE_LOGE("cuda_full: fix nonfinite gate_up failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -24;
          }

          if (ie_cuda_silu_mul_f32(impl->d_mlp_up, impl->d_mlp_up + d_ff,
                                   impl->d_mlp_gate, d_ff) != 0) {
            IE_LOGE("cuda_full: silu_mul failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -24;
          }
          if (ie_cuda_fix_nonfinite_f32(impl->d_mlp_gate, d_ff) != 0) {
            IE_LOGE("cuda_full: fix nonfinite mlp_gate failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -24;
          }

          if (ie_gemv_q4_0_f32_device_ex(impl, dn_b, dn_s, M->down_scale_bytes, dn_scale_fmt,
                                         impl->d_mlp_gate, impl->d_attn_out,
                                         (size_t)d_model, d_ff, dn_bias) != 0) {
            IE_LOGE("cuda_full: moe down failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -25;
          }
          if (ie_cuda_fix_nonfinite_f32(impl->d_attn_out, (size_t)d_model) != 0) {
            IE_LOGE("cuda_full: fix nonfinite mlp_down failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -25;
          }

          if (ie_cuda_add_scaled_inplace_f32(impl->d_x2, impl->d_attn_out, p, (size_t)d_model) != 0) {
            IE_LOGE("cuda_full: moe add scaled failed layer=%u pos=%u ex=%u",
                    (unsigned)l, (unsigned)pos, (unsigned)ex);
            return -25;
          }
        }
      }

      if (ie_cuda_add_inplace_f32(impl->d_x, impl->d_x2, (size_t)d_model) != 0) return -25;
      if (ie_cuda_debug_check_finite(impl, impl->d_x, (size_t)d_model, "x_post_mlp", l, pos) != 0) return -25;
      if (dbg_stats && l == 0u && pos == 0u)
        ie_cuda_debug_stats(impl, impl->d_x, (size_t)d_model, "x_post_mlp", l, pos);
    }

    if (ie_rmsnorm_cuda_f32(impl->d_x, impl->d_norm_w, impl->d_x1, 1u, (size_t)d_model, impl->rms_eps) != 0)
      return -26;
    if (ie_cuda_debug_check_finite(impl, impl->d_x1, (size_t)d_model, "x1_norm", 0u, pos) != 0) return -26;
    if (dbg_stats && pos == 0u)
      ie_cuda_debug_stats(impl, impl->d_x1, (size_t)d_model, "x1_norm", 0u, pos);

    if (impl->w_lm_q4_blocks && impl->w_lm_q4_scales) {
      if (ie_gemv_q4_0_f32_device(impl, impl->w_lm_q4_blocks, impl->w_lm_q4_scales,
                                  impl->w_lm_q4_scale_bytes, impl->d_x1, impl->d_logits,
                                  (size_t)vocab, (size_t)d_model, NULL) != 0) {
        return -27;
      }
    } else if (impl->use_f32_lm_head && impl->w_lm_f32) {
      if (ie_gemv_f32_device(impl, impl->w_lm_f32, impl->d_x1, impl->d_logits,
                             (size_t)vocab, (size_t)d_model, NULL) != 0) {
        return -27;
      }
    } else {
      if (ie_gemv_bf16_f32_device(impl, impl->w_lm_bf16, impl->d_x1, impl->d_logits,
                                  (size_t)vocab, (size_t)d_model, NULL) != 0) {
        return -27;
      }
    }
    if (dbg_stats && pos == 0u)
      ie_cuda_debug_stats(impl, impl->d_logits, (size_t)vocab, "logits", 0u, pos);
    if (ie_cuda_memcpy(out_logits, impl->d_logits, (size_t)vocab * sizeof(float), IE_CUDA_COPY_D2H) != 0)
      return -27;
    if (ie_cuda_debug_check_finite(impl, impl->d_logits, (size_t)vocab, "logits", 0u, pos) != 0) return -27;
    return 0;
  }
#endif

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

    if (W->ln1_w_f32) {
      if (ie_rmsnorm_cpu_f32(impl->x, W->ln1_w_f32, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
        IE_LOGE("rmsnorm1 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
        return -11;
      }
    } else if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln1_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
      IE_LOGE("rmsnorm1 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
      return -11;
    }
    if (ie_debug_nan_enabled_() &&
        ie_check_finite_f32_(impl->x1, (size_t)d_model, "rmsnorm1", l, pos) != 0) {
      return -90;
    }

    const int trace_on = ie_trace_bf16_enabled_() && !bf16_trace_done;
    if (trace_on && !W->q_q4_blocks && !W->q_w_f32) {
      fprintf(stderr, "[TRACE] bf16 gemv: layer=%u op=q rows=%zu cols=%zu\n",
              (unsigned)l, q_dim, (size_t)d_model);
    }
    if (W->q_q4_blocks && W->q_q4_scales) {
      if (ie_gemv_q4_0_f32_dispatch(impl, W->q_q4_blocks, W->q_q4_scales, W->q_q4_scale_bytes,
                                    impl->x1, impl->q, q_dim, (size_t)d_model, W->q_b) != 0) {
        return -12;
      }
      if (!W->q_b && W->q_b_f32) ie_add_bias_f32(impl->q, W->q_b_f32, q_dim);
    } else if (W->q_w_f32) {
      ie_gemv_f32(W->q_w_f32, impl->x1, impl->q, q_dim, (size_t)d_model, W->q_b_f32, 1);
    } else {
      if (ie_gemv_bf16_f32(W->q_w, impl->x1, impl->q, q_dim, (size_t)d_model, W->q_b) != 0) return -12;
    }
    if (trace_on && !W->k_q4_blocks && !W->k_w_f32) {
      fprintf(stderr, "[TRACE] bf16 gemv: layer=%u op=k rows=%zu cols=%zu\n",
              (unsigned)l, kv_dim, (size_t)d_model);
    }
    if (W->k_q4_blocks && W->k_q4_scales) {
      if (ie_gemv_q4_0_f32_dispatch(impl, W->k_q4_blocks, W->k_q4_scales, W->k_q4_scale_bytes,
                                    impl->x1, impl->k, kv_dim, (size_t)d_model, W->k_b) != 0) {
        return -13;
      }
      if (!W->k_b && W->k_b_f32) ie_add_bias_f32(impl->k, W->k_b_f32, kv_dim);
    } else if (W->k_w_f32) {
      ie_gemv_f32(W->k_w_f32, impl->x1, impl->k, kv_dim, (size_t)d_model, W->k_b_f32, 1);
    } else {
      if (ie_gemv_bf16_f32(W->k_w, impl->x1, impl->k, kv_dim, (size_t)d_model, W->k_b) != 0) return -13;
    }
    if (trace_on && !W->v_q4_blocks && !W->v_w_f32) {
      fprintf(stderr, "[TRACE] bf16 gemv: layer=%u op=v rows=%zu cols=%zu\n",
              (unsigned)l, kv_dim, (size_t)d_model);
    }
    if (W->v_q4_blocks && W->v_q4_scales) {
      if (ie_gemv_q4_0_f32_dispatch(impl, W->v_q4_blocks, W->v_q4_scales, W->v_q4_scale_bytes,
                                    impl->x1, impl->v, kv_dim, (size_t)d_model, W->v_b) != 0) {
        return -14;
      }
      if (!W->v_b && W->v_b_f32) ie_add_bias_f32(impl->v, W->v_b_f32, kv_dim);
    } else if (W->v_w_f32) {
      ie_gemv_f32(W->v_w_f32, impl->x1, impl->v, kv_dim, (size_t)d_model, W->v_b_f32, 1);
    } else {
      if (ie_gemv_bf16_f32(W->v_w, impl->x1, impl->v, kv_dim, (size_t)d_model, W->v_b) != 0) return -14;
    }

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

      int used_cuda_attn = 0;
#if defined(IE_CUDA_AVAILABLE)
      if (impl->use_cuda_attn && impl->d_q && impl->d_attn_out &&
          impl->d_kv_K && impl->d_kv_V && impl->d_kv_K[l] && impl->d_kv_V[l]) {
        const size_t kv_dim = (size_t)n_kv_heads * (size_t)d_head;
        const size_t q_dim = (size_t)n_heads * (size_t)d_head;
        const size_t kv_bytes = kv_dim * sizeof(float);
        const size_t q_bytes = q_dim * sizeof(float);
        const size_t off = (size_t)pos * kv_dim;

        int rc = 0;
        rc |= ie_cuda_memcpy(impl->d_kv_K[l] + off, impl->k, kv_bytes, IE_CUDA_COPY_H2D);
        rc |= ie_cuda_memcpy(impl->d_kv_V[l] + off, impl->v, kv_bytes, IE_CUDA_COPY_H2D);
        rc |= ie_cuda_memcpy(impl->d_q, impl->q, q_bytes, IE_CUDA_COPY_H2D);
        if (rc == 0) {
          int rc_attn = 0;
          if (n_kv_heads == n_heads) {
            rc_attn = ie_attn_cuda_causal_f32(impl->d_q, impl->d_kv_K[l], impl->d_kv_V[l],
                                              (size_t)(pos + 1u), (size_t)n_heads, (size_t)d_head,
                                              inv_sqrt_d, impl->d_attn_out);
          } else {
            rc_attn = ie_attn_cuda_causal_gqa_f32(impl->d_q, impl->d_kv_K[l], impl->d_kv_V[l],
                                                  (size_t)(pos + 1u), (size_t)n_heads,
                                                  (size_t)n_kv_heads, (size_t)d_head,
                                                  inv_sqrt_d, impl->d_attn_out);
          }
          if (rc_attn == 0 &&
              ie_cuda_memcpy(impl->attn_out, impl->d_attn_out, q_bytes, IE_CUDA_COPY_D2H) == 0) {
            used_cuda_attn = 1;
          }
        }
        if (!used_cuda_attn) {
          IE_LOGW("cuda attn failed: layer=%u pos=%u err=%s",
                  (unsigned)l, (unsigned)pos, ie_cuda_last_error_string());
          impl->use_cuda_attn = 0;
        }
      }
#endif
      if (!used_cuda_attn) {
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
    }

    if (trace_on && !W->o_q4_blocks && !W->o_w_f32) {
      fprintf(stderr, "[TRACE] bf16 gemv: layer=%u op=o rows=%zu cols=%zu\n",
              (unsigned)l, (size_t)d_model, q_dim);
    }
    if (W->o_q4_blocks && W->o_q4_scales) {
      if (ie_gemv_q4_0_f32_dispatch(impl, W->o_q4_blocks, W->o_q4_scales, W->o_q4_scale_bytes,
                                    impl->attn_out, impl->x2, (size_t)d_model, q_dim, W->o_b) != 0) {
        return -19;
      }
      if (!W->o_b && W->o_b_f32) ie_add_bias_f32(impl->x2, W->o_b_f32, (size_t)d_model);
    } else if (W->o_w_f32) {
      ie_gemv_f32(W->o_w_f32, impl->attn_out, impl->x2, (size_t)d_model, q_dim, W->o_b_f32, 1);
    } else {
      if (ie_gemv_bf16_f32(W->o_w, impl->attn_out, impl->x2, (size_t)d_model, q_dim, W->o_b) != 0) return -19;
    }
    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];

    if (W->ln2_w_f32) {
      if (ie_rmsnorm_cpu_f32(impl->x, W->ln2_w_f32, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
        IE_LOGE("rmsnorm2 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
        return -20;
      }
    } else if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln2_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
      IE_LOGE("rmsnorm2 failed: layer=%u pos=%u", (unsigned)l, (unsigned)pos);
      return -20;
    }

    {
      const ie_moe_w_t *M = &W->moe;
      const uint32_t n_exp = M->n_experts;
      if (n_exp == 0u || !impl->router_logits) return -21;

      if (trace_on) {
        fprintf(stderr, "[TRACE] bf16 gemv: layer=%u op=router rows=%u cols=%u\n",
                (unsigned)l, (unsigned)n_exp, (unsigned)d_model);
      }
      if (M->router_w_f32) {
        ie_gemv_f32(M->router_w_f32, impl->x1, impl->router_logits,
                    (size_t)n_exp, (size_t)d_model, M->router_b_f32, 1);
      } else if (ie_gemv_bf16_f32(M->router_w, impl->x1, impl->router_logits,
                                  (size_t)n_exp, (size_t)d_model, M->router_b) != 0) {
        IE_LOGE("router gemv failed: layer=%u pos=%u n_experts=%u",
                (unsigned)l, (unsigned)pos, (unsigned)n_exp);
        return -22;
      }
      if (ie_debug_nan_enabled_() &&
          ie_check_finite_f32_(impl->router_logits, (size_t)n_exp, "router_logits", l, pos) != 0) {
        return -90;
      }

      const uint32_t topk = ie_u32_min(impl->moe_topk, n_exp);
      if (topk == 0u) return -23;

      uint32_t idx[IE_GPTOSS_MOE_TOPK_DEFAULT] = {0, 0, 0, 0};
      float val[IE_GPTOSS_MOE_TOPK_DEFAULT] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
      const int gu_scale_fmt = (M->gate_up_scale_bytes == 1u) ? 1 : 0;
      const int dn_scale_fmt = (M->down_scale_bytes == 1u) ? 1 : 0;

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

        if (ie_gemv_q4_0_f32_dispatch_ex(impl, gu_b, gu_s, M->gate_up_scale_bytes, gu_scale_fmt,
                                         impl->x1, impl->mlp_up,
                                         2u * d_ff, (size_t)d_model, gu_bias) != 0) {
          IE_LOGE("moe gate_up gemv failed: layer=%u pos=%u ex=%u",
                  (unsigned)l, (unsigned)pos, (unsigned)ex);
          return -24;
        }
        if (ie_debug_nan_enabled_() &&
            ie_check_finite_f32_(impl->mlp_up, 2u * d_ff, "mlp_up", l, pos) != 0) {
          return -90;
        }

        for (size_t i = 0; i < d_ff; ++i) {
          const float g = ie_silu_f32(impl->mlp_up[i]);
          const float u = impl->mlp_up[d_ff + i];
          impl->mlp_gate[i] = g * u;
        }
        if (ie_debug_nan_enabled_() &&
            ie_check_finite_f32_(impl->mlp_gate, d_ff, "mlp_gate", l, pos) != 0) {
          return -90;
        }

        if (ie_gemv_q4_0_f32_dispatch_ex(impl, dn_b, dn_s, M->down_scale_bytes, dn_scale_fmt,
                                         impl->mlp_gate, impl->attn_out,
                                         (size_t)d_model, d_ff, dn_bias) != 0) {
          IE_LOGE("moe down gemv failed: layer=%u pos=%u ex=%u",
                  (unsigned)l, (unsigned)pos, (unsigned)ex);
          return -25;
        }
        if (ie_debug_nan_enabled_() &&
            ie_check_finite_f32_(impl->attn_out, (size_t)d_model, "mlp_down", l, pos) != 0) {
          return -90;
        }

        for (uint32_t i = 0; i < d_model; ++i) impl->x2[i] += p * impl->attn_out[i];
      }
    }

    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];
  }

  if (impl->w_norm_f32) {
    if (ie_rmsnorm_cpu_f32(impl->x, impl->w_norm_f32, (size_t)d_model, impl->rms_eps, impl->x1) != 0)
      return -26;
  } else if (ie_rmsnorm_cpu_f32_bf16w(impl->x, impl->w_norm_bf16, (size_t)d_model, impl->rms_eps, impl->x1) != 0) {
    return -26;
  }

  {
    const int trace_on = ie_trace_bf16_enabled_() && !bf16_trace_done;
    if (trace_on && !impl->w_lm_q4_blocks && !impl->w_lm_f32) {
      fprintf(stderr, "[TRACE] bf16 gemv: op=lm_head rows=%u cols=%u\n",
              (unsigned)vocab, (unsigned)d_model);
      bf16_trace_done = 1;
    }
  }
  if (impl->w_lm_q4_blocks && impl->w_lm_q4_scales) {
    if (ie_gemv_q4_0_f32_dispatch(impl, impl->w_lm_q4_blocks, impl->w_lm_q4_scales,
                                  impl->w_lm_q4_scale_bytes, impl->x1, out_logits,
                                  (size_t)vocab, (size_t)d_model, NULL) != 0)
      return -27;
  } else if (impl->w_lm_f32) {
    ie_gemv_f32(impl->w_lm_f32, impl->x1, out_logits, (size_t)vocab, (size_t)d_model, NULL, 0);
  } else {
    if (ie_gemv_bf16_f32(impl->w_lm_bf16, impl->x1, out_logits, (size_t)vocab, (size_t)d_model, NULL) != 0)
      return -27;
  }

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

  impl->use_f32_attn = ie_env_flag("IE_F32_ATTN");
  impl->use_f32_router = ie_env_flag("IE_F32_ROUTER");
  impl->use_f32_lm_head = ie_env_flag("IE_F32_LM_HEAD");
  impl->use_f32_rmsnorm = ie_env_flag("IE_F32_RMSNORM");

  IE_LOGI("create: f32_cache attn=%d router=%d lm_head=%d rmsnorm=%d",
          impl->use_f32_attn, impl->use_f32_router,
          impl->use_f32_lm_head, impl->use_f32_rmsnorm);

#if defined(IE_CUDA_AVAILABLE)
  if (impl->dev && ie_device_kind(impl->dev) == IE_DEV_CUDA && ie_env_flag("IE_CUDA_ATTN")) {
    impl->use_cuda_attn = 1;
  }
  if (impl->dev && ie_device_kind(impl->dev) == IE_DEV_CUDA && ie_env_flag("IE_CUDA_FULL")) {
    impl->use_cuda_full = 1;
    impl->use_cuda_attn = 1;
    impl->use_f32_rmsnorm = 1;
  }
#endif
  IE_LOGI("create: cuda_attn=%d cuda_full=%d", impl->use_cuda_attn, impl->use_cuda_full);

  /* Initialize Q4 LUTs up front to avoid pthread_once in hot path. */
  ie_q4_log2_u8_q3_init_generic();
  ie_q4_log2_u8_q3_init_avx2();

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
      "model.ie.compat.json",
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
    if (impl->dev && ie_env_flag("IE_CUDA_MIRROR_WEIGHTS")) {
      const int rc_blob = ie_device_register_blob(impl->dev, impl->bin_base, impl->bin_size);
      if (rc_blob != 0) {
        IE_LOGW("create: cuda mirror failed rc=%d (set IE_CUDA_MIRROR_WEIGHTS=0 to skip)",
                rc_blob);
      }
    }
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
      "model.norm.scale",
      "norm.weight",
      "norm.scale",
      "transformer.norm_f.weight",
      "transformer.norm_f.scale",
      "transformer.ln_f.weight",
      "transformer.ln_f.scale",
      "final_norm.weight",
      "final_norm.scale",
      "ln_f.weight",
      "ln_f.scale"
    };
    if (ie_w_get_first_view(impl, norm_candidates, 12, &v) != 0 || !v.desc) {
      IE_LOGE("create: missing final norm tensor (tried common candidates)");
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -9;
    }

    const uint64_t want_bf16 = (uint64_t)d_model * 2u;
    const uint64_t want_f32 = (uint64_t)d_model * 4u;

    if (v.desc->size_bytes == want_bf16) {
      impl->w_norm_bf16 = (const uint16_t *)v.ptr;
      if (impl->use_f32_rmsnorm) {
        if (ie_alloc_f32_from_bf16(impl->w_norm_bf16, (size_t)d_model, &impl->w_norm_f32) != 0) {
          IE_LOGE("create: failed to allocate final norm f32");
          ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
          return -10;
        }
      }
    } else if (v.desc->size_bytes == want_f32) {
      impl->w_norm_bf16 = NULL;
      impl->w_norm_f32 = (float *)malloc((size_t)want_f32);
      if (!impl->w_norm_f32) {
        IE_LOGE("create: failed to allocate final norm f32 (bytes=%llu)",
                (unsigned long long)want_f32);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -10;
      }
      memcpy(impl->w_norm_f32, v.ptr, (size_t)want_f32);
      IE_LOGW("create: final norm stored as f32 in weights; using f32 path");
    } else {
      char td_buf[256];
      if (tensor_desc_to_string(v.desc, td_buf, sizeof(td_buf)) >= 0) {
        IE_LOGE("create: final norm size mismatch: %s", td_buf);
      } else {
        IE_LOGE("create: final norm size mismatch: got_bytes=%llu want_bf16=%llu want_f32=%llu",
                (unsigned long long)v.desc->size_bytes,
                (unsigned long long)want_bf16,
                (unsigned long long)want_f32);
      }
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -10;
    }
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
    {
      ie_tensor_view_t v_q4_blocks;
      ie_tensor_view_t v_q4_scales;
      const char *const q4_blocks_candidates[] = {
        "lm_head.weight_blocks",
        "model.lm_head.weight_blocks",
        "transformer.lm_head.weight_blocks",
        "model.embed_tokens.weight_blocks"
      };
      const char *const q4_scales_candidates[] = {
        "lm_head.weight_scales",
        "model.lm_head.weight_scales",
        "transformer.lm_head.weight_scales",
        "model.embed_tokens.weight_scales"
      };
      const int has_blocks = (ie_w_get_first_view(impl, q4_blocks_candidates, 4, &v_q4_blocks) == 0 &&
                              v_q4_blocks.desc);
      const int has_scales = (ie_w_get_first_view(impl, q4_scales_candidates, 4, &v_q4_scales) == 0 &&
                              v_q4_scales.desc);
      if (has_blocks && has_scales) {
        uint8_t scale_bytes = 0;
        if (v_q4_scales.desc->dtype == TENSOR_DTYPE_BF16) scale_bytes = 2u;
        else if (v_q4_scales.desc->dtype == TENSOR_DTYPE_U8) scale_bytes = 1u;
        else {
          uint64_t want_blocks = 0, want_scales_1 = 0, want_scales_2 = 0;
          if (ie_q4_expected_bytes(vocab, d_model, 1u, &want_blocks, &want_scales_1) != 0 ||
              ie_q4_expected_bytes(vocab, d_model, 2u, &want_blocks, &want_scales_2) != 0) {
            IE_LOGE("create: lm_head q4 expected size failed");
            ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
            return -12;
          }
          if (v_q4_scales.desc->size_bytes == want_scales_1) scale_bytes = 1u;
          else if (v_q4_scales.desc->size_bytes == want_scales_2) scale_bytes = 2u;
        }
        if (scale_bytes != 0u) {
          uint64_t want_blocks = 0, want_scales = 0;
          if (ie_q4_expected_bytes(vocab, d_model, scale_bytes, &want_blocks, &want_scales) != 0 ||
              ie_require_size_eq(v_q4_blocks.desc, want_blocks) != 0 ||
              ie_require_size_eq(v_q4_scales.desc, want_scales) != 0) {
            IE_LOGE("create: lm_head q4 size mismatch");
            ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
            return -12;
          }
          impl->w_lm_q4_blocks = (const uint8_t *)v_q4_blocks.ptr;
          impl->w_lm_q4_scales = (const uint8_t *)v_q4_scales.ptr;
          impl->w_lm_q4_scale_bytes = scale_bytes;
          IE_LOGI("create: lm_head q4 enabled (scale_bytes=%u)", (unsigned)scale_bytes);
        }
      }
    }
    if (impl->use_f32_lm_head && !impl->w_lm_q4_blocks) {
      const uint64_t n64 = (uint64_t)vocab * (uint64_t)d_model;
      if (n64 > (uint64_t)SIZE_MAX) {
        IE_LOGE("create: lm_head f32 size overflow (vocab=%u d_model=%u)",
                (unsigned)vocab, (unsigned)d_model);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -12;
      }
      if (ie_alloc_f32_from_bf16(impl->w_lm_bf16, (size_t)n64, &impl->w_lm_f32) != 0) {
        IE_LOGE("create: failed to allocate lm_head f32");
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -12;
      }
    }
  }

  impl->layers = (ie_gptoss_layer_w_t *)calloc((size_t)n_layers, sizeof(*impl->layers));
  if (!impl->layers) {
    IE_LOGE("create: failed to allocate layer table (n_layers=%u)", (unsigned)n_layers);
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -13;
  }

  const uint64_t q_dim = (uint64_t)n_heads * (uint64_t)d_head;
  const uint64_t kv_dim = (uint64_t)n_kv_heads * (uint64_t)d_head;

#if SIZE_MAX < UINT64_MAX
  if (q_dim > (uint64_t)SIZE_MAX || kv_dim > (uint64_t)SIZE_MAX) {
    IE_LOGE("create: attn dim overflow for size_t (q_dim=%llu kv_dim=%llu)",
            (unsigned long long)q_dim, (unsigned long long)kv_dim);
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -13;
  }
#endif

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
    if (impl->use_f32_rmsnorm) {
      if (ie_alloc_f32_from_bf16(LW->ln1_w, (size_t)d_model, &LW->ln1_w_f32) != 0) {
        IE_LOGE("create: failed to allocate ln1 f32 layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -15;
      }
    }

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

    const char *const q_q4_blocks_fmts[] = {"model.layers.%u.self_attn.q_proj.weight_blocks"};
    const char *const q_q4_scales_fmts[] = {"model.layers.%u.self_attn.q_proj.weight_scales"};
    {
      int rc_q4 = ie_resolve_q4_attn_weight(
            impl, l, q_q4_blocks_fmts, q_q4_scales_fmts,
            (uint32_t)q_dim, d_model, "q",
            &LW->q_q4_blocks, &LW->q_q4_scales, &LW->q_q4_scale_bytes);
      if (rc_q4 < 0) {
        IE_LOGE("create: invalid q_proj q4 tensors layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
    }
    const char *const k_q4_blocks_fmts[] = {"model.layers.%u.self_attn.k_proj.weight_blocks"};
    const char *const k_q4_scales_fmts[] = {"model.layers.%u.self_attn.k_proj.weight_scales"};
    {
      int rc_q4 = ie_resolve_q4_attn_weight(
            impl, l, k_q4_blocks_fmts, k_q4_scales_fmts,
            (uint32_t)kv_dim, d_model, "k",
            &LW->k_q4_blocks, &LW->k_q4_scales, &LW->k_q4_scale_bytes);
      if (rc_q4 < 0) {
        IE_LOGE("create: invalid k_proj q4 tensors layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
    }
    const char *const v_q4_blocks_fmts[] = {"model.layers.%u.self_attn.v_proj.weight_blocks"};
    const char *const v_q4_scales_fmts[] = {"model.layers.%u.self_attn.v_proj.weight_scales"};
    {
      int rc_q4 = ie_resolve_q4_attn_weight(
            impl, l, v_q4_blocks_fmts, v_q4_scales_fmts,
            (uint32_t)kv_dim, d_model, "v",
            &LW->v_q4_blocks, &LW->v_q4_scales, &LW->v_q4_scale_bytes);
      if (rc_q4 < 0) {
        IE_LOGE("create: invalid v_proj q4 tensors layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
    }
    const char *const o_q4_blocks_fmts[] = {"model.layers.%u.self_attn.o_proj.weight_blocks"};
    const char *const o_q4_scales_fmts[] = {"model.layers.%u.self_attn.o_proj.weight_scales"};
    {
      int rc_q4 = ie_resolve_q4_attn_weight(
            impl, l, o_q4_blocks_fmts, o_q4_scales_fmts,
            d_model, (uint32_t)q_dim, "o",
            &LW->o_q4_blocks, &LW->o_q4_scales, &LW->o_q4_scale_bytes);
      if (rc_q4 < 0) {
        IE_LOGE("create: invalid o_proj q4 tensors layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
    }

    if (impl->use_f32_attn) {
      const size_t q_dim_sz = (size_t)q_dim;
      const size_t kv_dim_sz = (size_t)kv_dim;
      const size_t d_model_sz = (size_t)d_model;
      const uint64_t q_w_n64 = q_dim * (uint64_t)d_model;
      const uint64_t kv_w_n64 = kv_dim * (uint64_t)d_model;
      const uint64_t o_w_n64 = (uint64_t)d_model * q_dim;
#if SIZE_MAX < UINT64_MAX
      if (q_w_n64 > (uint64_t)SIZE_MAX || kv_w_n64 > (uint64_t)SIZE_MAX || o_w_n64 > (uint64_t)SIZE_MAX) {
        IE_LOGE("create: f32 attn weight size overflow layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
#endif
      if (ie_alloc_f32_from_bf16(LW->q_w, (size_t)q_w_n64, &LW->q_w_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->q_b, q_dim_sz, &LW->q_b_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->k_w, (size_t)kv_w_n64, &LW->k_w_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->k_b, kv_dim_sz, &LW->k_b_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->v_w, (size_t)kv_w_n64, &LW->v_w_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->v_b, kv_dim_sz, &LW->v_b_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->o_w, (size_t)o_w_n64, &LW->o_w_f32) != 0 ||
          ie_alloc_f32_from_bf16(LW->o_b, d_model_sz, &LW->o_b_f32) != 0) {
        IE_LOGE("create: failed to allocate f32 attn weights layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
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
    if (impl->use_f32_rmsnorm) {
      if (ie_alloc_f32_from_bf16(LW->ln2_w, (size_t)d_model, &LW->ln2_w_f32) != 0) {
        IE_LOGE("create: failed to allocate ln2 f32 layer=%u", (unsigned)l);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -29;
      }
    }

    {
      const int rc_moe = ie_resolve_moe_layer(impl, l, LW);
      if (rc_moe != 0) {
        IE_LOGE("create: moe resolve failed: layer=%u rc=%d d_model=%u d_ff=%u",
                (unsigned)l, rc_moe, (unsigned)d_model, (unsigned)d_ff);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -30;
      }
    }

    if (impl->use_f32_router) {
      const uint64_t n_exp = (uint64_t)LW->moe.n_experts;
      const uint64_t router_n64 = n_exp * (uint64_t)d_model;
      if (n_exp > 0 && router_n64 <= (uint64_t)SIZE_MAX) {
        if (ie_alloc_f32_from_bf16(LW->moe.router_w, (size_t)router_n64, &LW->moe.router_w_f32) != 0 ||
            ie_alloc_f32_from_bf16(LW->moe.router_b, (size_t)n_exp, &LW->moe.router_b_f32) != 0) {
          IE_LOGE("create: failed to allocate f32 router weights layer=%u", (unsigned)l);
          ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
          return -30;
        }
      } else if (n_exp > 0) {
        IE_LOGE("create: f32 router size overflow layer=%u n_exp=%u d_model=%u",
                (unsigned)l, (unsigned)LW->moe.n_experts, (unsigned)d_model);
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -30;
      }
    }
  }

#if defined(IE_CUDA_AVAILABLE)
  if (impl->use_cuda_full) {
    impl->d_ln1_w = (float **)calloc((size_t)n_layers, sizeof(float *));
    impl->d_ln2_w = (float **)calloc((size_t)n_layers, sizeof(float *));
    if (!impl->d_ln1_w || !impl->d_ln2_w) {
      IE_LOGW("create: cuda_full ln weight table alloc failed");
      impl->use_cuda_full = 0;
    } else if (!impl->w_norm_f32) {
      IE_LOGW("create: cuda_full requires f32 rmsnorm weights");
      impl->use_cuda_full = 0;
    } else {
      const size_t w_bytes = (size_t)d_model * sizeof(float);
      for (uint32_t l = 0; l < n_layers; ++l) {
        if (!impl->layers[l].ln1_w_f32 || !impl->layers[l].ln2_w_f32) {
          IE_LOGW("create: cuda_full missing ln f32 layer=%u", (unsigned)l);
          impl->use_cuda_full = 0;
          break;
        }
        impl->d_ln1_w[l] = (float *)ie_cuda_malloc(w_bytes);
        impl->d_ln2_w[l] = (float *)ie_cuda_malloc(w_bytes);
        if (!impl->d_ln1_w[l] || !impl->d_ln2_w[l]) {
          IE_LOGW("create: cuda_full ln weight alloc failed layer=%u", (unsigned)l);
          impl->use_cuda_full = 0;
          break;
        }
        if (ie_cuda_memcpy(impl->d_ln1_w[l], impl->layers[l].ln1_w_f32, w_bytes, IE_CUDA_COPY_H2D) != 0 ||
            ie_cuda_memcpy(impl->d_ln2_w[l], impl->layers[l].ln2_w_f32, w_bytes, IE_CUDA_COPY_H2D) != 0) {
          IE_LOGW("create: cuda_full ln weight copy failed layer=%u", (unsigned)l);
          impl->use_cuda_full = 0;
          break;
        }
      }
      if (impl->use_cuda_full) {
        impl->d_norm_w = (float *)ie_cuda_malloc((size_t)d_model * sizeof(float));
        if (!impl->d_norm_w ||
            ie_cuda_memcpy(impl->d_norm_w, impl->w_norm_f32, (size_t)d_model * sizeof(float),
                           IE_CUDA_COPY_H2D) != 0) {
          IE_LOGW("create: cuda_full norm weight copy failed");
          impl->use_cuda_full = 0;
        }
      }
    }

    if (!impl->use_cuda_full) {
      if (impl->d_ln1_w) {
        for (uint32_t l = 0; l < n_layers; ++l) if (impl->d_ln1_w[l]) ie_cuda_free(impl->d_ln1_w[l]);
      }
      if (impl->d_ln2_w) {
        for (uint32_t l = 0; l < n_layers; ++l) if (impl->d_ln2_w[l]) ie_cuda_free(impl->d_ln2_w[l]);
      }
      if (impl->d_norm_w) ie_cuda_free(impl->d_norm_w);
      free(impl->d_ln1_w);
      free(impl->d_ln2_w);
      impl->d_ln1_w = impl->d_ln2_w = NULL;
      impl->d_norm_w = NULL;
    }
  }
#endif

  const size_t d_model_sz = (size_t)d_model;
  const size_t q_dim_sz = (size_t)n_heads * (size_t)d_head;
  const size_t kv_dim_sz = (size_t)n_kv_heads * (size_t)d_head;
  const size_t d_ff_sz = (size_t)d_ff;

  const size_t align = 64u;
  impl->x = (float *)ie_aligned_or_malloc(align, d_model_sz * sizeof(float));
  impl->x1 = (float *)ie_aligned_or_malloc(align, d_model_sz * sizeof(float));
  impl->x2 = (float *)ie_aligned_or_malloc(align, d_model_sz * sizeof(float));
  impl->q = (float *)ie_aligned_or_malloc(align, q_dim_sz * sizeof(float));
  impl->k = (float *)ie_aligned_or_malloc(align, kv_dim_sz * sizeof(float));
  impl->v = (float *)ie_aligned_or_malloc(align, kv_dim_sz * sizeof(float));
  impl->attn_out = (float *)ie_aligned_or_malloc(align, q_dim_sz * sizeof(float));
  impl->mlp_gate = (float *)ie_aligned_or_malloc(align, d_ff_sz * sizeof(float));
  impl->mlp_up = (float *)ie_aligned_or_malloc(align, (2u * d_ff_sz) * sizeof(float));
  impl->scores = (float *)ie_aligned_or_malloc(align, (size_t)hp->max_seq * sizeof(float));
  impl->router_logits = (impl->max_experts > 0u)
                            ? (float *)ie_aligned_or_malloc(align, (size_t)impl->max_experts * sizeof(float))
                            : NULL;

  if (!impl->x || !impl->x1 || !impl->x2 || !impl->q || !impl->k || !impl->v || !impl->attn_out ||
      !impl->mlp_gate || !impl->mlp_up || !impl->scores || (impl->max_experts > 0u && !impl->router_logits)) {
    IE_LOGE("create: activation allocation failed (max_experts=%u)", (unsigned)impl->max_experts);
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -31;
  }

#if defined(IE_CUDA_AVAILABLE)
  if (impl->use_cuda_attn) {
    const size_t q_dim = (size_t)hp->n_heads * (size_t)hp->d_head;
    const size_t kv_dim = (size_t)hp->n_kv_heads * (size_t)hp->d_head;
    const size_t max_seq = (size_t)hp->max_seq;
    const uint32_t n_layers = (uint32_t)hp->n_layers;

    if (q_dim == 0 || kv_dim == 0 || max_seq == 0 || n_layers == 0 ||
        kv_dim > (SIZE_MAX / max_seq / sizeof(float))) {
      IE_LOGW("create: cuda_attn disabled (invalid dims)");
      impl->use_cuda_attn = 0;
    } else {
      const size_t kv_bytes = kv_dim * max_seq * sizeof(float);
      impl->d_q = (float *)ie_cuda_malloc(q_dim * sizeof(float));
      impl->d_attn_out = (float *)ie_cuda_malloc(q_dim * sizeof(float));
      impl->d_kv_K = (float **)calloc((size_t)n_layers, sizeof(float *));
      impl->d_kv_V = (float **)calloc((size_t)n_layers, sizeof(float *));

      if (!impl->d_q || !impl->d_attn_out || !impl->d_kv_K || !impl->d_kv_V) {
        IE_LOGW("create: cuda_attn alloc failed (q/out arrays)");
        impl->use_cuda_attn = 0;
      } else {
        for (uint32_t l = 0; l < n_layers; ++l) {
          impl->d_kv_K[l] = (float *)ie_cuda_malloc(kv_bytes);
          impl->d_kv_V[l] = (float *)ie_cuda_malloc(kv_bytes);
          if (!impl->d_kv_K[l] || !impl->d_kv_V[l]) {
            IE_LOGW("create: cuda_attn alloc failed (kv layer=%u)", (unsigned)l);
            impl->use_cuda_attn = 0;
            break;
          }
        }
      }
    }

    if (!impl->use_cuda_attn) {
      if (impl->d_q) ie_cuda_free(impl->d_q);
      if (impl->d_attn_out) ie_cuda_free(impl->d_attn_out);
      if (impl->d_kv_K) {
        for (uint32_t l = 0; l < (uint32_t)hp->n_layers; ++l) {
          if (impl->d_kv_K[l]) ie_cuda_free(impl->d_kv_K[l]);
        }
      }
      if (impl->d_kv_V) {
        for (uint32_t l = 0; l < (uint32_t)hp->n_layers; ++l) {
          if (impl->d_kv_V[l]) ie_cuda_free(impl->d_kv_V[l]);
        }
      }
      free(impl->d_kv_K);
      free(impl->d_kv_V);
      impl->d_q = NULL;
      impl->d_attn_out = NULL;
      impl->d_kv_K = NULL;
      impl->d_kv_V = NULL;
    }
  }

  if (impl->use_cuda_full) {
    const size_t d_model_sz = (size_t)hp->d_model;
    const size_t kv_dim = (size_t)hp->n_kv_heads * (size_t)hp->d_head;
    const size_t d_ff = (size_t)hp->d_ff;

    impl->d_x = (float *)ie_cuda_malloc(d_model_sz * sizeof(float));
    impl->d_x1 = (float *)ie_cuda_malloc(d_model_sz * sizeof(float));
    impl->d_x2 = (float *)ie_cuda_malloc(d_model_sz * sizeof(float));
    impl->d_k = (float *)ie_cuda_malloc(kv_dim * sizeof(float));
    impl->d_v = (float *)ie_cuda_malloc(kv_dim * sizeof(float));
    impl->d_mlp_gate = (float *)ie_cuda_malloc(d_ff * sizeof(float));
    impl->d_mlp_up = (float *)ie_cuda_malloc((2u * d_ff) * sizeof(float));
    impl->d_logits = (float *)ie_cuda_malloc((size_t)hp->vocab_size * sizeof(float));

    if (!impl->d_x || !impl->d_x1 || !impl->d_x2 || !impl->d_k || !impl->d_v ||
        !impl->d_mlp_gate || !impl->d_mlp_up || !impl->d_logits) {
      IE_LOGW("create: cuda_full alloc failed (activations)");
      impl->use_cuda_full = 0;
    } else if (!impl->d_kv_K || !impl->d_kv_V) {
      IE_LOGW("create: cuda_full requires cuda_attn kv buffers");
      impl->use_cuda_full = 0;
    } else {
      for (uint32_t l = 0; l < (uint32_t)hp->n_layers; ++l) {
        if (!impl->d_kv_K[l] || !impl->d_kv_V[l]) {
          IE_LOGW("create: cuda_full missing kv layer=%u", (unsigned)l);
          impl->use_cuda_full = 0;
          break;
        }
      }
    }

    if (!impl->use_cuda_full) {
      if (impl->d_x) ie_cuda_free(impl->d_x);
      if (impl->d_x1) ie_cuda_free(impl->d_x1);
      if (impl->d_x2) ie_cuda_free(impl->d_x2);
      if (impl->d_k) ie_cuda_free(impl->d_k);
      if (impl->d_v) ie_cuda_free(impl->d_v);
      if (impl->d_mlp_gate) ie_cuda_free(impl->d_mlp_gate);
      if (impl->d_mlp_up) ie_cuda_free(impl->d_mlp_up);
      if (impl->d_logits) ie_cuda_free(impl->d_logits);
      impl->d_x = impl->d_x1 = impl->d_x2 = NULL;
      impl->d_k = impl->d_v = NULL;
      impl->d_mlp_gate = impl->d_mlp_up = NULL;
      impl->d_logits = NULL;
    }
  }
#endif

  IE_LOGI("create: success");
  *out_ctx = (ie_gptoss_infer_t *)impl;
  return 0;
}

void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx) {
  if (!ctx) return;
  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  if (impl->bin_base) ie_munmap_ro(impl->bin_base, impl->bin_size);
  tensor_map_free(&impl->tmap);

  if (impl->layers) {
    for (uint32_t l = 0; l < impl->hp->n_layers; ++l) {
      ie_gptoss_layer_w_t *LW = &impl->layers[l];
      free(LW->ln1_w_f32);
      free(LW->ln2_w_f32);
      free(LW->q_w_f32);
      free(LW->q_b_f32);
      free(LW->k_w_f32);
      free(LW->k_b_f32);
      free(LW->v_w_f32);
      free(LW->v_b_f32);
      free(LW->o_w_f32);
      free(LW->o_b_f32);
      free(LW->moe.router_w_f32);
      free(LW->moe.router_b_f32);
    }
  }

  free(impl->w_norm_f32);
  free(impl->w_lm_f32);
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

  if (impl->d_ln1_w) {
    for (uint32_t l = 0; l < impl->hp->n_layers; ++l) {
      if (impl->d_ln1_w[l]) ie_cuda_free(impl->d_ln1_w[l]);
    }
  }
  if (impl->d_ln2_w) {
    for (uint32_t l = 0; l < impl->hp->n_layers; ++l) {
      if (impl->d_ln2_w[l]) ie_cuda_free(impl->d_ln2_w[l]);
    }
  }
  if (impl->d_norm_w) ie_cuda_free(impl->d_norm_w);
  free(impl->d_ln1_w);
  free(impl->d_ln2_w);

  if (impl->d_q) ie_cuda_free(impl->d_q);
  if (impl->d_attn_out) ie_cuda_free(impl->d_attn_out);
  if (impl->d_kv_K) {
    for (uint32_t l = 0; l < impl->hp->n_layers; ++l) {
      if (impl->d_kv_K[l]) ie_cuda_free(impl->d_kv_K[l]);
    }
  }
  if (impl->d_kv_V) {
    for (uint32_t l = 0; l < impl->hp->n_layers; ++l) {
      if (impl->d_kv_V[l]) ie_cuda_free(impl->d_kv_V[l]);
    }
  }
  free(impl->d_kv_K);
  free(impl->d_kv_V);
  if (impl->d_x) ie_cuda_free(impl->d_x);
  if (impl->d_x1) ie_cuda_free(impl->d_x1);
  if (impl->d_x2) ie_cuda_free(impl->d_x2);
  if (impl->d_k) ie_cuda_free(impl->d_k);
  if (impl->d_v) ie_cuda_free(impl->d_v);
  if (impl->d_mlp_gate) ie_cuda_free(impl->d_mlp_gate);
  if (impl->d_mlp_up) ie_cuda_free(impl->d_mlp_up);
  if (impl->d_logits) ie_cuda_free(impl->d_logits);

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
