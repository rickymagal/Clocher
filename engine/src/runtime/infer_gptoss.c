/* ============================================================================
 * File: engine/src/runtime/infer_gptoss.c
 * ============================================================================
 */
/**
 * @file infer_gptoss.c
 * @brief GPT-OSS inference entrypoints with a CPU-first forward pass.
 *
 * This implementation is intentionally "tensor_map-minimal":
 * - It does NOT require tensor_map.json to carry dtype/shape metadata.
 * - It only requires (name, offset, nbytes/size_bytes).
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
 *   - scales: BF16 per block
 */

#include <fcntl.h>
#include <math.h>
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
#include "tensor_map.h"

/* ------------------------------------------------------------------------- */
/* Configuration defaults                                                     */
/* ------------------------------------------------------------------------- */

/** @brief Default RMSNorm epsilon (matches common HF defaults). */
#define IE_GPTOSS_RMS_EPS_DEFAULT (1e-5f)

/** @brief Default RoPE theta (matches common HF defaults). */
#define IE_GPTOSS_ROPE_THETA_DEFAULT (10000.0f)

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

  /** @brief gate_up BF16 scales for all experts. */
  const uint16_t *gate_up_scales;

  /** @brief gate_up BF16 bias for all experts, optional. */
  const uint16_t *gate_up_bias;

  /** @brief down Q4 blocks for all experts. */
  const uint8_t *down_blocks;

  /** @brief down BF16 scales for all experts. */
  const uint16_t *down_scales;

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

  /** @brief MLP up buffer (FP32), reserved (currently unused). */
  float *mlp_up;

  /** @brief Softmax scores scratch (FP32), shape [max_seq]. */
  float *scores;

  /** @brief Router logits scratch (FP32), shape [max_experts]. */
  float *router_logits;

  /** @brief RMSNorm epsilon. */
  float rms_eps;

  /** @brief RoPE theta. */
  float rope_theta;

  /** @brief Maximum experts across all layers (for scratch allocation). */
  uint32_t max_experts;
};

/* ------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* ------------------------------------------------------------------------- */

/**
 * @brief Return the smaller of two sizes.
 * @param a First size.
 * @param b Second size.
 * @return min(a, b).
 */
static size_t ie_min_size(size_t a, size_t b) { return (a < b) ? a : b; }

/**
 * @brief Allocate a directory name string for a UNIX path.
 *
 * Behavior:
 * - If path contains '/', returns everything before the last '/'.
 * - If no '/', returns ".".
 * - If path is "/" or "/name", returns "/".
 *
 * @param path Input path.
 * @return Newly allocated string on success, NULL on allocation failure.
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
 * @brief Join two path components with exactly one '/' between them.
 * @param a First component.
 * @param b Second component.
 * @return Newly allocated joined path, or NULL on allocation failure.
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
 * @param path Path to file.
 * @param out_base Output base pointer for mapping.
 * @param out_size Output size in bytes for mapping.
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
 * @brief Unmap a read-only memory mapping.
 * @param base Base pointer.
 * @param size Mapping size in bytes.
 */
static void ie_munmap_ro(uint8_t *base, size_t size) {
  if (!base || size == 0) return;
  (void)munmap((void *)base, size);
}

/* ------------------------------------------------------------------------- */
/* BF16 helpers                                                               */
/* ------------------------------------------------------------------------- */

/**
 * @brief Convert BF16 bit pattern to FP32.
 * @param v BF16 bits.
 * @return Float value.
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
 * @brief SiLU activation (x * sigmoid(x)).
 * @param x Input.
 * @return SiLU(x).
 */
static inline float ie_silu_f32(float x) { return x / (1.0f + expf(-x)); }

/* ------------------------------------------------------------------------- */
/* Math primitives                                                            */
/* ------------------------------------------------------------------------- */

/**
 * @brief Dot product of two FP32 vectors.
 * @param a Vector a (length n).
 * @param b Vector b (length n).
 * @param n Length.
 * @return Sum_i a[i]*b[i].
 */
static float ie_dot_f32(const float *a, const float *b, size_t n) {
  float s = 0.0f;
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

/**
 * @brief In-place softmax for a FP32 vector.
 * @param x Vector (length n).
 * @param n Length.
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
 * @brief A resolved view into the weights mmap with its descriptor.
 */
typedef struct ie_tensor_view {
  /** @brief Pointer into mmap region for tensor bytes. */
  const void *ptr;
  /** @brief Tensor descriptor from tensor_map. */
  const tensor_desc_t *desc;
} ie_tensor_view_t;

/**
 * @brief Find a tensor descriptor by name in a tensor map.
 * @param tmap Tensor map.
 * @param name Tensor name.
 * @return Descriptor pointer if found, else NULL.
 */
static const tensor_desc_t *ie_w_find(const tensor_map_t *tmap, const char *name) {
  if (!tmap || !name) return NULL;
  return tensor_map_find(tmap, name);
}

/**
 * @brief Resolve a named tensor into a (ptr, desc) view.
 *
 * Validates that (offset + size_bytes) fits in the mapped bin.
 *
 * @param impl Inference implementation.
 * @param name Tensor name.
 * @param out Output view.
 * @return 0 on success, negative error code on failure.
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
 * @brief Resolve the first existing tensor among a list of names.
 * @param impl Inference implementation.
 * @param names Candidate names array.
 * @param count Number of candidates.
 * @param out Output view.
 * @return 0 if any candidate resolved, else negative error code.
 */
static int ie_w_get_first_view(const struct ie_gptoss_infer_impl *impl, const char *const *names,
                               size_t count, ie_tensor_view_t *out) {
  if (!impl || !names || !out) return -1;
  for (size_t i = 0; i < count; ++i) {
    if (ie_w_get_view(impl, names[i], out) == 0) return 0;
  }
  out->ptr = NULL;
  out->desc = NULL;
  return -2;
}

/**
 * @brief Resolve a layer tensor using printf-style name formats.
 * @param impl Inference implementation.
 * @param layer Layer index.
 * @param fmts Array of printf formats with a single %u for layer.
 * @param count Number of formats.
 * @param out Output view.
 * @param required If non-zero, fail when not found; if zero, return 0 with out cleared.
 * @return 0 on success (or optional miss), negative error code on failure.
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
    if (ie_w_get_view(impl, name, out) == 0) return 0;
  }

  return required ? -2 : 0;
}

/**
 * @brief Require exact tensor size in bytes.
 * @param td Tensor descriptor.
 * @param want_bytes Expected bytes.
 * @return 0 if matches, else negative error.
 */
static int ie_require_size_eq(const tensor_desc_t *td, uint64_t want_bytes) {
  if (!td) return -1;
  return (td->size_bytes == want_bytes) ? 0 : -2;
}

/**
 * @brief Require tensor size divisible by a value.
 * @param td Tensor descriptor.
 * @param div Divisor (non-zero).
 * @return 0 if divisible, else negative error.
 */
static int ie_require_size_div(const tensor_desc_t *td, uint64_t div) {
  if (!td || div == 0u) return -1;
  return ((td->size_bytes % div) == 0u) ? 0 : -2;
}

/* ------------------------------------------------------------------------- */
/* RMSNorm + RoPE (float activations, BF16 weights supported)                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief CPU RMSNorm for FP32 activations + FP32 weights.
 *
 * Marked weak so an optimized implementation can override.
 *
 * @param x Input vector (length n).
 * @param w Weight vector (length n).
 * @param n Length.
 * @param eps Epsilon.
 * @param y Output vector (length n). May alias x.
 * @return 0 on success, negative error on invalid inputs.
 */
IE_WEAK int ie_rmsnorm_cpu_f32(const float *x, const float *w, size_t n, float eps, float *y) {
  if (!x || !w || !y || n == 0u) return -1;

  float ss = 0.0f;
  for (size_t i = 0; i < n; ++i) ss += x[i] * x[i];

  const float inv = 1.0f / sqrtf(ss / (float)n + eps);

  if (y == x) {
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
    return 0;
  }

  for (size_t i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
  return 0;
}

/**
 * @brief CPU RMSNorm for FP32 activations + BF16 weights.
 * @param x Input vector (length n).
 * @param w_bf16 Weight vector (BF16, length n).
 * @param n Length.
 * @param eps Epsilon.
 * @param y Output vector (length n).
 * @return 0 on success, negative error on invalid inputs.
 */
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

/**
 * @brief Apply RoPE rotation to Q and/or K in-place.
 *
 * Marked weak so an optimized implementation can override.
 *
 * @param q Q buffer (may be NULL). Shape [heads, head_dim].
 * @param k K buffer (may be NULL). Shape [heads, head_dim].
 * @param heads Number of heads in the given buffer.
 * @param head_dim Head dimension (must be even).
 * @param pos Token position.
 * @param theta RoPE theta parameter.
 * @return 0 on success, negative error on invalid inputs.
 */
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

/**
 * @brief GEMV for BF16 matrix times FP32 vector, output FP32.
 * @param W_bf16 Matrix (row-major) BF16, shape [rows, cols].
 * @param x Input vector FP32, length cols.
 * @param y Output vector FP32, length rows.
 * @param rows Rows.
 * @param cols Cols.
 * @param bias_bf16 Optional BF16 bias (length rows) or NULL.
 * @return 0 on success, negative error on invalid inputs.
 */
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
/* Q4_0-like blocks: row dot                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief Compute dot product between a quantized Q4_0-like row and FP32 vector.
 *
 * Layout assumptions (per 32 columns block):
 * - 16 bytes of packed nibbles (2 weights per byte), values in [-8..7]
 * - 1 BF16 scale per block
 *
 * @param row_blocks Pointer to packed blocks for a single row.
 * @param row_scales_bf16 Pointer to BF16 scales for a single row (one per block).
 * @param x FP32 vector of length cols.
 * @param cols Number of columns (must be divisible by 32).
 * @return Dot product in FP32.
 */
static float ie_q4_0_row_dot_f32(const uint8_t *row_blocks, const uint16_t *row_scales_bf16,
                                const float *x, size_t cols) {
  const size_t qk = 32u;
  const size_t qs_bytes = qk / 2u;
  const size_t nb = cols / qk;

  float acc = 0.0f;
  for (size_t b = 0; b < nb; ++b) {
    const float d = ie_bf16_to_f32(row_scales_bf16[b]);
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

/**
 * @brief GEMV for Q4_0-like matrix times FP32 vector, output FP32.
 * @param W_blocks Packed blocks for matrix, row-major by blocks.
 * @param W_scales_bf16 BF16 scales for matrix, row-major by blocks.
 * @param x Input vector FP32, length cols.
 * @param y Output vector FP32, length rows.
 * @param rows Rows.
 * @param cols Cols (must be divisible by 32).
 * @param bias_bf16 Optional BF16 bias (length rows) or NULL.
 * @return 0 on success, negative error on invalid inputs.
 */
static int ie_gemv_q4_0_f32(const uint8_t *W_blocks, const uint16_t *W_scales_bf16, const float *x,
                           float *y, size_t rows, size_t cols, const uint16_t *bias_bf16) {
  if (!W_blocks || !W_scales_bf16 || !x || !y || rows == 0u || cols == 0u) return -1;
  if ((cols % 32u) != 0u) return -2;

  const size_t qk = 32u;
  const size_t qs_bytes = qk / 2u;
  const size_t nb = cols / qk;

  const size_t blocks_per_row_bytes = nb * qs_bytes;
  const size_t scales_per_row = nb;

  for (size_t r = 0; r < rows; ++r) {
    const uint8_t *rb = W_blocks + r * blocks_per_row_bytes;
    const uint16_t *rs = W_scales_bf16 + r * scales_per_row;
    float acc = ie_q4_0_row_dot_f32(rb, rs, x, cols);
    if (bias_bf16) acc += ie_bf16_to_f32(bias_bf16[r]);
    y[r] = acc;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Attention                                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief CPU causal attention for grouped-query attention (GQA).
 *
 * This is a simple reference implementation:
 * - Q is [n_heads, head_dim]
 * - K/V are [seq_len, n_kv_heads, head_dim]
 * - For each query head, choose its kv head via group mapping.
 *
 * @param Q Query buffer.
 * @param K Key buffer (sequence-major).
 * @param V Value buffer (sequence-major).
 * @param seq_len Number of tokens available (<= max_seq).
 * @param n_heads Number of query heads.
 * @param n_kv_heads Number of kv heads.
 * @param head_dim Head dim.
 * @param inv_sqrt_d 1/sqrt(head_dim).
 * @param out Output buffer [n_heads, head_dim].
 * @param scores Scratch buffer [seq_len].
 */
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

/**
 * @brief Infer number of experts from router weight tensor size.
 *
 * Router is BF16 with shape [n_experts, d_model] => bytes = n_experts*d_model*2.
 *
 * @param td_router_w Router weight descriptor.
 * @param d_model Model width.
 * @param out_n_experts Output inferred expert count.
 * @return 0 on success, negative error on mismatch/invalid.
 */
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

/**
 * @brief Compute expected byte sizes for a Q4_0-like expert matrix group.
 *
 * Each expert contains a matrix [rows, cols] quantized by blocks of 32 columns.
 *
 * @param n_experts Number of experts.
 * @param rows Rows.
 * @param cols Cols (must be multiple of 32).
 * @param out_blocks_bytes Output expected total blocks bytes for all experts.
 * @param out_scales_bytes Output expected total scales bytes for all experts.
 * @return 0 on success, negative error otherwise.
 */
static int ie_moe_expected_bytes(uint32_t n_experts, uint32_t rows, uint32_t cols,
                                uint64_t *out_blocks_bytes, uint64_t *out_scales_bytes) {
  if (!out_blocks_bytes || !out_scales_bytes) return -1;
  if (n_experts == 0u || rows == 0u || cols == 0u) return -2;
  if ((cols % 32u) != 0u) return -3;

  const uint64_t qk = 32u;
  const uint64_t qs_bytes = 16u;
  const uint64_t nb = (uint64_t)cols / qk;

  const uint64_t blocks_per_row_bytes = nb * qs_bytes;
  const uint64_t scales_per_row_bytes = nb * 2u;

  const uint64_t one_blocks = (uint64_t)rows * blocks_per_row_bytes;
  const uint64_t one_scales = (uint64_t)rows * scales_per_row_bytes;

  *out_blocks_bytes = one_blocks * (uint64_t)n_experts;
  *out_scales_bytes = one_scales * (uint64_t)n_experts;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* Resolve MoE tensors for one layer                                          */
/* ------------------------------------------------------------------------- */

/**
 * @brief Resolve and validate all MoE tensors for a single layer.
 *
 * Required:
 * - router.weight (BF16)
 * - experts gate_up blocks+scales (Q4 + BF16 scales)
 * - experts down blocks+scales (Q4 + BF16 scales)
 *
 * Optional:
 * - router.bias (BF16)
 * - experts gate_up bias (BF16)
 * - experts down bias (BF16)
 *
 * @param impl Inference implementation.
 * @param l Layer index.
 * @param LW Output layer weight structure to fill.
 * @return 0 on success, negative error code on failure.
 */
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

  if (ie_w_get_layer_fmt_view(impl, l, router_w_fmts, 1, &v_router_w, 1) != 0) return -2;
  if (ie_w_get_layer_fmt_view(impl, l, router_b_fmts, 1, &v_router_b, 0) != 0) {
    v_router_b.ptr = NULL;
    v_router_b.desc = NULL;
  }

  if (!v_router_w.desc) return -3;

  uint32_t n_experts = 0;
  if (ie_infer_n_experts_from_router(v_router_w.desc, d_model, &n_experts) != 0) return -4;

  /* HF linear weight layout is [out_features, in_features] => [n_experts, d_model]. */
  const int transposed = 0;

  if (v_router_b.desc) {
    if (ie_require_size_eq(v_router_b.desc, (uint64_t)n_experts * 2u) != 0) return -5;
  }

  if (ie_w_get_layer_fmt_view(impl, l, gu_blocks_fmts, 1, &v_gu_blocks, 1) != 0) return -6;
  if (ie_w_get_layer_fmt_view(impl, l, gu_scales_fmts, 1, &v_gu_scales, 1) != 0) return -7;
  if (ie_w_get_layer_fmt_view(impl, l, gu_bias_fmts, 1, &v_gu_bias, 0) != 0) {
    v_gu_bias.ptr = NULL;
    v_gu_bias.desc = NULL;
  }

  if (ie_w_get_layer_fmt_view(impl, l, dn_blocks_fmts, 1, &v_dn_blocks, 1) != 0) return -8;
  if (ie_w_get_layer_fmt_view(impl, l, dn_scales_fmts, 1, &v_dn_scales, 1) != 0) return -9;
  if (ie_w_get_layer_fmt_view(impl, l, dn_bias_fmts, 1, &v_dn_bias, 0) != 0) {
    v_dn_bias.ptr = NULL;
    v_dn_bias.desc = NULL;
  }

  if (!v_gu_blocks.desc || !v_gu_scales.desc || !v_dn_blocks.desc || !v_dn_scales.desc) return -10;

  uint64_t want_gu_blocks = 0, want_gu_scales = 0;
  if (ie_moe_expected_bytes(n_experts, 2u * d_ff, d_model, &want_gu_blocks, &want_gu_scales) != 0)
    return -11;

  uint64_t want_dn_blocks = 0, want_dn_scales = 0;
  if (ie_moe_expected_bytes(n_experts, d_model, d_ff, &want_dn_blocks, &want_dn_scales) != 0)
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
  LW->moe.router_transposed = transposed;

  LW->moe.gate_up_blocks = (const uint8_t *)v_gu_blocks.ptr;
  LW->moe.gate_up_scales = (const uint16_t *)v_gu_scales.ptr;
  LW->moe.gate_up_bias = (const uint16_t *)v_gu_bias.ptr;

  LW->moe.down_blocks = (const uint8_t *)v_dn_blocks.ptr;
  LW->moe.down_scales = (const uint16_t *)v_dn_scales.ptr;
  LW->moe.down_bias = (const uint16_t *)v_dn_bias.ptr;

  if (ie_require_size_div(v_gu_blocks.desc, n_experts) != 0) return -19;
  if (ie_require_size_div(v_gu_scales.desc, n_experts) != 0) return -20;
  if (v_gu_bias.desc && ie_require_size_div(v_gu_bias.desc, n_experts) != 0) return -21;

  if (ie_require_size_div(v_dn_blocks.desc, n_experts) != 0) return -22;
  if (ie_require_size_div(v_dn_scales.desc, n_experts) != 0) return -23;
  if (v_dn_bias.desc && ie_require_size_div(v_dn_bias.desc, n_experts) != 0) return -24;

  LW->moe.gate_up_blocks_stride = (size_t)(v_gu_blocks.desc->size_bytes / n_experts);
  LW->moe.gate_up_scales_stride = (size_t)(v_gu_scales.desc->size_bytes / n_experts);
  LW->moe.gate_up_bias_stride = v_gu_bias.desc ? (size_t)(v_gu_bias.desc->size_bytes / n_experts)
                                               : 0u;

  LW->moe.down_blocks_stride = (size_t)(v_dn_blocks.desc->size_bytes / n_experts);
  LW->moe.down_scales_stride = (size_t)(v_dn_scales.desc->size_bytes / n_experts);
  LW->moe.down_bias_stride = v_dn_bias.desc ? (size_t)(v_dn_bias.desc->size_bytes / n_experts) : 0u;

  if (impl->max_experts < n_experts) impl->max_experts = n_experts;

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Forward pass                                                               */
/* ------------------------------------------------------------------------- */

/**
 * @brief Execute a single-token forward pass.
 *
 * This performs:
 * - embed(token)
 * - for each layer:
 *   - RMSNorm
 *   - QKV projections
 *   - RoPE
 *   - KV cache store
 *   - causal attention (MHA or GQA)
 *   - output projection + residual
 *   - RMSNorm
 *   - MoE router top-2 selection
 *   - expert gate_up (Q4) -> SiLU(gate)*up
 *   - expert down (Q4) -> residual
 * - final RMSNorm + lm_head projection to logits
 *
 * @param impl Inference implementation.
 * @param kv_layers Array of kv caches (one per layer).
 * @param token_id Input token id.
 * @param pos Position index.
 * @param out_logits Output logits (FP32), length vocab.
 * @return 0 on success, negative error code on failure.
 */
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

  for (uint32_t l = 0; l < n_layers; ++l) {
    const ie_gptoss_layer_w_t *W = &impl->layers[l];
    ie_kv_cache *kv = &kv_layers[l];

    if (kv->storage != IE_KV_STORAGE_F32) return -7;
    if ((uint32_t)kv->heads != n_kv_heads) return -8;
    if ((uint32_t)kv->head_dim != d_head) return -9;
    if ((uint32_t)kv->max_seq < hp->max_seq) return -10;

    if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln1_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0)
      return -11;

    if (ie_gemv_bf16_f32(W->q_w, impl->x1, impl->q, q_dim, (size_t)d_model, W->q_b) != 0) return -12;
    if (ie_gemv_bf16_f32(W->k_w, impl->x1, impl->k, kv_dim, (size_t)d_model, W->k_b) != 0) return -13;
    if (ie_gemv_bf16_f32(W->v_w, impl->x1, impl->v, kv_dim, (size_t)d_model, W->v_b) != 0) return -14;

    if (ie_rope_apply_f32(impl->q, NULL, (size_t)n_heads, (size_t)d_head, pos, impl->rope_theta) != 0) return -15;
    if (ie_rope_apply_f32(NULL, impl->k, (size_t)n_kv_heads, (size_t)d_head, pos, impl->rope_theta) != 0) return -16;

    if (ie_kv_store_token_f32(kv, (size_t)pos, impl->k, impl->v) != 0) return -17;

    {
      const float *Kbase = (const float *)kv->K;
      const float *Vbase = (const float *)kv->V;

      if (n_kv_heads == n_heads) {
        if (ie_attn_cpu_causal_f32(impl->q, Kbase, Vbase, (size_t)(pos + 1u), (size_t)n_heads,
                                   (size_t)d_head, inv_sqrt_d, impl->attn_out, impl->scores) != 0) {
          return -18;
        }
      } else {
        ie_attn_causal_gqa_f32(impl->q, Kbase, Vbase, pos + 1u, n_heads, n_kv_heads, d_head, inv_sqrt_d,
                               impl->attn_out, impl->scores);
      }
    }

    if (ie_gemv_bf16_f32(W->o_w, impl->attn_out, impl->x2, (size_t)d_model, q_dim, W->o_b) != 0) return -19;

    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];

    if (ie_rmsnorm_cpu_f32_bf16w(impl->x, W->ln2_w, (size_t)d_model, impl->rms_eps, impl->x1) != 0) return -20;

    {
      const ie_moe_w_t *M = &W->moe;
      const uint32_t n_exp = M->n_experts;
      if (n_exp == 0u || !impl->router_logits) return -21;

      if (ie_gemv_bf16_f32(M->router_w, impl->x1, impl->router_logits, (size_t)n_exp, (size_t)d_model,
                           M->router_b) != 0)
        return -22;

      uint32_t e0 = 0, e1 = 0;
      float s0 = impl->router_logits[0];
      float s1 = -INFINITY;

      for (uint32_t e = 1; e < n_exp; ++e) {
        const float v = impl->router_logits[e];
        if (v > s0) {
          s1 = s0;
          e1 = e0;
          s0 = v;
          e0 = e;
        } else if (v > s1) {
          s1 = v;
          e1 = e;
        }
      }

      float wts[2];
      wts[0] = s0;
      wts[1] = s1;
      ie_softmax_inplace_f32(wts, 2);

      for (uint32_t i = 0; i < d_model; ++i) impl->x2[i] = 0.0f;

      for (uint32_t sel = 0; sel < 2; ++sel) {
        const uint32_t ex = (sel == 0) ? e0 : e1;
        const float p = wts[sel];
        if (!(p > 0.0f)) continue;

        const uint8_t *gu_b = M->gate_up_blocks + ex * M->gate_up_blocks_stride;
        const uint16_t *gu_s =
            (const uint16_t *)((const uint8_t *)M->gate_up_scales + ex * M->gate_up_scales_stride);
        const uint16_t *gu_bias =
            M->gate_up_bias ? (const uint16_t *)((const uint8_t *)M->gate_up_bias + ex * M->gate_up_bias_stride)
                            : NULL;

        const uint8_t *dn_b = M->down_blocks + ex * M->down_blocks_stride;
        const uint16_t *dn_s =
            (const uint16_t *)((const uint8_t *)M->down_scales + ex * M->down_scales_stride);
        const uint16_t *dn_bias =
            M->down_bias ? (const uint16_t *)((const uint8_t *)M->down_bias + ex * M->down_bias_stride) : NULL;

        if (ie_gemv_q4_0_f32(gu_b, gu_s, impl->x1, impl->router_logits, 2u * d_ff, (size_t)d_model, gu_bias) != 0)
          return -24;

        for (size_t i = 0; i < d_ff; ++i) {
          const float g = ie_silu_f32(impl->router_logits[i]);
          const float u = impl->router_logits[d_ff + i];
          impl->mlp_gate[i] = g * u;
        }

        if (ie_gemv_q4_0_f32(dn_b, dn_s, impl->mlp_gate, impl->attn_out, (size_t)d_model, d_ff, dn_bias) != 0)
          return -25;

        for (uint32_t i = 0; i < d_model; ++i) impl->x2[i] += p * impl->attn_out[i];
      }
    }

    for (uint32_t i = 0; i < d_model; ++i) impl->x[i] += impl->x2[i];
  }

  if (ie_rmsnorm_cpu_f32_bf16w(impl->x, impl->w_norm_bf16, (size_t)d_model, impl->rms_eps, impl->x1) != 0)
    return -26;

  if (ie_gemv_bf16_f32(impl->w_lm_bf16, impl->x1, out_logits, (size_t)vocab, (size_t)d_model, NULL) != 0)
    return -27;

  return 0;
}

/* ------------------------------------------------------------------------- */
/* Public API                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * @brief Create a GPT-OSS inference context.
 *
 * Loads tensor_map.json from the model directory and mmaps the model weights file.
 * Resolves and validates all required tensors against the provided hparams.
 *
 * @param dev_const Device handle (may be NULL for CPU-only use).
 * @param w Weights descriptor (expects json_path and weights_path).
 * @param hp Hyperparameters (dimensions, layer counts, etc.).
 * @param out_ctx Output context handle.
 * @return 0 on success, negative error code on failure.
 */
int ie_gptoss_infer_create(const ie_device_t *dev_const, const ie_weights_t *w, const ie_gptoss_hparams_t *hp,
                           ie_gptoss_infer_t **out_ctx) {
  if (!out_ctx) return -1;
  *out_ctx = NULL;
  if (!w || !hp) return -2;

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)calloc(1, sizeof(*impl));
  if (!impl) return -3;

  impl->dev = (ie_device_t *)dev_const;
  impl->weights = w;
  impl->hp = hp;
  impl->pos = 0;
  impl->rms_eps = IE_GPTOSS_RMS_EPS_DEFAULT;
  impl->rope_theta = IE_GPTOSS_ROPE_THETA_DEFAULT;
  impl->max_experts = 0;

  char *model_dir = ie_dirname_alloc(w->json_path);
  char *tmap_path = model_dir ? ie_path_join_alloc(model_dir, "tensor_map.json") : NULL;

  if (!model_dir || !tmap_path) {
    free(tmap_path);
    free(model_dir);
    free(impl);
    return -4;
  }

  memset(&impl->tmap, 0, sizeof(impl->tmap));
  if (tensor_map_load(tmap_path, &impl->tmap) != 0) {
    free(tmap_path);
    free(model_dir);
    free(impl);
    return -5;
  }

  if (ie_mmap_ro(w->weights_path, &impl->bin_base, &impl->bin_size) != 0) {
    tensor_map_free(&impl->tmap);
    free(tmap_path);
    free(model_dir);
    free(impl);
    return -6;
  }

  free(tmap_path);
  free(model_dir);

  const uint32_t d_model = hp->d_model;
  const uint32_t vocab = hp->vocab_size;
  const uint32_t n_layers = hp->n_layers;
  const uint32_t d_head = hp->d_head;
  const uint32_t n_heads = hp->n_heads;
  const uint32_t n_kv_heads = hp->n_kv_heads;
  const uint32_t d_ff = hp->d_ff;

  {
    ie_tensor_view_t v;
    const char *const embed_candidates[] = {"model.embed_tokens.weight", "transformer.wte.weight", "tok_embeddings.weight"};
    if (ie_w_get_first_view(impl, embed_candidates, 3, &v) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -7;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)vocab * (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -8;
    }
    impl->w_embed_bf16 = (const uint16_t *)v.ptr;
  }

  {
    ie_tensor_view_t v;
    const char *const norm_candidates[] = {"model.norm.weight", "transformer.ln_f.weight", "norm.weight"};
    if (ie_w_get_first_view(impl, norm_candidates, 3, &v) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -9;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -10;
    }
    impl->w_norm_bf16 = (const uint16_t *)v.ptr;
  }

  {
    ie_tensor_view_t v;
    const char *const lm_candidates[] = {"lm_head.weight", "model.lm_head.weight", "transformer.lm_head.weight",
                                         "model.embed_tokens.weight"};
    if (ie_w_get_first_view(impl, lm_candidates, 4, &v) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -11;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)vocab * (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -12;
    }
    impl->w_lm_bf16 = (const uint16_t *)v.ptr;
  }

  impl->layers = (ie_gptoss_layer_w_t *)calloc((size_t)n_layers, sizeof(*impl->layers));
  if (!impl->layers) {
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -13;
  }

  const uint64_t q_dim = (uint64_t)n_heads * (uint64_t)d_head;
  const uint64_t kv_dim = (uint64_t)n_kv_heads * (uint64_t)d_head;

  for (uint32_t l = 0; l < n_layers; ++l) {
    ie_gptoss_layer_w_t *LW = &impl->layers[l];
    ie_tensor_view_t v;

    const char *const ln1_fmts[] = {"model.layers.%u.input_layernorm.weight", "model.layers.%u.attention_norm.weight"};
    if (ie_w_get_layer_fmt_view(impl, l, ln1_fmts, 2, &v, 1) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -14;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -15;
    }
    LW->ln1_w = (const uint16_t *)v.ptr;

    const char *const q_w_fmts[] = {"model.layers.%u.self_attn.q_proj.weight"};
    const char *const q_b_fmts[] = {"model.layers.%u.self_attn.q_proj.bias"};
    if (ie_w_get_layer_fmt_view(impl, l, q_w_fmts, 1, &v, 1) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -16;
    }
    if (!v.desc || ie_require_size_eq(v.desc, q_dim * (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -17;
    }
    LW->q_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, q_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, q_dim * 2u) != 0) {
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
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -19;
    }
    if (!v.desc || ie_require_size_eq(v.desc, kv_dim * (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -20;
    }
    LW->k_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, k_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, kv_dim * 2u) != 0) {
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
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -22;
    }
    if (!v.desc || ie_require_size_eq(v.desc, kv_dim * (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -23;
    }
    LW->v_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, v_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, kv_dim * 2u) != 0) {
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
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -25;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * q_dim * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -26;
    }
    LW->o_w = (const uint16_t *)v.ptr;

    if (ie_w_get_layer_fmt_view(impl, l, o_b_fmts, 1, &v, 0) == 0 && v.desc) {
      if (ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
        ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
        return -27;
      }
      LW->o_b = (const uint16_t *)v.ptr;
    } else {
      LW->o_b = NULL;
    }

    const char *const ln2_fmts[] = {"model.layers.%u.post_attention_layernorm.weight", "model.layers.%u.ffn_norm.weight"};
    if (ie_w_get_layer_fmt_view(impl, l, ln2_fmts, 2, &v, 1) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -28;
    }
    if (!v.desc || ie_require_size_eq(v.desc, (uint64_t)d_model * 2u) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -29;
    }
    LW->ln2_w = (const uint16_t *)v.ptr;

    if (ie_resolve_moe_layer(impl, l, LW) != 0) {
      ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
      return -30;
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
  impl->mlp_up = (float *)malloc(d_ff_sz * sizeof(float));
  impl->scores = (float *)malloc((size_t)hp->max_seq * sizeof(float));

  impl->router_logits =
      (impl->max_experts > 0u) ? (float *)malloc((size_t)impl->max_experts * sizeof(float)) : NULL;

  if (!impl->x || !impl->x1 || !impl->x2 || !impl->q || !impl->k || !impl->v || !impl->attn_out ||
      !impl->mlp_gate || !impl->mlp_up || !impl->scores || (impl->max_experts > 0u && !impl->router_logits)) {
    ie_gptoss_infer_destroy((ie_gptoss_infer_t *)impl);
    return -31;
  }

  *out_ctx = (ie_gptoss_infer_t *)impl;
  return 0;
}

/**
 * @brief Destroy a GPT-OSS inference context.
 * @param ctx Context handle (may be NULL).
 */
void ie_gptoss_infer_destroy(ie_gptoss_infer_t *ctx) {
  if (!ctx) return;
  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  if (impl->bin_base) ie_munmap_ro(impl->bin_base, impl->bin_size);
  tensor_map_free(&impl->tmap);

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

/**
 * @brief Prefill KV cache by running a prompt through the model.
 *
 * Resets internal position to 0 and runs tokens sequentially. The output logits
 * correspond to the final processed token.
 *
 * @param ctx Context handle.
 * @param kv Array of KV caches (one per layer).
 * @param prompt Token ids.
 * @param n_prompt Number of tokens in prompt.
 * @param out_logits Output logits (FP32), length vocab.
 * @return 0 on success, negative error code on failure.
 */
int ie_gptoss_infer_prefill(ie_gptoss_infer_t *ctx, ie_kv_cache *kv, const uint32_t *prompt,
                           uint32_t n_prompt, float *out_logits) {
  if (!ctx || !kv || (!prompt && n_prompt > 0) || !out_logits) return -1;

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  impl->pos = 0;
  if (n_prompt == 0u) return -2;

  const uint32_t limit = (uint32_t)ie_min_size((size_t)n_prompt, (size_t)impl->hp->max_seq);

  int rc = 0;
  for (uint32_t i = 0; i < limit; ++i) {
    rc = ie_forward_one_token(impl, kv, prompt[i], impl->pos, out_logits);
    if (rc != 0) return rc;
    impl->pos += 1u;
  }

  return 0;
}

/**
 * @brief Single decode step: run one token and advance internal position.
 *
 * @param ctx Context handle.
 * @param kv Array of KV caches (one per layer).
 * @param token_id Current token id.
 * @param out_logits Output logits (FP32), length vocab.
 * @return 0 on success, negative error code on failure.
 */
int ie_gptoss_infer_step(ie_gptoss_infer_t *ctx, ie_kv_cache *kv, uint32_t token_id,
                         float *out_logits) {
  if (!ctx || !kv || !out_logits) return -1;

  struct ie_gptoss_infer_impl *impl = (struct ie_gptoss_infer_impl *)ctx;

  const uint32_t pos = impl->pos;
  const int rc = ie_forward_one_token(impl, kv, token_id, pos, out_logits);
  if (rc != 0) return rc;

  impl->pos = pos + 1u;
  return 0;
}
