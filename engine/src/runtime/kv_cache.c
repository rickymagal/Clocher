/* ============================================================================
 * File: engine/src/runtime/kv_cache.c
 * ============================================================================
 */
/**
 * @file kv_cache.c
 * @brief Implementation of a compact Key/Value attention cache with INT8/FP8 compression.
 *
 * @details
 * The cache stores per-token K/V slices laid out as [heads, head_dim] per token,
 * flattened with head_dim as the fastest-varying dimension:
 *
 *   index(t,h,d) = ((t * H + h) * D + d)
 *
 * For INT8 storage, per-group affine parameters are kept for each (token, head).
 * The parameter array index is:
 *
 *   param_index(t,h,g) = ((t * H + h) * G + g)
 *
 * where G = ceil(D / group_size).
 *
 * Layered KV specification:
 *  - Transformer inference uses an array of caches, one per layer.
 *  - This module can allocate a single contiguous backing slab for all layers
 *    and slice it into per-layer cache views via ::ie_kv_init_layers().
 *
 * Instrumentation:
 *  - This module can optionally report "KV hits" and "KV misses" via metrics
 *    hooks (IE_KV_HIT / IE_KV_MISS). Safe no-op fallbacks are provided so the
 *    file compiles regardless of whether the build exposes these macros.
 */

#include "ie_kv_cache.h"
#include "ie_quant_act.h"

/* Optional: metrics hooks (safe fallbacks). */
#include "ie_metrics.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ----------------------------- instrumentation ---------------------------- */

/**
 * @brief Report KV hit to metrics layer if enabled.
 *
 * @details
 * If IE_KV_HIT is not defined, this is a safe no-op.
 *
 * @param n Count to add.
 */
static inline void ie_kv_hit_add_local(uint64_t n) {
#if defined(IE_KV_HIT)
  IE_KV_HIT(n);
#else
  (void)n;
#endif
}

/**
 * @brief Report KV miss to metrics layer if enabled.
 *
 * @details
 * If IE_KV_MISS is not defined, this is a safe no-op.
 *
 * @param n Count to add.
 */
static inline void ie_kv_miss_add_local(uint64_t n) {
#if defined(IE_KV_MISS)
  IE_KV_MISS(n);
#else
  (void)n;
#endif
}

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Align @p x up to @p a (power-of-two).
 *
 * @param x Value.
 * @param a Alignment (power-of-two).
 * @return Aligned value.
 */
static size_t align_up_sz(size_t x, size_t a) {
  return (x + (a - 1u)) & ~(a - 1u);
}

/**
 * @brief Ceiling division for size_t.
 *
 * @param a Numerator.
 * @param b Denominator (must be > 0).
 * @return ceil(a/b).
 */
static inline size_t ceil_div_sz(size_t a, size_t b) {
  return (a + b - 1u) / b;
}

/**
 * @brief Compute parameter index for INT8 affine params.
 *
 * @param t Token index.
 * @param h Head index.
 * @param g Group index.
 * @param H Number of heads.
 * @param G Number of groups per head.
 * @return Flat index into parameter arrays.
 */
static inline size_t param_index(size_t t, size_t h, size_t g, size_t H, size_t G) {
  return ((t * H) + h) * G + g;
}

/**
 * @brief Check for multiplication overflow (size_t).
 *
 * @param a   Operand.
 * @param b   Operand.
 * @param out Output product.
 * @return 0 on success; non-zero on overflow or bad args.
 */
static int mul_overflow_sz(size_t a, size_t b, size_t *out) {
  if (!out) return 1;
  if (a == 0 || b == 0) {
    *out = 0;
    return 0;
  }
  if (a > (SIZE_MAX / b)) return 1;
  *out = a * b;
  return 0;
}

/**
 * @brief Check for 3-way multiplication overflow (size_t).
 *
 * @param a   Operand.
 * @param b   Operand.
 * @param c   Operand.
 * @param out Output product.
 * @return 0 on success; non-zero on overflow or bad args.
 */
static int mul3_overflow_sz(size_t a, size_t b, size_t c, size_t *out) {
  size_t ab = 0;
  if (mul_overflow_sz(a, b, &ab)) return 1;
  return mul_overflow_sz(ab, c, out);
}

/**
 * @brief Validate options and populate derived fields in @p kv (no allocation).
 *
 * @details
 * This populates:
 *  - heads/head_dim/max_seq/storage/symmetric/fp8_format
 *  - group_size/group_count (INT8 only)
 *  - elem_count = heads * max_seq * head_dim
 *
 * Pointers are cleared and ownership fields are reset.
 *
 * @param kv   Cache to populate.
 * @param opts Initialization options.
 * @return 0 on success; non-zero on invalid options.
 */
static int kv_validate_and_derive(ie_kv_cache *kv, const ie_kv_opts *opts) {
  if (!kv || !opts) return -1;
  if (opts->heads <= 0 || opts->head_dim <= 0 || opts->max_seq <= 0) return -1;
  if (opts->storage != IE_KV_STORAGE_F32 &&
      opts->storage != IE_KV_STORAGE_INT8 &&
      opts->storage != IE_KV_STORAGE_FP8) return -1;

  kv->heads      = opts->heads;
  kv->head_dim   = opts->head_dim;
  kv->max_seq    = opts->max_seq;
  kv->storage    = opts->storage;
  kv->symmetric  = opts->symmetric;
  kv->fp8_format = opts->fp8_format;

  if (kv->storage == IE_KV_STORAGE_INT8) {
    kv->group_size  = (opts->group_size > 0 ? opts->group_size : 1u);
    kv->group_count = ceil_div_sz((size_t)kv->head_dim, kv->group_size);
  } else {
    kv->group_size  = 1u;
    kv->group_count = 1u;
  }

  if (mul3_overflow_sz((size_t)kv->heads, (size_t)kv->max_seq, (size_t)kv->head_dim,
                       &kv->elem_count)) {
    return -1;
  }

  kv->K = kv->V = NULL;
  kv->scales_K = kv->scales_V = NULL;
  kv->zeros_K  = kv->zeros_V  = NULL;

  kv->scratch_scales = NULL;
  kv->scratch_zeros  = NULL;
  kv->scratch_u8     = NULL;

  kv->backing = NULL;
  kv->backing_bytes = 0;
  kv->backing_owner = 0;

  return 0;
}

/**
 * @brief Compute the backing slab size required for one cache given derived fields.
 *
 * @details
 * The slab contains, in order (with 64-byte alignment between segments):
 *  - K payload
 *  - V payload
 *  - INT8 params (scales_K, zeros_K, scales_V, zeros_V) if needed
 *  - scratch buffers (INT8 per-group scratch, FP8 scratch) if needed
 *
 * @param kv        Derived cache fields (must have elem_count/group_count set).
 * @param out_bytes Output size in bytes.
 * @return 0 on success; non-zero on error.
 */
static int kv_compute_slab_bytes(const ie_kv_cache *kv, size_t *out_bytes) {
  if (!kv || !out_bytes) return -1;
  *out_bytes = 0;

  const size_t align = 64u;
  const size_t elem_size = ie_kv_element_size(kv->storage);
  if (elem_size == 0) return -1;

  size_t payload_bytes = 0;
  if (mul_overflow_sz(kv->elem_count, elem_size, &payload_bytes)) return -1;

  size_t total = 0;

  /* K and V payloads. */
  total = align_up_sz(total, align);
  total += payload_bytes;
  total = align_up_sz(total, align);
  total += payload_bytes;

  /* INT8 params (two scales arrays + two zeros arrays). */
  if (kv->storage == IE_KV_STORAGE_INT8) {
    size_t params_len = 0;
    if (mul3_overflow_sz((size_t)kv->max_seq, (size_t)kv->heads, kv->group_count, &params_len)) {
      return -1;
    }

    const size_t scales_bytes = params_len * sizeof(float);
    const size_t zeros_bytes  = params_len * sizeof(int8_t);

    total = align_up_sz(total, align);
    total += scales_bytes; /* scales_K */
    total = align_up_sz(total, align);
    total += zeros_bytes;  /* zeros_K */
    total = align_up_sz(total, align);
    total += scales_bytes; /* scales_V */
    total = align_up_sz(total, align);
    total += zeros_bytes;  /* zeros_V */

    /* Scratch for per-group store: [G] scales + [G] zeros. */
    total = align_up_sz(total, align);
    total += kv->group_count * sizeof(float);
    total = align_up_sz(total, align);
    total += kv->group_count * sizeof(int8_t);
  }

  /* FP8 scratch: [D] bytes. */
  if (kv->storage == IE_KV_STORAGE_FP8) {
    total = align_up_sz(total, align);
    total += (size_t)kv->head_dim * sizeof(uint8_t);
  }

  *out_bytes = align_up_sz(total, align);
  return 0;
}

/**
 * @brief Slice a backing slab into pointers inside @p kv.
 *
 * @param kv         Cache to populate (must have derived fields set).
 * @param slab       Base pointer.
 * @param slab_bytes Slab size in bytes.
 * @return 0 on success; non-zero on bounds/overflow error.
 */
static int kv_assign_from_slab(ie_kv_cache *kv, uint8_t *slab, size_t slab_bytes) {
  if (!kv || !slab || slab_bytes == 0) return -1;

  const size_t align = 64u;
  const size_t elem_size = ie_kv_element_size(kv->storage);
  if (elem_size == 0) return -1;

  size_t payload_bytes = 0;
  if (mul_overflow_sz(kv->elem_count, elem_size, &payload_bytes)) return -1;

  size_t off = 0;

  /* K */
  off = align_up_sz(off, align);
  if (off + payload_bytes > slab_bytes) return -1;
  kv->K = slab + off;
  off += payload_bytes;

  /* V */
  off = align_up_sz(off, align);
  if (off + payload_bytes > slab_bytes) return -1;
  kv->V = slab + off;
  off += payload_bytes;

  if (kv->storage == IE_KV_STORAGE_INT8) {
    size_t params_len = 0;
    if (mul3_overflow_sz((size_t)kv->max_seq, (size_t)kv->heads, kv->group_count, &params_len)) {
      return -1;
    }

    const size_t scales_bytes = params_len * sizeof(float);
    const size_t zeros_bytes  = params_len * sizeof(int8_t);

    off = align_up_sz(off, align);
    if (off + scales_bytes > slab_bytes) return -1;
    kv->scales_K = (float *)(void *)(slab + off);
    off += scales_bytes;

    off = align_up_sz(off, align);
    if (off + zeros_bytes > slab_bytes) return -1;
    kv->zeros_K = (int8_t *)(void *)(slab + off);
    off += zeros_bytes;

    off = align_up_sz(off, align);
    if (off + scales_bytes > slab_bytes) return -1;
    kv->scales_V = (float *)(void *)(slab + off);
    off += scales_bytes;

    off = align_up_sz(off, align);
    if (off + zeros_bytes > slab_bytes) return -1;
    kv->zeros_V = (int8_t *)(void *)(slab + off);
    off += zeros_bytes;

    /* Scratch */
    off = align_up_sz(off, align);
    if (off + kv->group_count * sizeof(float) > slab_bytes) return -1;
    kv->scratch_scales = (float *)(void *)(slab + off);
    off += kv->group_count * sizeof(float);

    off = align_up_sz(off, align);
    if (off + kv->group_count * sizeof(int8_t) > slab_bytes) return -1;
    kv->scratch_zeros = (int8_t *)(void *)(slab + off);
    off += kv->group_count * sizeof(int8_t);
  }

  if (kv->storage == IE_KV_STORAGE_FP8) {
    off = align_up_sz(off, align);
    if (off + (size_t)kv->head_dim * sizeof(uint8_t) > slab_bytes) return -1;
    kv->scratch_u8 = (uint8_t *)(void *)(slab + off);
    off += (size_t)kv->head_dim * sizeof(uint8_t);
  }

  return 0;
}

/* --------------------------------- public API ----------------------------- */

/**
 * @brief Return the element size for a KV storage type.
 *
 * @param storage Storage type.
 * @return Element size in bytes; 0 for unknown.
 */
size_t ie_kv_element_size(ie_kv_storage_type storage) {
  switch (storage) {
    case IE_KV_STORAGE_F32:  return sizeof(float);
    case IE_KV_STORAGE_INT8: return sizeof(int8_t);
    case IE_KV_STORAGE_FP8:  return sizeof(uint8_t);
    default: return 0;
  }
}

/**
 * @brief Initialize a single KV cache and allocate its backing memory.
 *
 * @param kv   Cache to initialize.
 * @param opts Options describing cache geometry and storage format.
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_init(ie_kv_cache *kv, const ie_kv_opts *opts) {
  if (!kv) return -1;
  if (kv_validate_and_derive(kv, opts) != 0) return -1;

  size_t slab_bytes = 0;
  if (kv_compute_slab_bytes(kv, &slab_bytes) != 0) return -2;

  uint8_t *slab = (uint8_t *)calloc(1, slab_bytes);
  if (!slab) return -3;

  if (kv_assign_from_slab(kv, slab, slab_bytes) != 0) {
    free(slab);
    return -4;
  }

  kv->backing = slab;
  kv->backing_bytes = slab_bytes;
  kv->backing_owner = 1;
  return 0;
}

/**
 * @brief Initialize an array of KV cache layers backed by one contiguous slab.
 *
 * @details
 * This allocates a single slab large enough for @p n_layers caches and then
 * assigns each cache view to its slice. The first layer is marked as the owner
 * of the slab and will free it when ::ie_kv_free_layers() is called.
 *
 * @param kv_layers Array of caches, length n_layers.
 * @param n_layers  Number of layers (must be > 0).
 * @param opts      Cache options applied to every layer.
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_init_layers(ie_kv_cache *kv_layers, int n_layers, const ie_kv_opts *opts) {
  if (!kv_layers || n_layers <= 0 || !opts) return -1;

  /* Derive one layer to compute per-layer slab bytes. */
  ie_kv_cache tmp;
  memset(&tmp, 0, sizeof(tmp));
  if (kv_validate_and_derive(&tmp, opts) != 0) return -2;

  size_t per_layer_bytes = 0;
  if (kv_compute_slab_bytes(&tmp, &per_layer_bytes) != 0) return -3;

  /* Make each slice start aligned. */
  per_layer_bytes = align_up_sz(per_layer_bytes, 64u);

  size_t total_bytes = 0;
  if (mul_overflow_sz((size_t)n_layers, per_layer_bytes, &total_bytes)) return -4;

  uint8_t *slab_all = (uint8_t *)calloc(1, total_bytes);
  if (!slab_all) return -5;

  for (int l = 0; l < n_layers; ++l) {
    ie_kv_cache *kv = &kv_layers[l];
    memset(kv, 0, sizeof(*kv));
    if (kv_validate_and_derive(kv, opts) != 0) {
      free(slab_all);
      for (int j = 0; j < l; ++j) memset(&kv_layers[j], 0, sizeof(kv_layers[j]));
      return -6;
    }

    uint8_t *slice = slab_all + (size_t)l * per_layer_bytes;
    if (kv_assign_from_slab(kv, slice, per_layer_bytes) != 0) {
      free(slab_all);
      for (int j = 0; j < l; ++j) memset(&kv_layers[j], 0, sizeof(kv_layers[j]));
      return -7;
    }

    kv->backing = slab_all;
    kv->backing_bytes = total_bytes;
    kv->backing_owner = (l == 0) ? 1 : 0;
  }

  return 0;
}

/**
 * @brief Free a single KV cache.
 *
 * @details
 * Only the cache marked as owner frees the shared slab. After freeing, the
 * structure is zeroed to prevent accidental reuse.
 *
 * @param kv Cache instance.
 */
void ie_kv_free(ie_kv_cache *kv) {
  if (!kv) return;

  if (kv->backing_owner && kv->backing) {
    free(kv->backing);
  }

  memset(kv, 0, sizeof(*kv));
}

/**
 * @brief Free a layered KV cache allocation.
 *
 * @details
 * Freeing the owner (layer 0) is sufficient; other layers are cleared.
 *
 * @param kv_layers Array of layer caches.
 * @param n_layers  Number of layers.
 */
void ie_kv_free_layers(ie_kv_cache *kv_layers, int n_layers) {
  if (!kv_layers || n_layers <= 0) return;

  ie_kv_free(&kv_layers[0]);
  for (int i = 1; i < n_layers; ++i) {
    memset(&kv_layers[i], 0, sizeof(kv_layers[i]));
  }
}

/**
 * @brief Store one token's K/V into an F32 KV cache.
 *
 * @param kv    Cache.
 * @param t     Token index.
 * @param K_f32 K data (length heads*head_dim).
 * @param V_f32 V data (length heads*head_dim).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_store_token_f32(ie_kv_cache *kv, size_t t,
                          const float *K_f32, const float *V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_F32) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;
  const size_t slice = H * D;

  float *Kdst = (float *)kv->K;
  float *Vdst = (float *)kv->V;

  memcpy(Kdst + t * slice, K_f32, slice * sizeof(float));
  memcpy(Vdst + t * slice, V_f32, slice * sizeof(float));
  return 0;
}

/**
 * @brief Store one token's K/V into an INT8 KV cache using per-tensor params per head.
 *
 * @details
 * Computes a single affine scale/zero for each (token, head) using min/max
 * over the whole head_dim, then quantizes all D values. The same parameters
 * are replicated into every group slot to keep the load path consistent.
 *
 * @param kv    Cache.
 * @param t     Token index.
 * @param K_f32 K data (length heads*head_dim).
 * @param V_f32 V data (length heads*head_dim).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_store_token_int8_per_tensor(ie_kv_cache *kv, size_t t,
                                      const float *K_f32, const float *V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_INT8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;
  const size_t G = kv->group_count;

  int8_t *Kdst = (int8_t *)kv->K;
  int8_t *Vdst = (int8_t *)kv->V;

  for (size_t h = 0; h < H; ++h) {
    const float *Ksrc = K_f32 + h * D;
    const float *Vsrc = V_f32 + h * D;

    float Kmn = Ksrc[0], Kmx = Ksrc[0];
    float Vmn = Vsrc[0], Vmx = Vsrc[0];
    for (size_t d = 1; d < D; ++d) {
      const float kvv = Ksrc[d];
      const float vvv = Vsrc[d];
      if (kvv < Kmn) Kmn = kvv;
      if (kvv > Kmx) Kmx = kvv;
      if (vvv < Vmn) Vmn = vvv;
      if (vvv > Vmx) Vmx = vvv;
    }

    ie_act_i8_params pK, pV;
    ie_act_i8_params_from_minmax(Kmn, Kmx, kv->symmetric, &pK.scale, &pK.zero_point);
    ie_act_i8_params_from_minmax(Vmn, Vmx, kv->symmetric, &pV.scale, &pV.zero_point);

    const size_t off = (t * H + h) * D;
    ie_quantize_act_int8(Ksrc, Kdst + off, D, pK, kv->symmetric);
    ie_quantize_act_int8(Vsrc, Vdst + off, D, pV, kv->symmetric);

    for (size_t g = 0; g < G; ++g) {
      const size_t pi = param_index(t, h, g, H, G);
      kv->scales_K[pi] = pK.scale;
      kv->zeros_K[pi]  = pK.zero_point;
      kv->scales_V[pi] = pV.scale;
      kv->zeros_V[pi]  = pV.zero_point;
    }
  }

  return 0;
}

/**
 * @brief Store one token's K/V into an INT8 KV cache using per-group params.
 *
 * @details
 * Computes affine scale/zero per group of size group_size along head_dim and
 * stores those parameters into (token, head, group). Quantization is applied
 * group-wise to the payload.
 *
 * @param kv    Cache.
 * @param t     Token index.
 * @param K_f32 K data (length heads*head_dim).
 * @param V_f32 V data (length heads*head_dim).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_store_token_int8_per_group(ie_kv_cache *kv, size_t t,
                                     const float *K_f32, const float *V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_INT8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H  = (size_t)kv->heads;
  const size_t D  = (size_t)kv->head_dim;
  const size_t G  = kv->group_count;
  const size_t Gs = kv->group_size;

  int8_t *Kdst = (int8_t *)kv->K;
  int8_t *Vdst = (int8_t *)kv->V;

  float  *scales = kv->scratch_scales;
  int8_t *zeros  = kv->scratch_zeros;
  if (!scales || !zeros) return -4;

  for (size_t h = 0; h < H; ++h) {
    const float *Ksrc = K_f32 + h * D;
    const float *Vsrc = V_f32 + h * D;
    const size_t off  = (t * H + h) * D;

    ie_act_i8_group_params_from_data(Ksrc, D, Gs, kv->symmetric, scales, zeros);
    ie_quantize_act_int8_per_group(Ksrc, Kdst + off, D, Gs, scales, zeros, kv->symmetric);
    for (size_t g = 0; g < G; ++g) {
      const size_t pi = param_index(t, h, g, H, G);
      kv->scales_K[pi] = scales[g];
      kv->zeros_K[pi]  = zeros[g];
    }

    ie_act_i8_group_params_from_data(Vsrc, D, Gs, kv->symmetric, scales, zeros);
    ie_quantize_act_int8_per_group(Vsrc, Vdst + off, D, Gs, scales, zeros, kv->symmetric);
    for (size_t g = 0; g < G; ++g) {
      const size_t pi = param_index(t, h, g, H, G);
      kv->scales_V[pi] = scales[g];
      kv->zeros_V[pi]  = zeros[g];
    }
  }

  return 0;
}

/**
 * @brief Store one token's K/V into an FP8 KV cache.
 *
 * @details
 * Quantizes activations to FP8 using the configured fp8_format and stores the
 * packed bytes into the cache. Uses a per-head scratch buffer of length head_dim.
 *
 * @param kv    Cache.
 * @param t     Token index.
 * @param K_f32 K data (length heads*head_dim).
 * @param V_f32 V data (length heads*head_dim).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_store_token_fp8(ie_kv_cache *kv, size_t t,
                          const float *K_f32, const float *V_f32) {
  if (!kv || !K_f32 || !V_f32) return -1;
  if (kv->storage != IE_KV_STORAGE_FP8) return -2;
  if (t >= (size_t)kv->max_seq) return -3;

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;

  uint8_t *Kdst = (uint8_t *)kv->K;
  uint8_t *Vdst = (uint8_t *)kv->V;

  uint8_t *tmp = kv->scratch_u8;
  if (!tmp) return -4;

  for (size_t h = 0; h < H; ++h) {
    const float *Ksrc = K_f32 + h * D;
    const float *Vsrc = V_f32 + h * D;
    const size_t off  = (t * H + h) * D;

    ie_quantize_act_fp8(Ksrc, tmp, D, kv->fp8_format);
    memcpy(Kdst + off, tmp, D * sizeof(uint8_t));

    ie_quantize_act_fp8(Vsrc, tmp, D, kv->fp8_format);
    memcpy(Vdst + off, tmp, D * sizeof(uint8_t));
  }

  return 0;
}

/**
 * @brief Load one token's K/V into float buffers.
 *
 * @details
 * Performs dequantization for INT8/FP8 caches. On success, reports a KV hit.
 * If t is out of range, reports a KV miss and returns an error.
 *
 * @param kv    Cache (const).
 * @param t     Token index.
 * @param K_out Output buffer for K (length heads*head_dim).
 * @param V_out Output buffer for V (length heads*head_dim).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_load_token_f32(const ie_kv_cache *kv, size_t t,
                         float *K_out, float *V_out) {
  if (!kv || !K_out || !V_out) return -1;

  if (t >= (size_t)kv->max_seq) {
    ie_kv_miss_add_local(1);
    return -2;
  }

  const size_t H = (size_t)kv->heads;
  const size_t D = (size_t)kv->head_dim;
  const size_t slice = H * D;

  if (kv->storage == IE_KV_STORAGE_F32) {
    const float *Ksrc = (const float *)kv->K;
    const float *Vsrc = (const float *)kv->V;
    memcpy(K_out, Ksrc + t * slice, slice * sizeof(float));
    memcpy(V_out, Vsrc + t * slice, slice * sizeof(float));
    ie_kv_hit_add_local(1);
    return 0;
  }

  if (kv->storage == IE_KV_STORAGE_INT8) {
    const int8_t *Kq = (const int8_t *)kv->K;
    const int8_t *Vq = (const int8_t *)kv->V;
    const size_t G   = kv->group_count;
    const size_t Gs  = kv->group_size;

    for (size_t h = 0; h < H; ++h) {
      const size_t off = (t * H + h) * D;

      for (size_t d = 0; d < D; ++d) {
        const size_t g  = d / Gs;
        const size_t pi = param_index(t, h, g, H, G);
        const float s   = kv->scales_K[pi];
        const int   z   = (int)kv->zeros_K[pi];
        K_out[h * D + d] = s * ((int)Kq[off + d] - z);
      }

      for (size_t d = 0; d < D; ++d) {
        const size_t g  = d / Gs;
        const size_t pi = param_index(t, h, g, H, G);
        const float s   = kv->scales_V[pi];
        const int   z   = (int)kv->zeros_V[pi];
        V_out[h * D + d] = s * ((int)Vq[off + d] - z);
      }
    }

    ie_kv_hit_add_local(1);
    return 0;
  }

  if (kv->storage == IE_KV_STORAGE_FP8) {
    const uint8_t *K8 = (const uint8_t *)kv->K;
    const uint8_t *V8 = (const uint8_t *)kv->V;

    for (size_t h = 0; h < H; ++h) {
      const size_t off = (t * H + h) * D;
      ie_dequantize_act_fp8(K8 + off, K_out + h * D, D, kv->fp8_format);
      ie_dequantize_act_fp8(V8 + off, V_out + h * D, D, kv->fp8_format);
    }

    ie_kv_hit_add_local(1);
    return 0;
  }

  ie_kv_miss_add_local(1);
  return -3;
}

/**
 * @brief Return raw pointers to the K and V payload buffers.
 *
 * @details
 * This is intended for debug/interop. The returned pointers reference internal
 * cache storage and remain valid until the cache is freed.
 *
 * @param kv      Cache.
 * @param out_K   Output pointer for K base address (optional).
 * @param out_V   Output pointer for V base address (optional).
 * @param out_lds Output leading dimension (head_dim) (optional).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_raw_ptrs(ie_kv_cache *kv, void **out_K, void **out_V, size_t *out_lds) {
  if (!kv) return -1;
  if (out_K)   *out_K   = kv->K;
  if (out_V)   *out_V   = kv->V;
  if (out_lds) *out_lds = (size_t)kv->head_dim;
  return 0;
}

/**
 * @brief Return element strides for the cache layout.
 *
 * @details
 * Strides are returned in units of elements of the underlying storage type
 * (not bytes). The layout is:
 *  - d stride = 1
 *  - h stride = head_dim
 *  - t stride = heads * head_dim
 *
 * @param kv       Cache (const).
 * @param stride_t Output stride for token index (optional).
 * @param stride_h Output stride for head index (optional).
 * @param stride_d Output stride for dim index (optional).
 * @return 0 on success; non-zero on failure.
 */
int ie_kv_raw_strides(const ie_kv_cache *kv, size_t *stride_t, size_t *stride_h, size_t *stride_d) {
  if (!kv) return -1;
  if (stride_d) *stride_d = 1u;
  if (stride_h) *stride_h = (size_t)kv->head_dim;
  if (stride_t) *stride_t = (size_t)kv->heads * (size_t)kv->head_dim;
  return 0;
}
