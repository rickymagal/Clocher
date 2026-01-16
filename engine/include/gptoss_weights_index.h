/* ============================================================================
 * File: engine/include/gptoss_weights_index.h
 * ============================================================================
 */
/**
 * @file gptoss_weights_index.h
 * @brief GPT-OSS weight indexing and name-resolution over tensor_map.json and (optional) dedup views.
 *
 * @details
 * Real inference requires fast access to model tensors (embedding, per-layer
 * projection matrices, norms, and the LM head). In this repository, tensor
 * payloads are described by `tensor_map.json` (logical name → offset/shape/dtype)
 * and stored in `model.ie.bin`. When IE_DEDUP is enabled, tensors may be stored
 * losslessly in auxiliary blobs and materialized through the dedup loader.
 *
 * This module provides:
 *  - A loader for `tensor_map.json`.
 *  - A stable, explicit mapping of logical model components to resolved tensor names.
 *  - A small "tensor handle" abstraction that supports:
 *      * direct pointers into `model.ie.bin` (non-dedup),
 *      * dedup views (direct or reconstructable),
 *      * best-effort shape/dtype validation.
 *
 * The forward pass implementation should:
 *  - call ::gptoss_weights_index_open once per model,
 *  - call ::gptoss_weights_index_build_model once to resolve all needed tensors,
 *  - then reuse the returned ::gptoss_model_weights_t throughout inference.
 */

#ifndef GPTOSS_WEIGHTS_INDEX_H
#define GPTOSS_WEIGHTS_INDEX_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "ie_infer.h"
#include "ie_io.h"
#include "tensor_map.h"
#include "weights_dedup.h"

/* ============================================================================
 * Public model/arch description
 * ========================================================================== */

/**
 * @enum gptoss_arch_kind_t
 * @brief Logical weight naming scheme detected in tensor_map.json.
 *
 * @details
 * GPT-OSS snapshots can be exported with different naming conventions. This
 * module probes for common layouts and records the detected convention so the
 * forward pass can choose the correct projection wiring (fused QKV vs. separate,
 * SwiGLU vs. GeLU MLP, etc.).
 */
typedef enum gptoss_arch_kind_t {
  GPTOSS_ARCH_UNKNOWN = 0, /**< Unrecognized naming scheme. */
  GPTOSS_ARCH_LLAMA   = 1, /**< LLaMA-style: model.layers.N.* */
  GPTOSS_ARCH_GPTNEOX = 2  /**< GPT-NeoX-style: gpt_neox.layers.N.* */
} gptoss_arch_kind_t;

/**
 * @struct gptoss_tensor_t
 * @brief A resolved tensor handle.
 *
 * @details
 * The underlying bytes may be available directly (non-dedup or IE_WVIEW_DIRECT),
 * or reconstructable via IE_WVIEW_DEDUP + ::ie_weights_dedup_materialize.
 *
 * The descriptor pointer refers to an entry in ::tensor_map_t and remains valid
 * for the lifetime of the index.
 */
typedef struct gptoss_tensor_t {
  /** Tensor descriptor from tensor_map.json (shape/dtype/size/offset). */
  const tensor_desc_t *desc;

  /** Non-zero if this tensor must be accessed through the dedup loader. */
  int is_dedup;

  /** Valid when is_dedup != 0. */
  ie_weight_view_t view;

  /**
   * Direct pointer to bytes when available:
   *  - non-dedup: pointer into mapped `model.ie.bin`,
   *  - dedup direct: view.data.
   * NULL when the tensor requires materialization.
   */
  const uint8_t *direct;

  /** Byte size of the logical tensor payload. */
  size_t nbytes;
} gptoss_tensor_t;

/**
 * @struct gptoss_layer_weights_t
 * @brief Per-layer weight handles for a GPT-OSS transformer block.
 */
typedef struct gptoss_layer_weights_t {
  /* Attention + norms */
  gptoss_tensor_t attn_norm_w; /**< Input (pre-attention) norm weight. */
  gptoss_tensor_t mlp_norm_w;  /**< Post-attention norm weight. */

  /* Attention projections */
  gptoss_tensor_t q_proj_w;    /**< Q projection (separate) OR empty if fused. */
  gptoss_tensor_t k_proj_w;    /**< K projection (separate) OR empty if fused. */
  gptoss_tensor_t v_proj_w;    /**< V projection (separate) OR empty if fused. */
  gptoss_tensor_t qkv_proj_w;  /**< Fused QKV projection OR empty if separate. */
  gptoss_tensor_t o_proj_w;    /**< Output projection. */

  /* MLP projections */
  gptoss_tensor_t gate_proj_w; /**< SwiGLU gate projection (optional). */
  gptoss_tensor_t up_proj_w;   /**< SwiGLU up projection (optional). */
  gptoss_tensor_t down_proj_w; /**< Down projection (SwiGLU) OR output dense (GeLU). */
  gptoss_tensor_t fc_in_w;     /**< GeLU-style in projection (optional). */
} gptoss_layer_weights_t;

/**
 * @struct gptoss_model_weights_t
 * @brief Resolved weight handles for the full model.
 *
 * @details
 * This is a pure index: it does not own file mappings or dedup state. Ownership
 * remains in ::gptoss_weights_index_t. Call ::gptoss_model_weights_free to free
 * the per-layer array allocated by ::gptoss_weights_index_build_model.
 */
typedef struct gptoss_model_weights_t {
  gptoss_arch_kind_t arch; /**< Detected naming scheme. */

  /** Non-zero if attention uses a fused QKV projection tensor. */
  int attn_fused_qkv;

  /** Non-zero if MLP uses SwiGLU (gate+up+down). */
  int mlp_swiglu;

  /** Token embedding weights. */
  gptoss_tensor_t tok_embed_w;

  /** Optional absolute position embedding weights (rare; RoPE models omit). */
  gptoss_tensor_t pos_embed_w;

  /** Final norm weight. */
  gptoss_tensor_t final_norm_w;

  /** LM head weights (may be tied to tok_embed_w if absent). */
  gptoss_tensor_t lm_head_w;

  /** Layer array (length = hp->n_layers). */
  gptoss_layer_weights_t *layers;
  uint32_t n_layers;
} gptoss_model_weights_t;

/* ============================================================================
 * Index handle
 * ========================================================================== */

/**
 * @struct gptoss_weights_index_t
 * @brief Loaded tensor_map + mapped weights binary + optional dedup loader handle.
 *
 * @details
 * The index owns:
 *  - the loaded ::tensor_map_t,
 *  - a mapping (mmap or malloc buffer) of `model.ie.bin`,
 *  - an optional dedup loader handle (either borrowed from ::ie_weights_t or opened locally).
 *
 * The index must outlive any ::gptoss_model_weights_t built from it.
 */
typedef struct gptoss_weights_index_t gptoss_weights_index_t;

/* ============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Open a weights index for a given model directory and weights descriptor.
 *
 * @details
 * This loads `tensor_map.json` and maps `model.ie.bin`.
 *
 * If `weights->is_dedup != 0` and `weights->dedup_handle != NULL`, this function
 * will reuse that dedup loader (borrowed, not owned). If dedup is requested but
 * no handle is present, it attempts to open a dedup loader from @p model_dir.
 *
 * @param out       Output index (written on success).
 * @param model_dir Model directory (used to locate tensor_map.json and dedup files).
 * @param weights   Weights descriptor opened by ::ie_weights_open.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_weights_index_open(gptoss_weights_index_t *out,
                             const char *model_dir,
                             const ie_weights_t *weights);

/**
 * @brief Close an index and free all owned resources.
 *
 * @param idx Index handle (may be NULL).
 */
void gptoss_weights_index_close(gptoss_weights_index_t *idx);

/**
 * @brief Resolve and validate all tensors required by the GPT-OSS forward pass.
 *
 * @details
 * This performs name-resolution for a known set of tensors (embedding, norms,
 * per-layer projections, and LM head). It allocates `out->layers` and fills
 * every ::gptoss_tensor_t with a descriptor and a pointer/view.
 *
 * Missing optional tensors (e.g., pos_embed, lm_head when tied) are left empty
 * (desc == NULL). Missing required tensors produce an error.
 *
 * @param idx  Opened weights index.
 * @param hp   Hyperparameters (used for layer count).
 * @param out  Output model weights index (written on success).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_weights_index_build_model(const gptoss_weights_index_t *idx,
                                    const ie_gptoss_hparams_t *hp,
                                    gptoss_model_weights_t *out);

/**
 * @brief Free the per-layer array allocated by ::gptoss_weights_index_build_model.
 *
 * @details
 * This does not close the underlying ::gptoss_weights_index_t.
 *
 * @param mw Model weights.
 */
void gptoss_model_weights_free(gptoss_model_weights_t *mw);

/**
 * @brief Check whether a tensor handle is populated.
 *
 * @param t Tensor handle.
 * @return Non-zero if populated, 0 otherwise.
 */
int gptoss_tensor_is_valid(const gptoss_tensor_t *t);

/**
 * @brief Return a direct pointer to tensor bytes if available.
 *
 * @details
 * For IE_WVIEW_DEDUP tensors, this returns NULL; callers must use
 * ::gptoss_tensor_materialize.
 *
 * @param t Tensor handle.
 * @return Pointer to bytes, or NULL if materialization is required.
 */
const uint8_t *gptoss_tensor_bytes(const gptoss_tensor_t *t);

/**
 * @brief Materialize tensor bytes into a caller-provided buffer.
 *
 * @details
 * - For non-dedup tensors, this copies from the mapped `model.ie.bin`.
 * - For IE_WVIEW_DIRECT, this copies from `view.data`.
 * - For IE_WVIEW_DEDUP, this reconstructs bytes via ::ie_weights_dedup_materialize.
 *
 * @param t          Tensor handle.
 * @param dst        Destination buffer.
 * @param dst_nbytes Capacity of destination buffer.
 * @return Number of bytes written (0 on failure).
 */
size_t gptoss_tensor_materialize(const gptoss_tensor_t *t, void *dst, size_t dst_nbytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPTOSS_WEIGHTS_INDEX_H */
