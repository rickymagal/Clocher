/* engine/include/io/weights_dedup.h */
/**
 * @file weights_dedup.h
 * @brief Lossless deduplicated-weights loader (mmap-friendly) and uniform weight view API.
 *
 * This module loads a deduplicated model representation described by `model.dedup.json`
 * and maps the backing binary blobs (defaults/exceptions/masks) using `mmap(2)`.
 *
 * The loader exposes a uniform `ie_weights_dedup_get_weight_view()` API returning an
 * `ie_weight_view_t` that can represent either:
 *  - A direct contiguous tensor view (rare in dedup artifacts but supported), or
 *  - A deduplicated tensor view referencing (defaults + mask + exceptions).
 *
 * The intent is to keep this loader mmap-friendly:
 *  - all large blobs are memory-mapped read-only
 *  - tensor views are lightweight pointers into those mappings
 *
 * A reference materializer (`ie_weights_dedup_materialize()`) is provided primarily
 * for correctness tests (hash comparison against the non-dedup baseline path) and
 * as a debugging aid.
 *
 * JSON format expectations (minimal):
 *  - A top-level "files" object mapping logical names to relative file paths:
 *      { "files": { "defaults": "...bin", "exceptions": "...bin", "masks": "...bin" }, ... }
 *  - A top-level "tensors" array, each with:
 *      - "name" (string)
 *      - "dtype" (string) : "fp32","fp16","bf16","int8","int4" (others may be added later)
 *      - "shape" (int array)
 *      - "default" / "mask" / "exceptions" objects:
 *          { "file": "defaults", "offset": N, "nbytes": M }
 *
 * The mask is interpreted as a bitset over *elements* (not bytes):
 *  - bit=0 => element equals defaults
 *  - bit=1 => element is overridden by the next value in exceptions (packed sequentially)
 *
 * For int4 tensors, "elements" refers to nibbles (4-bit values) in row-major order.
 *
 * @note This header intentionally does not depend on engine-global tensor metadata types.
 *       The engine can bridge `ie_weight_view_t` into its internal representation.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a deduplicated-weights mapping.
 */
typedef struct ie_weights_dedup ie_weights_dedup_t;

/**
 * @brief Status codes returned by the dedup loader.
 */
typedef enum ie_wdedup_status {
  IE_WDEDUP_OK = 0,
  IE_WDEDUP_EINVAL = 1,
  IE_WDEDUP_ENOENT = 2,
  IE_WDEDUP_EIO = 3,
  IE_WDEDUP_ENOMEM = 4,
  IE_WDEDUP_EFORMAT = 5,
  IE_WDEDUP_ERANGE = 6
} ie_wdedup_status_t;

/**
 * @brief Supported dtypes for dedup views.
 */
typedef enum ie_wdtype {
  IE_WDTYPE_UNKNOWN = 0,
  IE_WDTYPE_FP32,
  IE_WDTYPE_FP16,
  IE_WDTYPE_BF16,
  IE_WDTYPE_INT8,
  IE_WDTYPE_INT4
} ie_wdtype_t;

/**
 * @brief Weight view kind.
 */
typedef enum ie_weight_view_kind {
  IE_WVIEW_DIRECT = 0, /**< Direct contiguous bytes */
  IE_WVIEW_DEDUP  = 1  /**< Defaults + mask + exceptions */
} ie_weight_view_kind_t;

/**
 * @brief Maximum rank supported for shapes carried in the view.
 *
 * This is only for metadata conveyance; the engine may ignore shape here if it
 * already has authoritative shape info elsewhere.
 */
#ifndef IE_WDEDUP_MAX_RANK
#define IE_WDEDUP_MAX_RANK 8
#endif

/**
 * @brief A uniform weight view describing either direct or deduplicated storage.
 *
 * For IE_WVIEW_DIRECT:
 *  - @c data points to the contiguous tensor bytes
 *  - @c nbytes is the tensor byte size
 *
 * For IE_WVIEW_DEDUP:
 *  - @c defaults points to the defaults tensor bytes (same dtype/shape as logical tensor)
 *  - @c mask points to a bitset over elements/nibbles
 *  - @c exceptions points to packed exception values (same dtype packing)
 *
 * Mask semantics:
 *  - bit 0 => use defaults value
 *  - bit 1 => consume next value from exceptions stream and override
 */
typedef struct ie_weight_view {
  ie_weight_view_kind_t kind;
  ie_wdtype_t dtype;

  int32_t rank;
  int64_t shape[IE_WDEDUP_MAX_RANK];

  /* Direct view */
  const void *data;
  size_t nbytes;

  /* Dedup view */
  const void *defaults;
  size_t defaults_nbytes;

  const uint8_t *mask;
  size_t mask_nbytes;

  const void *exceptions;
  size_t exceptions_nbytes;

  /* Derived info (filled by loader) */
  size_t elem_count;   /**< Number of elements (or nibbles for int4). */
  size_t elem_size;    /**< Element size in bytes; for int4 this is 0 (nibble-based). */
} ie_weight_view_t;

/**
 * @brief Options controlling mapping and paging policy.
 */
typedef struct ie_weights_dedup_opts {
  /**
   * @brief Whether to request prefaulting (best-effort) via POSIX madvise.
   *
   * 0 => no advice (default)
   * 1 => hint sequential read
   * 2 => hint will-need / prefault (where supported)
   */
  int prefault_policy;
} ie_weights_dedup_opts_t;

/**
 * @brief Open a deduplicated model directory.
 *
 * This function expects to find `model.dedup.json` inside @p model_dir.
 * It mmaps referenced files (defaults/exceptions/masks) read-only.
 *
 * @param[out] out      Returned handle on success.
 * @param[in]  model_dir Directory containing `model.dedup.json` and blobs.
 * @param[in]  opts     Optional mapping options (may be NULL).
 * @return IE_WDEDUP_OK on success, otherwise an error code.
 */
ie_wdedup_status_t ie_weights_dedup_open(ie_weights_dedup_t **out,
                                        const char *model_dir,
                                        const ie_weights_dedup_opts_t *opts);

/**
 * @brief Close a deduplicated-weights handle and release mappings.
 *
 * @param[in,out] h Handle to close (set to NULL on return).
 */
void ie_weights_dedup_close(ie_weights_dedup_t **h);

/**
 * @brief Lookup a tensor by name and return a uniform view.
 *
 * The view's pointers remain valid until @c ie_weights_dedup_close().
 *
 * @param[in]  h     Open dedup handle.
 * @param[in]  name  Tensor name.
 * @param[out] out   Filled view on success.
 * @return IE_WDEDUP_OK if found, otherwise an error code.
 */
ie_wdedup_status_t ie_weights_dedup_get_weight_view(const ie_weights_dedup_t *h,
                                                    const char *name,
                                                    ie_weight_view_t *out);

/**
 * @brief Materialize a weight view into a contiguous buffer (lossless).
 *
 * This is primarily intended for tests:
 *  - Load baseline (non-dedup) tensor bytes and hash them.
 *  - Load dedup tensor view, materialize into scratch, hash, compare.
 *
 * For IE_WVIEW_DIRECT, this performs a memcpy.
 * For IE_WVIEW_DEDUP, this reconstructs defaults + overrides according to the mask.
 *
 * @param[in]  view      Weight view (direct or dedup).
 * @param[out] dst       Destination buffer.
 * @param[in]  dst_nbytes Capacity of destination buffer in bytes.
 * @return Number of bytes written on success, 0 on failure.
 */
size_t ie_weights_dedup_materialize(const ie_weight_view_t *view, void *dst, size_t dst_nbytes);

/**
 * @brief Convert a dtype string ("fp32", "int4", ...) to ie_wdtype_t.
 *
 * @param[in] s NUL-terminated dtype string.
 * @return Parsed dtype or IE_WDTYPE_UNKNOWN.
 */
ie_wdtype_t ie_weights_dedup_parse_dtype(const char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

