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
 *  - A direct contiguous tensor view, or
 *  - A deduplicated tensor view referencing (defaults + mask + exceptions).
 *
 * Mask semantics:
 *  - bit=0 => use defaults value
 *  - bit=1 => consume next value from exceptions stream and override
 *
 * For int4 tensors, elements are nibbles (4-bit values) in row-major order.
 *
 * @note This header intentionally does not depend on engine-global tensor metadata types.
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
 */
#ifndef IE_WDEDUP_MAX_RANK
#define IE_WDEDUP_MAX_RANK 8
#endif

/**
 * @brief A uniform weight view describing either direct or deduplicated storage.
 *
 * For IE_WVIEW_DIRECT:
 *  - data points to contiguous tensor bytes
 *  - nbytes is the tensor byte size
 *
 * For IE_WVIEW_DEDUP:
 *  - defaults points to defaults tensor bytes
 *  - mask is a bitset over elements (or nibbles for int4)
 *    - required length is ceil(elem_count / 8) bytes
 *  - exceptions points to packed exception values in consumption order
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
   * 0 => no advice (default)
   * 1 => hint sequential read
   * 2 => hint will-need / prefault (where supported)
   */
  int prefault_policy;
} ie_weights_dedup_opts_t;

/**
 * @brief Open a deduplicated model directory.
 */
ie_wdedup_status_t ie_weights_dedup_open(ie_weights_dedup_t **out,
                                        const char *model_dir,
                                        const ie_weights_dedup_opts_t *opts);

/**
 * @brief Close a deduplicated-weights handle and release mappings.
 */
void ie_weights_dedup_close(ie_weights_dedup_t **h);

/**
 * @brief Lookup a tensor by name and return a uniform view.
 *
 * The view's pointers remain valid until ie_weights_dedup_close().
 */
ie_wdedup_status_t ie_weights_dedup_get_weight_view(const ie_weights_dedup_t *h,
                                                    const char *name,
                                                    ie_weight_view_t *out);

/**
 * @brief Materialize a weight view into a contiguous buffer (lossless).
 */
size_t ie_weights_dedup_materialize(const ie_weight_view_t *view, void *dst, size_t dst_nbytes);

/**
 * @brief Convert a dtype string ("fp32", "int4", ...) to ie_wdtype_t.
 */
ie_wdtype_t ie_weights_dedup_parse_dtype(const char *s);

#ifdef __cplusplus
} /* extern "C" */
#endif

