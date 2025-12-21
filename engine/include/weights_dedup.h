#ifndef IE_WEIGHTS_DEDUP_H
#define IE_WEIGHTS_DEDUP_H

/**
 * @file weights_dedup.h
 * @brief Public API for the lossless deduplicated weights loader.
 *
 * @details
 * This loader:
 *  - reads `model.dedup.json` from a model directory,
 *  - mmaps the referenced binary blobs,
 *  - provides "views" for each tensor (defaults/mask/exceptions),
 *  - supports lossless materialization into a contiguous buffer.
 *
 * It is intentionally self-contained (no external JSON library requirement).
 */

#include <stddef.h>
#include <stdint.h>

#include "dedup_spec.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle for the dedup weights loader. */
typedef struct ie_weights_dedup ie_weights_dedup_t;

/**
 * @enum ie_wdedup_status_t
 * @brief Error/status codes for the dedup loader.
 */
typedef enum ie_wdedup_status_t {
  IE_WDEDUP_OK = 0,
  IE_WDEDUP_EINVAL,
  IE_WDEDUP_ENOMEM,
  IE_WDEDUP_ENOENT,
  IE_WDEDUP_EFORMAT,
  IE_WDEDUP_ERANGE,
  IE_WDEDUP_EIO
} ie_wdedup_status_t;

/**
 * @enum ie_weight_view_kind_t
 * @brief View kinds returned by ie_weights_dedup_get_weight_view().
 */
typedef enum ie_weight_view_kind_t {
  /** Direct contiguous bytes; no mask/exceptions needed. */
  IE_WVIEW_DIRECT = 0,
  /** Dedup view: defaults + mask + exceptions. */
  IE_WVIEW_DEDUP
} ie_weight_view_kind_t;

/**
 * @struct ie_weight_view_t
 * @brief A lightweight view into the weight bytes.
 */
typedef struct ie_weight_view_t {
  ie_weight_view_kind_t kind;

  ie_wdtype_t dtype;
  int32_t rank;
  int64_t shape[IE_DEDUP_MAX_RANK];

  /** Total logical element count. */
  size_t elem_count;

  /** Element size for non-int4; 0 for int4. */
  size_t elem_size;

  /** For IE_WVIEW_DIRECT: pointer/size. */
  const uint8_t* data;
  size_t nbytes;

  /** For IE_WVIEW_DEDUP: defaults/mask/exceptions pointers/sizes. */
  const uint8_t* defaults;
  size_t defaults_nbytes;

  const uint8_t* mask;
  size_t mask_nbytes;

  const uint8_t* exceptions;
  size_t exceptions_nbytes;
} ie_weight_view_t;

/**
 * @struct ie_weights_dedup_opts_t
 * @brief Loader options.
 */
typedef struct ie_weights_dedup_opts_t {
  /**
   * Prefault policy for mmapped blobs:
   *  0 = none,
   *  1 = sequential,
   *  2 = will-need
   */
  int prefault_policy;
} ie_weights_dedup_opts_t;

/**
 * @brief Open the dedup weights loader from a model directory.
 *
 * @param out Output handle.
 * @param model_dir Model directory containing model.dedup.json and blobs.
 * @param opts Options (may be NULL).
 * @return Status code.
 */
ie_wdedup_status_t ie_weights_dedup_open(ie_weights_dedup_t** out,
                                        const char* model_dir,
                                        const ie_weights_dedup_opts_t* opts);

/**
 * @brief Close the loader and free all associated memory.
 *
 * @param h Handle pointer.
 */
void ie_weights_dedup_close(ie_weights_dedup_t** h);

/**
 * @brief Get a weight view by tensor name.
 *
 * @param h Loader handle.
 * @param name Tensor name.
 * @param out Output view.
 * @return Status code.
 */
ie_wdedup_status_t ie_weights_dedup_get_weight_view(const ie_weights_dedup_t* h,
                                                    const char* name,
                                                    ie_weight_view_t* out);

/**
 * @brief Materialize a view into a contiguous destination buffer.
 *
 * @details
 * For IE_WVIEW_DIRECT, this copies bytes directly.
 * For IE_WVIEW_DEDUP, this reconstructs the exact original tensor bytes.
 *
 * @param view View to materialize.
 * @param dst Destination buffer.
 * @param dst_nbytes Capacity of destination buffer.
 * @return Bytes written, or 0 on failure.
 */
size_t ie_weights_dedup_materialize(const ie_weight_view_t* view,
                                    void* dst,
                                    size_t dst_nbytes);

/**
 * @brief Convert a status code to a short stable string.
 *
 * @param st Status.
 * @return String literal.
 */
const char* ie_wdedup_status_str(ie_wdedup_status_t st);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_WEIGHTS_DEDUP_H */

