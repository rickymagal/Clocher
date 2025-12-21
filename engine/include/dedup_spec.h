#ifndef IE_DEDUP_SPEC_H
#define IE_DEDUP_SPEC_H

/**
 * @file dedup_spec.h
 * @brief Types that describe the on-disk lossless deduplicated weights layout.
 *
 * @details
 * The dedup format splits weights into:
 *  - defaults: a baseline tensor byte stream,
 *  - mask: a bitset that marks which logical elements are overridden,
 *  - exceptions: a packed stream of overridden elements.
 *
 * A tensor is reconstructed by copying defaults then patching in exceptions
 * wherever the mask bit is 1.
 *
 * The "spec" describes where each of those byte ranges live inside one or more
 * binary files. It is intentionally lightweight and loader-agnostic.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum tensor rank supported by the dedup loader/spec. */
#ifndef IE_DEDUP_MAX_RANK
#define IE_DEDUP_MAX_RANK 8
#endif

/**
 * @enum ie_wdtype_t
 * @brief Minimal dtype set used by the dedup weights layer.
 */
typedef enum ie_wdtype_t {
  IE_WDTYPE_UNKNOWN = 0,
  IE_WDTYPE_FP32,
  IE_WDTYPE_FP16,
  IE_WDTYPE_BF16,
  IE_WDTYPE_INT8,
  IE_WDTYPE_INT4
} ie_wdtype_t;

/**
 * @struct ie_dedup_blobref_t
 * @brief Reference to a contiguous byte range inside a logical file.
 */
typedef struct ie_dedup_blobref_t {
  /** Index into ie_dedup_spec_t::files[] */
  uint32_t file_index;
  /** Offset in bytes from the start of that file. */
  uint64_t offset;
  /** Number of bytes in the referenced range. */
  uint64_t nbytes;
} ie_dedup_blobref_t;

/**
 * @struct ie_dedup_tensor_t
 * @brief Specification entry for one tensor in the dedup format.
 */
typedef struct ie_dedup_tensor_t {
  /** Tensor name (NUL-terminated). Owned by the spec object. */
  char* name;

  /** Tensor dtype. */
  ie_wdtype_t dtype;

  /** Tensor rank. */
  int32_t rank;

  /** Tensor shape (length = rank). */
  int64_t shape[IE_DEDUP_MAX_RANK];

  /** Defaults payload location. */
  ie_dedup_blobref_t defaults;

  /** Mask payload location. */
  ie_dedup_blobref_t mask;

  /** Exceptions payload location. */
  ie_dedup_blobref_t exceptions;

  /** Logical element count (product(shape)). */
  size_t elem_count;

  /**
   * Size in bytes of one logical element for non-int4 dtypes.
   * For int4, elem_size is 0 (nibbles are packed).
   */
  size_t elem_size;
} ie_dedup_tensor_t;

/**
 * @struct ie_dedup_file_t
 * @brief One file listed in the spec.
 */
typedef struct ie_dedup_file_t {
  /** Logical name used in the JSON. */
  char* logical;
  /** Relative path (as stored in JSON). */
  char* relpath;
} ie_dedup_file_t;

/**
 * @struct ie_dedup_spec_t
 * @brief Full dedup spec describing all files and tensors.
 */
typedef struct ie_dedup_spec_t {
  /** Model directory the spec was loaded from (NUL-terminated). */
  char* model_dir;

  /** File table. */
  ie_dedup_file_t* files;
  size_t files_count;

  /** Tensor table (usually sorted by name for lookup). */
  ie_dedup_tensor_t* tensors;
  size_t tensors_count;
} ie_dedup_spec_t;

/**
 * @brief Parse a dtype string into ie_wdtype_t.
 *
 * @param s Dtype string (e.g. "fp16", "int4").
 * @return Parsed dtype or IE_WDTYPE_UNKNOWN.
 */
ie_wdtype_t ie_dedup_parse_dtype(const char* s);

/**
 * @brief Return the element size (bytes) for a dtype.
 *
 * @details
 * For int4 this returns 0 because elements are packed into nibbles.
 *
 * @param dt Dtype.
 * @return Size in bytes (0 for int4 / unknown).
 */
size_t ie_dedup_dtype_elem_size(ie_wdtype_t dt);

/**
 * @brief Compute element count from shape.
 *
 * @param shape Shape array.
 * @param rank Rank.
 * @param out_elem_count Output element count.
 * @return 1 on success, 0 on overflow/invalid.
 */
int ie_dedup_shape_elem_count(const int64_t* shape, int32_t rank, size_t* out_elem_count);

/**
 * @brief Free all heap memory owned by a spec and set *spec to NULL.
 *
 * @param spec Pointer to spec pointer.
 */
void ie_dedup_spec_free(ie_dedup_spec_t** spec);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEDUP_SPEC_H */

