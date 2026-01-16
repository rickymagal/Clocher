/* ============================================================================
 * File: engine/include/tensor_map.h
 * ============================================================================
 */
#ifndef TENSOR_MAP_H_
#define TENSOR_MAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @file tensor_map.h
 * @brief Loader and lookup API for tensor_map.json.
 *
 * @details
 * tensor_map.json describes how logical tensor names map to byte offsets,
 * shapes, and dtypes inside model.ie.bin.
 *
 * Supported input shapes:
 *  - Array style: { "tensors": [ { "name": "...", "offset": ..., ... }, ... ] }
 *  - Direct array: [ { "name": "...", ... }, ... ]
 *  - Map style: { "tensor.name": { "offset": ..., ... }, ... }
 *
 * Logging philosophy:
 * - The map container is "data-only"; it does not print by itself.
 * - Callers should log failures with tensor_desc_t using the helper formatters
 *   provided below (dtype name, shape string, and a compact debug string).
 */

/* ------------------------------------------------------------------------- */
/* Tensor descriptor                                                          */
/* ------------------------------------------------------------------------- */

/**
 * @brief Tensor data type tag.
 *
 * @note
 * tensor_map.json may omit dtype; loaders may leave dtype as TENSOR_DTYPE_UNKNOWN.
 * Callers must not assume dtype is populated unless their loader guarantees it.
 */
typedef enum tensor_dtype_e {
  TENSOR_DTYPE_F32 = 0,
  TENSOR_DTYPE_F16 = 1,
  TENSOR_DTYPE_INT4 = 2,
  TENSOR_DTYPE_BF16 = 3,
  TENSOR_DTYPE_U8 = 4,
  TENSOR_DTYPE_UNKNOWN = 255
} tensor_dtype_t;

/**
 * @brief Return the byte size for a dtype when representable as plain elements.
 *
 * @param dt Dtype tag.
 * @return Element size in bytes for dense element types; 0 for packed/unknown.
 */
static inline size_t tensor_dtype_size_bytes(tensor_dtype_t dt) {
  switch (dt) {
    case TENSOR_DTYPE_F32: return 4u;
    case TENSOR_DTYPE_F16: return 2u;
    case TENSOR_DTYPE_BF16: return 2u;
    case TENSOR_DTYPE_U8:  return 1u;
    case TENSOR_DTYPE_INT4: return 0u; /* packed, size is tensor-specific */
    default: return 0u;
  }
}

/**
 * @brief Convert dtype tag to a stable human-readable name.
 *
 * @param dt Dtype tag.
 * @return Constant string name.
 */
static inline const char *tensor_dtype_name(tensor_dtype_t dt) {
  switch (dt) {
    case TENSOR_DTYPE_F32: return "f32";
    case TENSOR_DTYPE_F16: return "f16";
    case TENSOR_DTYPE_INT4: return "int4";
    case TENSOR_DTYPE_BF16: return "bf16";
    case TENSOR_DTYPE_U8:  return "u8";
    default: return "unknown";
  }
}

/**
 * @brief Tensor descriptor resolved from tensor_map.json.
 */
typedef struct tensor_desc_s {
  /** @brief Tensor name (owned, heap). */
  char           *name;
  /** @brief Byte offset in model.ie.bin. */
  uint64_t        offset;
  /** @brief Size in bytes. */
  uint64_t        size_bytes;
  /** @brief Dtype tag (may be unknown if absent in json). */
  tensor_dtype_t  dtype;

  /** @brief Shape dims array (owned, heap), may be NULL when absent. */
  uint32_t       *shape;
  /** @brief Number of dims in shape. */
  uint32_t        ndim;
} tensor_desc_t;

/* ------------------------------------------------------------------------- */
/* Tensor map container                                                       */
/* ------------------------------------------------------------------------- */

/**
 * @brief In-memory tensor map.
 *
 * @details
 * The loader owns all memory referenced by @ref tensor_desc_t entries.
 */
typedef struct tensor_map_s {
  /** @brief Tensor array. */
  tensor_desc_t *tensors;
  /** @brief Number of entries in @ref tensors. */
  uint32_t       count;
  /** @brief Whether loader succeeded. */
  int            loaded;
} tensor_map_t;

/* ------------------------------------------------------------------------- */
/* Debug formatting helpers                                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Format a shape into a compact string like "[a,b,c]".
 *
 * @param shape Shape pointer (may be NULL).
 * @param ndim Number of dims.
 * @param out Output buffer.
 * @param outsz Output buffer size.
 * @return Number of bytes written (excluding NUL) on success, or -1 on error.
 */
static inline int tensor_shape_to_string(const uint32_t *shape, uint32_t ndim,
                                        char *out, size_t outsz) {
  if (!out || outsz == 0u) return -1;
  out[0] = '\0';

  size_t used = 0;
  if (used + 1u >= outsz) return -1;
  out[used++] = '[';

  if (!shape || ndim == 0u) {
    if (used + 2u >= outsz) return -1;
    out[used++] = ']';
    out[used] = '\0';
    return (int)(used);
  }

  for (uint32_t i = 0; i < ndim; ++i) {
    char tmp[32];
    int n = 0;

    if (i == 0u) {
      n = snprintf(tmp, sizeof(tmp), "%u", (unsigned)shape[i]);
    } else {
      n = snprintf(tmp, sizeof(tmp), ",%u", (unsigned)shape[i]);
    }
    if (n < 0) return -1;

    const size_t nn = (size_t)n;
    if (used + nn + 2u >= outsz) return -1; /* + ']' + NUL */
    for (size_t k = 0; k < nn; ++k) out[used++] = tmp[k];
  }

  if (used + 2u >= outsz) return -1;
  out[used++] = ']';
  out[used] = '\0';
  return (int)used;
}

/**
 * @brief Format a tensor descriptor into a compact debug string.
 *
 * Format:
 *   "name=... off=... bytes=... dtype=... shape=[...]"
 *
 * @param td Tensor descriptor (may be NULL).
 * @param out Output buffer.
 * @param outsz Output buffer size.
 * @return Number of bytes written (excluding NUL) on success, or -1 on error.
 */
static inline int tensor_desc_to_string(const tensor_desc_t *td, char *out, size_t outsz) {
  if (!out || outsz == 0u) return -1;
  out[0] = '\0';
  if (!td) {
    int n0 = snprintf(out, outsz, "<null tensor_desc>");
    return (n0 < 0 || (size_t)n0 >= outsz) ? -1 : n0;
  }

  char shape_buf[128];
  if (tensor_shape_to_string(td->shape, td->ndim, shape_buf, sizeof(shape_buf)) < 0) {
    snprintf(shape_buf, sizeof(shape_buf), "[?]");
  }

  int n = snprintf(out, outsz,
                   "name=%s off=%llu bytes=%llu dtype=%s shape=%s",
                   (td->name ? td->name : "<null>"),
                   (unsigned long long)td->offset,
                   (unsigned long long)td->size_bytes,
                   tensor_dtype_name(td->dtype),
                   shape_buf);

  return (n < 0 || (size_t)n >= outsz) ? -1 : n;
}

/* ------------------------------------------------------------------------- */
/* API                                                                        */
/* ------------------------------------------------------------------------- */

/**
 * @brief Load tensor_map.json from disk.
 *
 * @param path Path to tensor_map.json.
 * @param out Output tensor map.
 * @return 0 on success, non-zero on failure.
 */
int tensor_map_load(const char *path, tensor_map_t *out);

/**
 * @brief Free tensor map and all owned memory.
 *
 * @param map Tensor map (may be NULL).
 */
void tensor_map_free(tensor_map_t *map);

/**
 * @brief Find a tensor descriptor by exact name.
 *
 * @param map Loaded tensor map.
 * @param name Tensor name.
 * @return Pointer to descriptor, or NULL if not found.
 */
const tensor_desc_t *tensor_map_find(const tensor_map_t *map, const char *name);

/**
 * @brief Validate a loaded tensor map against a mapped weights binary.
 *
 * @details
 * This is a hard consistency check intended to catch stale/mismatched tensor_map.json
 * versus model.ie.bin before any inference occurs. It validates:
 *  - name is present
 *  - size_bytes is non-zero
 *  - offset + size_bytes fits within bin_size
 *  - no overlaps (after sorting by offset)
 *
 * @param map Loaded tensor map.
 * @param bin_size Size in bytes of the mapped weights file.
 * @param err Optional error buffer for a human-readable error message.
 * @param errsz Size of err buffer.
 * @return 0 if valid, non-zero if invalid.
 */
int tensor_map_validate_against_bin(const tensor_map_t *map, size_t bin_size, char *err, size_t errsz);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_MAP_H_ */
