#ifndef TENSOR_MAP_H_
#define TENSOR_MAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * @file tensor_map.h
 * @brief Loader and lookup API for tensor_map.json.
 *
 * tensor_map.json describes how logical tensor names map to byte offsets,
 * shapes, and dtypes inside model.ie.bin.
 *
 * Supported input shapes:
 *  - Array style: { "tensors": [ { "name": "...", "offset": ..., ... }, ... ] }
 *  - Direct array: [ { "name": "...", ... }, ... ]
 *  - Map style: { "tensor.name": { "offset": ..., ... }, ... }
 */

/* ------------------------------------------------------------------------- */
/* Tensor descriptor                                                         */
/* ------------------------------------------------------------------------- */

typedef enum tensor_dtype_e {
  TENSOR_DTYPE_F32 = 0,
  TENSOR_DTYPE_F16 = 1,
  TENSOR_DTYPE_INT4 = 2,
  TENSOR_DTYPE_BF16 = 3,
  TENSOR_DTYPE_U8 = 4,
  TENSOR_DTYPE_UNKNOWN = 255
} tensor_dtype_t;

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

typedef struct tensor_desc_s {
  char           *name;        /* tensor name (owned, heap) */
  uint64_t        offset;      /* byte offset in model.ie.bin */
  uint64_t        size_bytes;  /* size in bytes */
  tensor_dtype_t  dtype;

  uint32_t       *shape;       /* array of dims (owned, heap) */
  uint32_t        ndim;
} tensor_desc_t;

/* ------------------------------------------------------------------------- */
/* Tensor map container                                                      */
/* ------------------------------------------------------------------------- */

typedef struct tensor_map_s {
  tensor_desc_t *tensors;
  uint32_t       count;
  int            loaded;
} tensor_map_t;

/* ------------------------------------------------------------------------- */
/* API                                                                       */
/* ------------------------------------------------------------------------- */

/**
 * @brief Load tensor_map.json from disk.
 *
 * @param path Path to tensor_map.json
 * @param out  Output tensor map
 * @return 0 on success, non-zero on failure
 */
int tensor_map_load(const char *path, tensor_map_t *out);

/**
 * @brief Free tensor map and all owned memory.
 *
 * @param map Tensor map
 */
void tensor_map_free(tensor_map_t *map);

/**
 * @brief Find a tensor descriptor by exact name.
 *
 * @param map Loaded tensor map
 * @param name Tensor name
 * @return Pointer to descriptor, or NULL if not found
 */
const tensor_desc_t *tensor_map_find(const tensor_map_t *map, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_MAP_H_ */
