/**
 * @file ie_tensor.h
 * @brief Minimal tensor utilities for the baseline.
 */
#ifndef IE_TENSOR_H
#define IE_TENSOR_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Contiguous 1D tensor view. */
typedef struct {
  float   *data;     /**< Pointer to data (owned by caller unless noted). */
  size_t   len;      /**< Number of float elements. */
} ie_f32_vec_t;

/** @brief Initialize a vector view. */
static inline ie_f32_vec_t ie_f32_vec(float *p, size_t n) {
  ie_f32_vec_t v; v.data = p; v.len = n; return v;
}

#ifdef __cplusplus
}
#endif
#endif /* IE_TENSOR_H */
