/**
 * @file ie_tensor.h
 * @brief Minimal tensor/array utilities for the baseline.
 *
 * @defgroup IE_TENSOR Tensor Utilities
 * @brief Lightweight views used internally by the engine.
 * @{
 */
#ifndef IE_TENSOR_H
#define IE_TENSOR_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Contiguous 1D float vector view.
 *
 * The view does not own memory; the caller controls lifetime.
 */
typedef struct {
  float  *data;  /**< Pointer to the first element. */
  size_t  len;   /**< Number of float elements referenced by @ref data. */
} ie_f32_vec_t;

/**
 * @brief Construct a 1D float vector view.
 *
 * @param[in] p  Pointer to float data (may be NULL if @p n == 0).
 * @param[in] n  Number of elements.
 * @return A view referencing the provided memory.
 */
static inline ie_f32_vec_t ie_f32_vec(float *p, size_t n) {
  ie_f32_vec_t v; v.data = p; v.len = n; return v;
}

#ifdef __cplusplus
}
#endif
/** @} */ /* end of IE_TENSOR */
#endif /* IE_TENSOR_H */
