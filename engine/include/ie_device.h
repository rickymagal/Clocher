/* File: engine/include/ie_device.h */
/**
 * @file ie_device.h
 * @brief Device abstraction for CPU/GPU backends selected at runtime.
 */
#ifndef IE_DEVICE_H
#define IE_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Device kinds supported by the abstraction layer.
 */
typedef enum ie_device_kind {
  IE_DEV_CPU = 0,   /**< CPU backend (always available). */
  IE_DEV_CUDA = 1,  /**< NVIDIA CUDA backend (loaded via dlopen at runtime). */
  IE_DEV_ZE   = 2   /**< oneAPI Level Zero backend (loaded via dlopen). */
} ie_device_kind_t;

/**
 * @brief Opaque device handle (implementation-specific).
 */
typedef struct ie_device ie_device_t;

/**
 * @brief Forward declaration of block-sparse matrix descriptor.
 *
 * The full definition lives in `sparse_format.h`. This forward declaration
 * is sufficient for pointer-based APIs.
 */
typedef struct ie_block_sparse_matrix ie_block_sparse_matrix_t;

/**
 * @brief Capabilities reported by a device backend.
 */
typedef struct ie_device_caps {
  int has_gemv_f32;     /**< 1 if GEMV FP32 is implemented, else 0. */
  int has_mem_copy;     /**< 1 if host/device memcpy is implemented, else 0. */
  int has_streams;      /**< 1 if async streams/queues are supported, else 0. */
  char name[64];        /**< Human-readable device name. */
  char driver[64];      /**< Driver/runtime string for diagnostics. */
} ie_device_caps_t;

/**
 * @brief Create a device handle for the requested kind (with fallback).
 *
 * If the requested GPU backend is unavailable at runtime, a CPU device is
 * created instead so that inference can proceed.
 *
 * @param kind Requested device kind.
 * @param out_dev Output: device handle on success (non-NULL).
 * @return 0 on success; non-zero on error.
 */
int ie_device_create(ie_device_kind_t kind, ie_device_t **out_dev);

/**
 * @brief Destroy a device handle and release resources.
 *
 * @param dev Device handle previously created by ie_device_create().
 */
void ie_device_destroy(ie_device_t *dev);

/**
 * @brief Query device capabilities.
 *
 * @param dev Device handle.
 * @param out_caps Output capabilities structure (non-NULL).
 * @return 0 on success; non-zero on error.
 */
int ie_device_caps(const ie_device_t *dev, ie_device_caps_t *out_caps);

/**
 * @brief Return the logical kind of a device handle.
 *
 * @param dev Device handle.
 * @return Device kind (CPU, CUDA, ZE).
 */
ie_device_kind_t ie_device_kind(const ie_device_t *dev);

/**
 * @brief Device-side GEMV (FP32): y = W * x (+ optional bias).
 *
 * Dimensions: W is (rows x cols), x is (cols), y is (rows). Backends may copy
 * buffers or operate in-place if memory is resident on the device.
 *
 * @param dev Device handle.
 * @param W Row-major FP32 weights matrix of size rows*cols.
 * @param x FP32 input vector of length cols.
 * @param y FP32 output vector of length rows.
 * @param rows Number of rows in W (and y).
 * @param cols Number of columns in W (and x).
 * @param bias Optional FP32 bias vector of length rows (NULL if none).
 * @param blk_k Optional blocked-K hint (0 to ignore).
 * @return 0 on success; non-zero if unimplemented or error.
 */
int ie_device_gemv_f32(ie_device_t *dev,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k);

/**
 * @brief Device-side GEMV (Q4_0 weights, FP32 activations).
 *
 * The Q4_0 layout matches ie_gemv_q4_0_f32() (see ie_kernels.h).
 *
 * @param dev Device handle.
 * @param w_q4 Packed Q4_0 weights.
 * @param w_scales Per-block scales (BF16 or log2_u8_q3).
 * @param scale_bytes Bytes per scale (1 or 2).
 * @param x FP32 input vector.
 * @param y FP32 output vector.
 * @param rows Rows.
 * @param cols Cols (multiple of 32).
 * @param bias_bf16 Optional BF16 bias vector (rows) or NULL.
 * @return 0 on success; non-zero on error.
 */
int ie_device_gemv_q4_0_f32(ie_device_t *dev,
                            const uint8_t *w_q4,
                            const uint8_t *w_scales,
                            size_t scale_bytes,
                            const float *x,
                            float *y,
                            size_t rows,
                            size_t cols,
                            const uint16_t *bias_bf16);

/**
 * @brief Block-sparse GEMV (FP32): y = W * x (+ optional bias).
 *
 * This entry point operates on matrices stored in block-row CSR (BSR) layout
 * described by @ref ie_block_sparse_matrix_t. Backends are free to implement
 * native sparse kernels; if a backend does not implement this method, the
 * implementation falls back to a CPU block-sparse kernel when possible.
 *
 * @param dev Device handle.
 * @param m   Block-sparse matrix descriptor (must be fully initialized).
 * @param x   Input vector of length m->cols.
 * @param y   Output vector of length m->rows.
 * @param bias Optional bias vector of length m->rows (may be NULL).
 * @return 0 on success; non-zero if unimplemented or error.
 */
int ie_device_gemv_block_sparse_f32(ie_device_t *dev,
                                    const ie_block_sparse_matrix_t *m,
                                    const float *x,
                                    float *y,
                                    const float *bias);

/**
 * @brief Copy memory using the device backend.
 *
 * For CPU this is a plain memcpy; for GPU this would schedule host/device
 * copies when implemented. Direction is backend-defined for now.
 *
 * @param dev Device handle.
 * @param dst Destination pointer (host pointer in current implementation).
 * @param src Source pointer (host pointer in current implementation).
 * @param nbytes Number of bytes to copy.
 * @return 0 on success; non-zero on error.
 */
int ie_device_memcpy(ie_device_t *dev, void *dst, const void *src, size_t nbytes);

/**
 * @brief Parse a device kind from a string.
 *
 * Accepts "cpu", "cuda", "ze" (case-insensitive). Unknown strings return CPU.
 *
 * @param s Input string (may be NULL).
 * @return Parsed device kind (defaults to IE_DEV_CPU).
 */
ie_device_kind_t ie_device_kind_from_str(const char *s);

#ifdef __cplusplus
}
#endif
#endif /* IE_DEVICE_H */
