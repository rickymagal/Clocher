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
