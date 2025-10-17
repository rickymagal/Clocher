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

/** @brief Device kind enumeration. */
typedef enum ie_device_kind {
  IE_DEV_CPU = 0,   /**< CPU backend (always available). */
  IE_DEV_CUDA = 1,  /**< NVIDIA CUDA backend (loaded via dlopen). */
  IE_DEV_ZE   = 2   /**< oneAPI Level Zero backend (dlopen). */
} ie_device_kind_t;

/** @brief Opaque device handle. */
typedef struct ie_device ie_device_t;

/**
 * @brief Device capabilities advertised by a backend.
 */
typedef struct ie_device_caps {
  int has_gemv_f32;     /**< 1 if GEMV FP32 is implemented, 0 otherwise. */
  int has_mem_copy;     /**< 1 if host<->device copies are implemented. */
  int has_streams;      /**< 1 if async streams/queues are supported. */
  char name[64];        /**< Human-readable device name. */
  char driver[64];      /**< Driver/runtime info string. */
} ie_device_caps_t;

/**
 * @brief Create a device handle for the requested kind.
 *
 * The function attempts to load dynamic libraries as needed and returns a
 * valid CPU device if the requested GPU backend is unavailable.
 *
 * @param kind        Requested device kind.
 * @param out_dev     Output: device handle on success. Must be non-NULL.
 * @return 0 on success; non-zero on error.
 */
int ie_device_create(ie_device_kind_t kind, ie_device_t **out_dev);

/**
 * @brief Destroy a device handle and release resources.
 *
 * @param dev         Device handle previously created by ie_device_create().
 */
void ie_device_destroy(ie_device_t *dev);

/**
 * @brief Query device capabilities.
 *
 * @param dev         Device handle.
 * @param out_caps    Output structure to receive capabilities (non-NULL).
 * @return 0 on success; non-zero on error.
 */
int ie_device_caps(const ie_device_t *dev, ie_device_caps_t *out_caps);

/**
 * @brief Device-side GEMV (FP32): y = W * x (+ optional bias).
 *
 * Dimensions: W is (rows x cols), x is (cols), y is (rows).
 * Backends may copy buffers or operate in-place if memory is resident.
 *
 * @param dev         Device handle.
 * @param W           Pointer to weights matrix in row-major FP32.
 * @param x           Pointer to input vector FP32 (length: cols).
 * @param y           Pointer to output vector FP32 (length: rows).
 * @param rows        Number of rows in W and length of y.
 * @param cols        Number of cols in W and length of x.
 * @param bias        Optional bias vector FP32 of length rows (NULL if none).
 * @param blk_k       Optional blocked-K hint (0 to ignore).
 * @return 0 on success; non-zero on error (or unimplemented).
 */
int ie_device_gemv_f32(ie_device_t *dev,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k);

/**
 * @brief Copy memory to or from the device (backend-dependent).
 *
 * Direction is inferred by implementation; for CPU it is a simple memcpy.
 *
 * @param dev         Device handle.
 * @param dst         Destination pointer.
 * @param src         Source pointer.
 * @param nbytes      Number of bytes to copy.
 * @return 0 on success; non-zero on error.
 */
int ie_device_memcpy(ie_device_t *dev, void *dst, const void *src, size_t nbytes);

/**
 * @brief Parse device kind from string.
 *
 * Accepts "cpu", "cuda", "ze" (case-insensitive). Defaults to IE_DEV_CPU.
 *
 * @param s           Input string (may be NULL).
 * @return Parsed device kind.
 */
ie_device_kind_t ie_device_kind_from_str(const char *s);

#ifdef __cplusplus
}
#endif
#endif /* IE_DEVICE_H */
