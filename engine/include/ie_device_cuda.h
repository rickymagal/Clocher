/**
 * @file ie_device_cuda.h
 * @brief C ABI wrapper for CUDA runtime functionality used by the engine.
 *
 * This header intentionally exposes a small, C-callable surface so that C
 * translation units (compiled with a C compiler) can drive CUDA work without
 * including CUDA headers.
 *
 * The implementation is provided by a .cu translation unit compiled by NVCC.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copy direction selector for ie_cuda_memcpy().
 *
 * IE_CUDA_COPY_DEFAULT uses cudaMemcpyDefault, which relies on unified virtual
 * addressing when available. If you know the direction, prefer the explicit
 * modes.
 */
typedef enum ie_cuda_copy_kind {
  IE_CUDA_COPY_DEFAULT = 0, /**< Use cudaMemcpyDefault. */
  IE_CUDA_COPY_H2D     = 1, /**< Host to device. */
  IE_CUDA_COPY_D2H     = 2, /**< Device to host. */
  IE_CUDA_COPY_D2D     = 3  /**< Device to device. */
} ie_cuda_copy_kind_t;

/**
 * @brief Returns non-zero if at least one CUDA device is available.
 *
 * This call will initialize the CUDA runtime enough to query device count.
 *
 * @return 1 if devices exist, 0 otherwise.
 */
int ie_cuda_is_available(void);

/**
 * @brief Initialize CUDA for a given device ordinal and return basic identity strings.
 *
 * This performs cudaSetDevice() and a minimal runtime initialization (cudaFree(0))
 * so that driver activity is observable (e.g., /dev/nvidiactl gets touched).
 *
 * @param device_ordinal CUDA device ordinal (0..N-1).
 * @param out_name Output buffer for the GPU name (may be NULL if name_cap=0).
 * @param name_cap Capacity of out_name in bytes.
 * @param out_driver Output buffer for a driver/runtime version string (may be NULL if driver_cap=0).
 * @param driver_cap Capacity of out_driver in bytes.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_init(int device_ordinal,
                 char *out_name, size_t name_cap,
                 char *out_driver, size_t driver_cap);

/**
 * @brief Allocate device memory.
 *
 * @param nbytes Number of bytes.
 * @return Device pointer on success, NULL on failure.
 */
void *ie_cuda_malloc(size_t nbytes);

/**
 * @brief Free device memory allocated by ie_cuda_malloc().
 *
 * @param p Device pointer (may be NULL).
 */
void ie_cuda_free(void *p);

/**
 * @brief Copy memory using CUDA.
 *
 * For IE_CUDA_COPY_DEFAULT, cudaMemcpyDefault is used.
 *
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nbytes Number of bytes.
 * @param kind Copy kind.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_memcpy(void *dst, const void *src, size_t nbytes, ie_cuda_copy_kind_t kind);

/**
 * @brief Launch a GEMV FP32 kernel: y = W*x + bias (bias optional).
 *
 * All pointers must be device pointers. bias may be NULL.
 *
 * @param dW Device pointer to row-major matrix W (rows x cols).
 * @param dx Device pointer to vector x (cols).
 * @param dy Device pointer to vector y (rows).
 * @param rows Number of rows in W / entries in y.
 * @param cols Number of cols in W / entries in x.
 * @param dbias Optional device pointer to bias (rows) or NULL.
 * @return 0 on success, negative on failure.
 */
int ie_cuda_gemv_f32(const float *dW,
                     const float *dx,
                     float *dy,
                     size_t rows,
                     size_t cols,
                     const float *dbias);

/**
 * @brief Return a human-readable description of the last CUDA error seen by this wrapper.
 *
 * @return Pointer to an internal static string (valid until next CUDA wrapper call).
 */
const char *ie_cuda_last_error_string(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
