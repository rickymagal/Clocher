/**
 * @file ie_device_ze.h
 * @brief oneAPI Level Zero backend entry points (skeleton).
 *
 * This header defines a minimal C ABI for a Level Zero backend so the
 * engine can be wired once and the implementation can evolve without
 * touching call sites. The current implementation is a clean “unavailable”
 * skeleton that compiles and links; it returns IE_ZE_UNAVAILABLE until
 * we add a real kernel + runtime plumbing.
 *
 * Build toggle: define IE_WITH_ZE=1 to compile/link Level Zero files.
 */

#ifndef IE_DEVICE_ZE_H_
#define IE_DEVICE_ZE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Return codes for Level Zero backend. */
enum {
  IE_ZE_OK            =  0,  /**< Success. */
  IE_ZE_ERR_RUNTIME   = -1,  /**< Level Zero runtime error. */
  IE_ZE_UNAVAILABLE   = -2,  /**< Level Zero not available / not compiled. */
  IE_ZE_EINVAL        = -3   /**< Invalid arguments. */
};

/**
 * @brief Check if Level Zero is available at runtime.
 *
 * When compiled without IE_WITH_ZE, this returns IE_ZE_UNAVAILABLE.
 *
 * @return IE_ZE_OK if the loader/driver is usable; IE_ZE_UNAVAILABLE
 *         otherwise; IE_ZE_ERR_RUNTIME on runtime init failure.
 */
int ie_ze_is_available(void);

/**
 * @brief Run y = W * x using a Level Zero kernel (FP32).
 *
 * This is a placeholder signature that mirrors the CUDA one.
 * The current implementation returns IE_ZE_UNAVAILABLE until
 * the kernel path is implemented.
 *
 * @param W    Host pointer to row-major matrix W(rows, cols).
 * @param x    Host pointer to vector x(cols).
 * @param y    Host pointer to vector y(rows).
 * @param rows Rows of W / length of y.
 * @param cols Cols of W / length of x.
 * @param ldw  Leading dimension of W in elements.
 * @return IE_ZE_OK on success or an error code.
 */
int ie_ze_gemv_f32(const float *W,
                   const float *x,
                   float *y,
                   int rows,
                   int cols,
                   int ldw);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEVICE_ZE_H_ */
