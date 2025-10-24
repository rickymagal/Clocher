/**
 * @file ie_device_cuda.h
 * @brief CUDA backend entry points (first-party, no cuBLAS).
 *
 * This header exposes a minimal C ABI to offload FP32 GEMV to CUDA.
 * It is backend-agnostic for the engine core: you can call these
 * utilities opportunistically when CUDA is present, and fall back
 * to the CPU path otherwise. No global state is leaked.
 *
 * ## Design goals
 * - Keep the ABI in plain C for easy inclusion in C11 code.
 * - Avoid third-party libraries (no cuBLAS); ship a simple kernel.
 * - Permit "one-shot" usage (alloc/copy/launch/free) for correctness
 *   first; later we can add persistent buffers for performance.
 *
 * ## Thread safety
 * All functions are thread-safe as long as you do not share the same
 * device pointers across threads. These entry points do not expose
 * device pointers, so they are safe by construction.
 *
 * ## Return codes
 * -  0: success
 * - -1: runtime error (CUDA API error)
 * - -2: unavailable (no CUDA device detected / not compiled with CUDA)
 * - -3: invalid arguments
 *
 * @note Build is controlled by a compile-time macro:
 *       Define IE_WITH_CUDA=1 to compile/link the CUDA backend.
 */

#ifndef IE_DEVICE_CUDA_H_
#define IE_DEVICE_CUDA_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Return codes for CUDA backend. */
enum {
  IE_CUDA_OK            =  0,  /**< Success. */
  IE_CUDA_ERR_RUNTIME   = -1,  /**< CUDA runtime API error. */
  IE_CUDA_UNAVAILABLE   = -2,  /**< CUDA not available or not compiled in. */
  IE_CUDA_EINVAL        = -3   /**< Invalid arguments. */
};

/**
 * @brief Check if a CUDA device is available at runtime.
 *
 * This attempts to initialize the CUDA runtime and query the
 * number of devices. When the code is compiled without CUDA
 * support (IE_WITH_CUDA!=1), this returns IE_CUDA_UNAVAILABLE.
 *
 * @return IE_CUDA_OK if at least one device is present;
 *         IE_CUDA_UNAVAILABLE otherwise;
 *         IE_CUDA_ERR_RUNTIME on CUDA API failure.
 */
int ie_cuda_is_available(void);

/**
 * @brief Run y = W * x using a native CUDA kernel (FP32).
 *
 * Layout:
 * - W is row-major with leading dimension @p ldw (ldw >= cols).
 * - x is a length-@p cols vector.
 * - y is a length-@p rows vector (output).
 *
 * Semantics:
 * - Allocates device buffers, copies W and x H2D, launches kernel,
 *   copies y D2H, and frees device buffers. This is correctness-first.
 *   We can add persistent allocations in a later iteration.
 *
 * Preconditions:
 * - All pointers must be non-NULL.
 * - rows > 0, cols > 0, ldw >= cols.
 *
 * @param W    Host pointer to row-major matrix of shape (rows, cols).
 * @param x    Host pointer to vector of shape (cols).
 * @param y    Host pointer to output vector of shape (rows).
 * @param rows Number of rows in W / y length.
 * @param cols Number of cols in W / x length.
 * @param ldw  Leading dimension of W in elements (>= cols).
 * @return IE_CUDA_OK on success; see return codes above otherwise.
 */
int ie_cuda_gemv_f32(const float *W,
                     const float *x,
                     float *y,
                     int rows,
                     int cols,
                     int ldw);

/**
 * @brief Convenience that also accepts strides in bytes.
 *
 * Same as @ref ie_cuda_gemv_f32 but allows a byte-stride for W.
 * Useful if the host matrix is subviewed or padded in bytes.
 *
 * @param W        Host pointer to W (row-major).
 * @param x        Host pointer to x.
 * @param y        Host pointer to y.
 * @param rows     Rows.
 * @param cols     Cols.
 * @param ldw_e    Leading dimension in elements (>= cols).
 * @param row_stride_bytes  Distance in bytes from W[r,0] to W[r+1,0].
 *                          If 0, computed as ldw_e*sizeof(float).
 * @return IE_CUDA_OK on success or an error code.
 */
int ie_cuda_gemv_f32_strided(const float *W,
                             const float *x,
                             float *y,
                             int rows,
                             int cols,
                             int ldw_e,
                             size_t row_stride_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEVICE_CUDA_H_ */
