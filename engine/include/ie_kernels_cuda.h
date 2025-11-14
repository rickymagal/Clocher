/**
 * @file ie_kernels_cuda.h
 * @brief CUDA GPU kernels and C-ABI launchers for hot-path vector/matrix ops.
 *
 * This header exposes a minimal C ABI to launch CUDA kernels from the engine
 * without leaking CUDA types into other translation units.
 *
 * ### Design goals
 * - Stable C ABI: callable from C or C++.
 * - CUDA isolation: only this header/TU knows about CUDA; the rest of the
 *   project remains CUDA-agnostic.
 * - No global state: all configuration is passed via arguments.
 *
 * ### Error model
 * All launchers return `0` on success and a negative value on failure.
 * Call ::ie_cuda_last_error_string() to retrieve a printable diagnostic.
 *
 * ### Pointer / memory rules
 * Unless explicitly stated otherwise, all pointers refer to device memory.
 * Passing host pointers is undefined behavior and will surface as CUDA errors.
 *
 * @ingroup IE_GPU
 */

#ifndef IE_KERNELS_CUDA_H
#define IE_KERNELS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "ie_quant_act.h"  /* for ie_fp8_format */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup IE_GPU CUDA kernels & launchers
 * @brief Public API to invoke CUDA kernels for GEMV, activations and packing.
 * @{
 */

/**
 * @brief Opaque CUDA stream handle used by this C API.
 *
 * In non-CUDA translation units this is a `void*`. In the CUDA TU it is
 * replaced by `cudaStream_t`. You may pass `NULL` to use the default stream.
 */
#ifndef IE_CUDA_STREAM_T_DEFINED
#define IE_CUDA_STREAM_T_DEFINED
typedef void* ie_cuda_stream_t;
#endif

/**
 * @brief Activation types supported by fused kernels.
 */
typedef enum ie_act_kind_e {
  IE_ACT_NONE = 0,   /**< Identity.                          */
  IE_ACT_RELU = 1,   /**< ReLU: max(0, x).                   */
  IE_ACT_TANH = 2    /**< Hyperbolic tangent.                */
} ie_act_kind_t;

/* -------------------------------------------------------------------------- */
/* Error handling                                                             */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return a thread-local message describing the last CUDA error.
 *
 * The pointer remains valid until the next launcher call on the same thread.
 * An empty string means "no error".
 *
 * @return NUL-terminated error message (never `NULL`).
 */
const char* ie_cuda_last_error_string(void);

/* -------------------------------------------------------------------------- */
/* GEMV FP32 (row-wise)                                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compute `y = alpha * W * x + beta * y` (row-wise GEMV).
 *
 * - `W` is dense row-major with `rows` rows and `ldW` columns stride.
 * - Each row is reduced in parallel; one block per output row.
 *
 * @param W       Device pointer to row-major matrix `(rows x ldW)`.
 * @param x       Device pointer to input vector (length `cols`).
 * @param y       Device pointer to output vector (length `rows`).
 * @param rows    Number of rows in `W` and elements in `y` (>= 1).
 * @param cols    Number of columns in `W` and elements in `x` (>= 1).
 * @param ldW     Leading dimension of `W` in elements (>= `cols`).
 * @param alpha   Scalar multiplier for `W*x`.
 * @param beta    Scalar multiplier for existing `y` (use `0.f` to overwrite).
 * @param stream  CUDA stream handle (may be `NULL`).
 * @return `0` on success, negative on error (see ::ie_cuda_last_error_string()).
 *
 * @pre Pointers are non-NULL and reference device memory.
 * @pre `rows > 0`, `cols > 0`, `ldW >= cols`.
 * @post `y` contains the result.
 */
int ie_cuda_launch_gemv_rowwise_f32(const float* W,
                                    const float* x,
                                    float*       y,
                                    int          rows,
                                    int          cols,
                                    int          ldW,
                                    float        alpha,
                                    float        beta,
                                    ie_cuda_stream_t stream);

/* -------------------------------------------------------------------------- */
/* Fused GEMV + bias + activation (FP32)                                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compute `y = act(alpha * W*x + bias + beta*y)` in one pass.
 *
 * - Applies optional per-row `bias` (may be NULL → treated as zeros).
 * - Applies activation specified by ::ie_act_kind_t.
 *
 * @param W       Device pointer to row-major matrix `(rows x ldW)`.
 * @param x       Device pointer to input vector (length `cols`).
 * @param bias    Device pointer to per-row bias (length `rows`) or `NULL`.
 * @param y       Device pointer to output vector (length `rows`).
 * @param rows    Number of rows in `W`/`y`/`bias` (>= 1).
 * @param cols    Number of columns in `W` and elements in `x` (>= 1).
 * @param ldW     Leading dimension of `W` in elements (>= `cols`).
 * @param alpha   Scalar multiplier for `W*x`.
 * @param beta    Scalar multiplier for existing `y` (use `0.f` to overwrite).
 * @param act     Activation kind (see ::ie_act_kind_t).
 * @param stream  CUDA stream handle (may be `NULL`).
 * @return `0` on success, negative on error.
 *
 * @pre Pointers are non-NULL device pointers (except `bias`, which may be NULL).
 * @pre `rows > 0`, `cols > 0`, `ldW >= cols`.
 * @post `y` contains the result with activation applied.
 */
int ie_cuda_launch_gemv_bias_act_f32(const float* W,
                                     const float* x,
                                     const float* bias,
                                     float*       y,
                                     int          rows,
                                     int          cols,
                                     int          ldW,
                                     float        alpha,
                                     float        beta,
                                     ie_act_kind_t act,
                                     ie_cuda_stream_t stream);

/* -------------------------------------------------------------------------- */
/* GEMV with INT8 activations (per-tensor)                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Row-wise GEMV with INT8 activations (per-tensor), fused dequantization.
 *
 * Dequantization model: `real = scale * (q - zero_point)`.
 *
 * @param W       Float weights on device (row-major), size rows x ldW.
 * @param xq      INT8 activations on device, length cols.
 * @param y       Output on device, length rows.
 * @param rows    Rows.
 * @param cols    Cols.
 * @param ldW     Leading dimension (>= cols).
 * @param scale   Per-tensor scale.
 * @param zp      Per-tensor zero-point.
 * @param alpha   Scale for W*x.
 * @param beta    Scale for existing y.
 * @param stream  CUDA stream (may be NULL).
 * @return `0` on success, negative on error.
 */
int ie_cuda_launch_gemv_rowwise_qi8_f32(const float*   W,
                                        const int8_t*  xq,
                                        float*         y,
                                        int            rows,
                                        int            cols,
                                        int            ldW,
                                        float          scale,
                                        int            zp,
                                        float          alpha,
                                        float          beta,
                                        ie_cuda_stream_t stream);

/* -------------------------------------------------------------------------- */
/* GEMV with FP8 activations (E4M3/E5M2)                                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Row-wise GEMV with FP8 activations (E4M3/E5M2), fused byte decode.
 *
 * @param W       Float weights on device (row-major), size rows x ldW.
 * @param x8      FP8 activations on device (bytes), length cols.
 * @param y       Output on device, length rows.
 * @param rows    Rows.
 * @param cols    Cols.
 * @param ldW     Leading dimension (>= cols).
 * @param fmt     FP8 format (::IE_FP8_E4M3 or ::IE_FP8_E5M2).
 * @param alpha   Scale for W*x.
 * @param beta    Scale for existing y.
 * @param stream  CUDA stream (may be NULL).
 * @return `0` on success, negative on error.
 */
int ie_cuda_launch_gemv_rowwise_qfp8_f32(const float*   W,
                                         const uint8_t* x8,
                                         float*         y,
                                         int            rows,
                                         int            cols,
                                         int            ldW,
                                         ie_fp8_format  fmt,
                                         float          alpha,
                                         float          beta,
                                         ie_cuda_stream_t stream);

/* -------------------------------------------------------------------------- */
/* Vector activations (FP32)                                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compute `y[i] = tanh(x[i])` for `i in [0, n)`.
 *
 * @param y       Device pointer to output vector (length `n`).
 * @param x       Device pointer to input vector  (length `n`).
 * @param n       Number of elements (>= 1).
 * @param stream  CUDA stream handle (may be `NULL`).
 * @return `0` on success, negative on error.
 *
 * @pre `x` and `y` are valid device pointers with at least `n` elements.
 * @post `y` contains elementwise `tanh(x)`.
 */
int ie_cuda_launch_vec_tanh_f32(float*       y,
                                const float* x,
                                int          n,
                                ie_cuda_stream_t stream);

/* -------------------------------------------------------------------------- */
/* Weight packing (Blocked-K)                                                 */
/* -------------------------------------------------------------------------- */

/**
 * @brief Pack a row-major matrix into Blocked-K layout for better memory access.
 *
 * The K dimension (columns) is partitioned into tiles of size `block_k`.
 * Data are stored as contiguous tiles:
 * @code
 *   int kb = k / block_k;  // tile index along K
 *   int ko = k % block_k;  // in-tile offset
 *   size_t dst = ((size_t)kb * rows + r) * (size_t)block_k + (size_t)ko;
 *   Wp[dst] = W[(size_t)r * ldW + (size_t)k];
 * @endcode
 *
 * @param Wp       Device pointer to destination buffer (size = rows*cols).
 * @param W        Device pointer to source row-major matrix.
 * @param rows     Number of rows (>= 1).
 * @param cols     Number of columns (>= 1).
 * @param ldW      Leading dimension of `W` in elements (>= `cols`).
 * @param block_k  Tile size along K (e.g., 32/64/128; must be > 0).
 * @param stream   CUDA stream handle (may be `NULL`).
 * @return `0` on success, negative on error.
 *
 * @pre `Wp` and `W` are device pointers with sufficient capacity.
 * @post `Wp` contains the packed matrix in Blocked-K layout.
 */
int ie_cuda_launch_pack_w_blockedk_f32(float*       Wp,
                                       const float* W,
                                       int          rows,
                                       int          cols,
                                       int          ldW,
                                       int          block_k,
                                       ie_cuda_stream_t stream);

/** @} */ /* end of group IE_GPU */

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* IE_KERNELS_CUDA_H */
