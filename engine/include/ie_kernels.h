/* File: engine/include/ie_kernels.h
 * -----------------------------------------------------------------------------
 * @file ie_kernels.h
 * @brief Kernel dispatch points (generic/AVX2) for GEMV and vector ops.
 *
 * @details
 * This header defines the public CPU-kernel API used by the engine.
 *
 * Responsibilities:
 * - Runtime selection of ISA-specific implementations (e.g., AVX2/FMA).
 * - GEMV entry points for fp32 weights and quantized-activation variants.
 * - Small vector math helpers used by fused epilogues (e.g., tanh).
 * - Reference CPU kernels for attention and MLP building blocks.
 *
 * Conventions:
 * - Callers own all buffers; kernels do not retain pointers after returning.
 * - After initialization, entry points are thread-safe assuming inputs do not alias.
 */

#ifndef IE_KERNELS_H_
#define IE_KERNELS_H_

#include <stddef.h>
#include <stdint.h>

#include "ie_quant_act.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Install the best available CPU kernels for this process.
 *
 * @details
 * Selects optimized implementations based on runtime CPU feature detection and
 * updates an internal dispatch table. Safe to call multiple times; last call wins.
 *
 * @param use_avx2 Non-zero to allow selecting AVX2/FMA-optimized paths.
 */
void ie_kernels_install(int use_avx2);

/**
 * @brief GEMV (fp32): y[r] = dot(W[r,:], x) + (bias ? bias[r] : 0).
 *
 * @details
 * Computes a matrix-vector product using the best installed implementation.
 *
 * Layout:
 * - Default is row-major: W is rows*cols floats, each row contiguous.
 * - Some kernels also accept a "blocked-K contiguous" interpretation per row;
 *   pass blk_k to describe the block size. Passing 0 disables blocking.
 *
 * @param W     Pointer to weights (row-major or compatible packed layout).
 * @param x     Input vector (length cols).
 * @param y     Output vector (length rows).
 * @param rows  Number of rows.
 * @param cols  Number of columns.
 * @param bias  Optional bias vector (length rows) or NULL.
 * @param blk_k Column block size for blocked-K layout; 0 for plain row-major.
 */
void ie_gemv_f32(const float *W, const float *x, float *y,
                 size_t rows, size_t cols,
                 const float *bias, size_t blk_k);

/**
 * @brief GEMV with per-tensor INT8 activations (fused dequantization).
 *
 * @details
 * Interprets activations with the affine model:
 *   real = scale * (q - zero_point)
 * Implementations may fuse dequantization into the dot product or dequantize
 * into a temporary float buffer depending on ISA.
 *
 * @param W         Weights (float).
 * @param x_q       INT8 activations (length cols).
 * @param y         Output (float, length rows).
 * @param rows      Number of rows.
 * @param cols      Number of columns.
 * @param bias      Optional bias (length rows) or NULL.
 * @param blk_k     Column block size; 0 disables blocking.
 * @param params    Per-tensor quantization parameters.
 * @param symmetric Informational flag (unused by some paths).
 */
void ie_gemv_qi8_f32(const float *W, const int8_t *x_q, float *y,
                     size_t rows, size_t cols,
                     const float *bias, size_t blk_k,
                     ie_act_i8_params params, int symmetric);

/**
 * @brief GEMV with per-group INT8 activations (blockwise parameters).
 *
 * @details
 * Each group of group_size elements uses its own (scale, zero_point):
 *   g = index / group_size
 *   real = scales[g] * (q - zeros[g])
 *
 * @param W           Weights (float).
 * @param x_q         INT8 activations (length cols).
 * @param y           Output (float, length rows).
 * @param rows        Rows.
 * @param cols        Cols.
 * @param bias        Optional bias or NULL.
 * @param blk_k       Column block size; 0 disables blocking.
 * @param group_size  Activation group size (>= 1).
 * @param scales      Scales array (ceil(cols/group_size)).
 * @param zeros       Zero-points array (ceil(cols/group_size)).
 * @param symmetric   Informational flag.
 */
void ie_gemv_qi8pg_f32(const float *W, const int8_t *x_q, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k,
                       size_t group_size, const float *scales,
                       const int8_t *zeros, int symmetric);

/**
 * @brief GEMV with FP8 activations (software decode, fused).
 *
 * @details
 * Decodes FP8 bytes (E4M3 or E5M2) on the fly while accumulating the dot product.
 *
 * @param W      Weights (float).
 * @param x_fp8  FP8 activations (bytes, length cols).
 * @param y      Output (float, length rows).
 * @param rows   Rows.
 * @param cols   Cols.
 * @param bias   Optional bias or NULL.
 * @param blk_k  Column block size; 0 disables blocking.
 * @param fmt    FP8 format selector.
 */
void ie_gemv_qfp8_f32(const float *W, const uint8_t *x_fp8, float *y,
                      size_t rows, size_t cols,
                      const float *bias, size_t blk_k,
                      ie_fp8_format fmt);

/**
 * @brief Causal self-attention (FP32 reference).
 *
 * @details
 * Computes per-head softmax(QK^T * inv_sqrt_d) V over seq_len tokens.
 *
 * Layout:
 * - Q and out are [heads, head_dim] flattened (head_dim fastest).
 * - K and V are [seq_len, heads, head_dim] flattened (head_dim fastest).
 *
 * scratch must have length >= seq_len (floats) and is reused per head.
 *
 * @return 0 on success, non-zero on invalid args.
 */
int ie_attn_cpu_causal_f32(const float *Q, const float *K, const float *V,
                           size_t seq_len, size_t heads, size_t head_dim,
                           float inv_sqrt_d, float *out, float *scratch);

/**
 * @brief SwiGLU MLP forward (FP32 reference).
 *
 * @details
 *   gate = W_gate * x + b_gate
 *   up   = W_up   * x + b_up
 *   act  = silu(gate) * up
 *   out  = W_down * act + b_down
 *
 * W_* are row-major matrices.
 *
 * tmp_gate and tmp_up must have length >= hidden_dim.
 *
 * @return 0 on success, non-zero on invalid args.
 */
int ie_mlp_cpu_swiglu_f32(const float *W_gate, const float *W_up, const float *W_down,
                          const float *x, size_t in_dim, size_t hidden_dim, size_t out_dim,
                          const float *b_gate, const float *b_up, const float *b_down,
                          float *out, float *tmp_gate, float *tmp_up);

/**
 * @brief Vector tanh on fp32 data (in-place).
 *
 * @param v         Input/output vector.
 * @param n         Number of elements.
 * @param fast_tanh Non-zero to use a fast approximation; zero to use tanhf().
 */
void ie_vec_tanh_f32(float *v, size_t n, int fast_tanh);

/**
 * @brief Fast scalar tanh approximation.
 *
 * @param x Input value.
 * @return Approximated tanh(x).
 */
float ie_fast_tanhf(float x);

#ifdef __cplusplus
}
#endif

#endif /* IE_KERNELS_H_ */
