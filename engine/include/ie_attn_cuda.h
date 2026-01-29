/* File: engine/include/ie_attn_cuda.h */
/**
 * @file ie_attn_cuda.h
 * @brief C ABI for CUDA attention kernels.
 */
#ifndef IE_ATTN_CUDA_H
#define IE_ATTN_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CUDA causal attention for the standard (non-GQA) case.
 *
 * All pointers refer to device memory.
 */
int ie_attn_cuda_causal_f32(const float *Q,
                            const float *K,
                            const float *V,
                            size_t seq_len,
                            size_t heads,
                            size_t head_dim,
                            float inv_sqrt_d,
                            float *out);

/**
 * @brief CUDA causal attention for GQA (n_heads != n_kv_heads).
 *
 * All pointers refer to device memory. K/V are laid out as
 * [seq_len, n_kv_heads, head_dim].
 */
int ie_attn_cuda_causal_gqa_f32(const float *Q,
                                const float *K,
                                const float *V,
                                size_t seq_len,
                                size_t n_heads,
                                size_t n_kv_heads,
                                size_t head_dim,
                                float inv_sqrt_d,
                                float *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_ATTN_CUDA_H */
