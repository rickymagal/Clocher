/* ============================================================================
 * File: engine/include/ie_rope.h
 * ============================================================================
 */
/**
 * @file ie_rope.h
 * @brief Rotary Position Embedding (RoPE) interfaces.
 *
 * @details
 * RoPE rotates interleaved pairs (2*i, 2*i+1) in each head.
 *
 * This header matches the runtime usage in infer_gptoss.c where calls may pass
 * only q or only k (the other pointer may be NULL), and the function returns
 * an int status code.
 */

#ifndef IE_ROPE_H
#define IE_ROPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply RoPE to a single head vector in-place.
 *
 * @param x        Pointer to head vector (length head_dim).
 * @param head_dim Per-head dimension (must be even and >= 2).
 * @param pos      Position index.
 * @param theta    RoPE base theta (must be > 0).
 * @return 0 on success, negative on error.
 */
int ie_rope_apply_one_f32(float *x, size_t head_dim, uint32_t pos, float theta);

/**
 * @brief Apply RoPE to Q and/or K for multiple heads.
 *
 * @details
 * Either q or k may be NULL. If both are NULL, this is a no-op returning 0.
 *
 * @param q        Q buffer, layout [heads, head_dim], may be NULL.
 * @param k        K buffer, layout [heads, head_dim], may be NULL.
 * @param heads    Number of heads in q/k (must be > 0 if q or k is non-NULL).
 * @param head_dim Per-head dimension (must be even and >= 2).
 * @param pos      Position index.
 * @param theta    RoPE base theta (must be > 0).
 * @return 0 on success, negative on error.
 */
int ie_rope_apply_f32(float *q, float *k, size_t heads, size_t head_dim, uint32_t pos,
                      float theta);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_ROPE_H */
