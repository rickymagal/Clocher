/* ============================================================================
 * File: engine/include/ie_rope.h
 * ============================================================================
 */
/**
 * @file ie_rope.h
 * @brief Rotary Position Embedding (RoPE) helpers for fp32 Q/K vectors.
 *
 * @details
 * RoPE applies a position-dependent complex rotation to pairs of channels in the
 * query/key vectors. For each pair (x0, x1) at index i, the rotation angle is:
 *
 *   angle(i, pos) = pos * theta^(-2i / head_dim)
 *
 * and the rotated pair is:
 *
 *   y0 = x0 * cos(angle) - x1 * sin(angle)
 *   y1 = x0 * sin(angle) + x1 * cos(angle)
 *
 * This module provides a correctness-first implementation for CPU inference.
 */

#ifndef IE_ROPE_H
#define IE_ROPE_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Apply RoPE rotation in-place to one Q vector and one K vector.
 *
 * @details
 * The function rotates the first `head_dim` elements of @p q and @p k in-place,
 * treating them as interleaved 2D components (0,1), (2,3), ...
 *
 * If @p head_dim is odd, the last element is left unchanged.
 *
 * @param q        Query vector (length >= @p head_dim), modified in-place.
 * @param k        Key vector (length >= @p head_dim), modified in-place.
 * @param head_dim Per-head hidden dimension.
 * @param pos      Position index (0-based).
 * @param theta    RoPE base theta (typically 10000.0).
 */
void ie_rope_apply_f32(float *q,
                       float *k,
                       size_t head_dim,
                       uint32_t pos,
                       float theta);

/**
 * @brief Apply RoPE rotation in-place to a single vector.
 *
 * @details
 * This is useful when Q and K are handled separately or when applying RoPE to
 * only one of them. Rotation uses the same definition as ::ie_rope_apply_f32.
 *
 * If @p head_dim is odd, the last element is left unchanged.
 *
 * @param x        Vector to rotate (length >= @p head_dim), modified in-place.
 * @param head_dim Per-head hidden dimension.
 * @param pos      Position index (0-based).
 * @param theta    RoPE base theta (typically 10000.0).
 */
void ie_rope_apply_one_f32(float *x,
                           size_t head_dim,
                           uint32_t pos,
                           float theta);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_ROPE_H */
