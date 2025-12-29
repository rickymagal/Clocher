/* ============================================================================
 * File: engine/src/kernels/mlp_cpu.c
 * ============================================================================
 */
/**
 * @file mlp_cpu.c
 * @brief Reference CPU MLP kernels (FP32).
 *
 * Implements a common "SwiGLU" MLP:
 *   gate = W_gate * x + b_gate
 *   up   = W_up   * x + b_up
 *   act  = silu(gate) * up
 *   out  = W_down * act + b_down
 *
 * All GEMVs use ie_gemv_f32() so the installed kernel backend is honored.
 * This implementation is allocation-free: caller provides temporaries.
 */

#include "ie_kernels.h"

#include <math.h>
#include <stddef.h>

static inline float silu_f32(float x) {
  return x / (1.0f + expf(-x));
}

/**
 * @brief SwiGLU MLP forward (FP32 weights/activations).
 *
 * Shapes:
 *   x:        [in_dim]
 *   W_gate:   [hidden_dim, in_dim]
 *   W_up:     [hidden_dim, in_dim]
 *   W_down:   [out_dim, hidden_dim]
 *   tmp_gate: [hidden_dim]
 *   tmp_up:   [hidden_dim]
 *   out:      [out_dim]
 *
 * Bias pointers may be NULL.
 *
 * @return 0 on success, non-zero on invalid args.
 */
int ie_mlp_cpu_swiglu_f32(const float *W_gate, const float *W_up, const float *W_down,
                          const float *x, size_t in_dim, size_t hidden_dim, size_t out_dim,
                          const float *b_gate, const float *b_up, const float *b_down,
                          float *out, float *tmp_gate, float *tmp_up) {
  if (!W_gate || !W_up || !W_down || !x || !out || !tmp_gate || !tmp_up) return -1;
  if (in_dim == 0 || hidden_dim == 0 || out_dim == 0) return -1;

  ie_gemv_f32(W_gate, x, tmp_gate, hidden_dim, in_dim, b_gate, 0);
  ie_gemv_f32(W_up,   x, tmp_up,   hidden_dim, in_dim, b_up,   0);

  for (size_t i = 0; i < hidden_dim; ++i) {
    tmp_gate[i] = silu_f32(tmp_gate[i]) * tmp_up[i];
  }

  ie_gemv_f32(W_down, tmp_gate, out, out_dim, hidden_dim, b_down, 0);
  return 0;
}
