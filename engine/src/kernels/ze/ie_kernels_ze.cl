/*!
 * @file ie_kernels_ze.cl
 * @brief OpenCL C kernels for the Level Zero backend (compiled to SPIR-V).
 *
 * Build to SPIR-V:
 *
 *   Using Intel ocloc (recommended):
 *     ocloc compile \
 *       -file ie_kernels_ze.cl -spirv \
 *       -out_dir build/engine/src/kernels/ze -output ie_kernels_ze \
 *       -options "-cl-std=CL2.0 -cl-kernel-arg-info"
 *
 *   Using Clang + llvm-spirv:
 *     clang -target spir64 -x cl -cl-std=CL2.0 -O2 -finclude-default-header \
 *           -emit-llvm -c ie_kernels_ze.cl -o ie_kernels_ze.bc
 *     llvm-spirv ie_kernels_ze.bc -o ie_kernels_ze.spv
 */

#define IE_WG_TILE 256  /* Required local size X for reductions */

/* ---------------------------- helpers ----------------------------------- */

/**
 * @brief Apply an activation function that mirrors the host enum layout.
 *
 * The mapping is:
 *   0 -> NONE   : f(x) = x
 *   1 -> RELU   : f(x) = max(x, 0)
 *   2 -> TANH   : f(x) = tanh(x)
 *
 * @param x   Input value (float).
 * @param act Activation kind (0 none, 1 relu, 2 tanh).
 * @return Activated value.
 */
static inline float ie_apply_activation(float x, int act) {
  if (act == 1) {            /* RELU */
    return x > 0.0f ? x : 0.0f;
  } else if (act == 2) {     /* TANH */
    return tanh(x);
  }
  return x;                  /* NONE */
}

/* ------------------------------ GEMV ------------------------------------ */

/**
 * @brief Row-wise GEMV (FP32): y = alpha * W * x + beta * y.
 *
 * Each work-item computes a partial dot for its row and participates in a
 * reduction within the work-group. The local X dimension MUST be 256 so that
 * reduction uses a fixed-size local array without overflow.
 *
 * Preconditions:
 *  - Local work-group size X is exactly 256.
 *  - Global X dimension covers `rows` (excess work-items exit early).
 *
 * @param W     __global const float* (rows x ldW), row-major
 * @param x     __global const float* (length cols)
 * @param y     __global float*       (length rows)
 * @param rows  Number of rows in W (and length of y)
 * @param cols  Number of columns in W (and length of x)
 * @param ldW   Leading dimension for W (ldW >= cols)
 * @param alpha Scalar multiplier for W*x
 * @param beta  Scalar multiplier for existing y
 */
__kernel __attribute__((reqd_work_group_size(IE_WG_TILE, 1, 1)))
void gemv_rowwise_f32(__global const float* restrict W,
                      __global const float* restrict x,
                      __global float*       restrict y,
                      int rows, int cols, int ldW,
                      float alpha, float beta) {
  const int r  = get_global_id(0);
  const int lr = get_local_id(0);
  if (r >= rows) return;

  float acc = 0.0f;
  /* Strided partial dot-product over this thread's lane */
  for (int k = lr; k < cols; k += IE_WG_TILE) {
    const size_t idx = (size_t)r * (size_t)ldW + (size_t)k;
    acc += W[idx] * x[k];
  }

  /* Intra-group reduction */
  __local float buf[IE_WG_TILE];
  buf[lr] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = IE_WG_TILE >> 1; s > 0; s >>= 1) {
    if (lr < s) {
      buf[lr] += buf[lr + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lr == 0) {
    float out = alpha * buf[0];
    if (beta != 0.0f) out += beta * y[r];
    y[r] = out;
  }
}

/**
 * @brief Fused GEMV + bias + activation:
 *        y = act(alpha * W*x + bias + beta*y)
 *
 * Identical reduction scheme and local-size requirement as gemv_rowwise_f32.
 *
 * Preconditions:
 *  - Local work-group size X is exactly 256.
 *  - If @p bias is NULL, bias is treated as 0.
 *
 * @param W     __global const float* (rows x ldW), row-major
 * @param x     __global const float* (cols)
 * @param bias  __global const float* (rows) or NULL
 * @param y     __global float*       (rows)
 * @param rows  Number of rows in W / y
 * @param cols  Number of columns in W / x
 * @param ldW   Leading dimension for W (ldW >= cols)
 * @param alpha Scalar multiplier for W*x
 * @param beta  Scalar multiplier for existing y
 * @param act   Activation kind (0 none, 1 relu, 2 tanh)
 */
__kernel __attribute__((reqd_work_group_size(IE_WG_TILE, 1, 1)))
void gemv_bias_act_f32(__global const float* restrict W,
                       __global const float* restrict x,
                       __global const float* restrict bias,
                       __global float*       restrict y,
                       int rows, int cols, int ldW,
                       float alpha, float beta, int act) {
  const int r  = get_global_id(0);
  const int lr = get_local_id(0);
  if (r >= rows) return;

  float acc = 0.0f;
  for (int k = lr; k < cols; k += IE_WG_TILE) {
    const size_t idx = (size_t)r * (size_t)ldW + (size_t)k;
    acc += W[idx] * x[k];
  }

  __local float buf[IE_WG_TILE];
  buf[lr] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = IE_WG_TILE >> 1; s > 0; s >>= 1) {
    if (lr < s) {
      buf[lr] += buf[lr + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lr == 0) {
    const float b = (bias == NULL) ? 0.0f : bias[r];
    float out = alpha * buf[0] + b;
    if (beta != 0.0f) out += beta * y[r];
    y[r] = ie_apply_activation(out, act);
  }
}

/* --------------------------- Vector tanh --------------------------------- */

/**
 * @brief Elementwise hyperbolic tangent: y[i] = tanh(x[i]).
 *
 * Work-items iterate in a grid-stride loop, covering the full range.
 *
 * @param y __global float* (output, length n)
 * @param x __global const float* (input, length n)
 * @param n int (number of elements)
 */
__kernel void vec_tanh_f32(__global float*       restrict y,
                           __global const float* restrict x,
                           int n) {
  const int gid = get_global_id(0);
  const int gsz = get_global_size(0);
  for (int i = gid; i < n; i += gsz) {
    y[i] = tanh(x[i]);
  }
}

/* ------------------------------ Packing ---------------------------------- */

/**
 * @brief Pack row-major W (rows x ldW) into a blocked-K layout Wp.
 *
 * Destination layout (for block_k):
 *   int kb  = k / block_k;                // tile index along K
 *   int ko  = k % block_k;                // in-tile offset
 *   size_t dst = ((size_t)kb * rows + r) * (size_t)block_k + (size_t)ko;
 *   Wp[dst] = W[(size_t)r * ldW + (size_t)k];
 *
 * Global NDRange:
 *   Gx over K dimension; Gy over rows.
 *   Each work-item grid-strides in both r and k to cover the matrix.
 *
 * @param Wp      __global float*       (rows*cols), destination (blocked-K)
 * @param W       __global const float* (rows x ldW), source (row-major)
 * @param rows    int
 * @param cols    int
 * @param ldW     int (>= cols)
 * @param block_k int (tile size along K)
 */
__kernel void pack_w_blockedk_f32(__global float*       restrict Wp,
                                  __global const float* restrict W,
                                  int rows, int cols, int ldW,
                                  int block_k) {
  const int k0 = get_global_id(0);
  const int r0 = get_global_id(1);
  const int Gx = get_global_size(0);
  const int Gy = get_global_size(1);

  for (int r = r0; r < rows; r += Gy) {
    for (int k = k0; k < cols; k += Gx) {
      const int kb = k / block_k;
      const int ko = k % block_k;
      const size_t dst = ((size_t)kb * (size_t)rows + (size_t)r) * (size_t)block_k + (size_t)ko;
      Wp[dst] = W[(size_t)r * (size_t)ldW + (size_t)k];
    }
  }
}
