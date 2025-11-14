/* ============================================================================
 * File: engine/src/gemm_block_sparse.c
 * ============================================================================
 */
/**
 * @file gemm_block_sparse.c
 * @brief Block-sparse GEMV kernel for single-precision weights.
 *
 * @details
 * This translation unit implements a reference GEMV kernel for matrices
 * stored in @ref ie_block_sparse_matrix_t (block-row CSR / BSR format).
 * It is intended as a baseline, portable implementation:
 *
 * - Single-threaded by design; higher-level code is responsible for
 *   sharding work across threads if desired.
 * - Numerically identical to the dense product (up to floating-point
 *   roundoff) when the block-sparse representation is exact.
 * - Safe in the presence of tail blocks (partial coverage at the edges).
 *
 * The main entry point is:
 * - @ref ie_gemv_block_sparse_f32
 */

#include "sparse_format.h"

#include <stddef.h>

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compute the pointer to the first element of a given block.
 *
 * @param m      Block-sparse matrix descriptor (must not be NULL).
 * @param blk_id Block index in the [0, m->nnzb) range.
 * @return Pointer into @ref ie_block_sparse_matrix_t::values for the
 *         first element of the requested block.
 */
static inline const float *block_values_ptr(const ie_block_sparse_matrix_t *m,
                                            uint32_t blk_id) {
  const size_t block_elems =
      (size_t)m->block_rows * (size_t)m->block_cols;
  return m->values + (size_t)blk_id * block_elems;
}

/* -------------------------------------------------------------------------- */
/* Public GEMV                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * @brief Perform a single-precision block-sparse matrix-vector product.
 *
 * @details
 * This function computes:
 * @code
 * y = W * x  (+ bias, if provided)
 * @endcode
 * where @p W is represented in block-sparse form using
 * @ref ie_block_sparse_matrix_t. The input vector @p x has length
 * @c m->cols, and the output @p y has length @c m->rows.
 *
 * Tail blocks:
 * - If the matrix size is not an exact multiple of the block dimensions,
 *   the implementation clips accesses to @c rows and @c cols, ensuring
 *   no out-of-bounds reads/writes occur.
 *
 * Threading:
 * - This function is intentionally single-threaded. Thread-level parallelism
 *   should be managed by higher-level components (e.g., a thread pool).
 *
 * @param m    Block-sparse matrix descriptor (must be fully initialized).
 * @param x    Input vector of length @c m->cols (must not be NULL).
 * @param y    Output vector of length @c m->rows (must not be NULL).
 * @param bias Optional bias vector of length @c m->rows (may be NULL).
 */
void ie_gemv_block_sparse_f32(const ie_block_sparse_matrix_t *m,
                              const float *x,
                              float *y,
                              const float *bias) {
  if (!m || !x || !y) {
    return;
  }

  const uint32_t rows         = m->rows;
  const uint32_t cols         = m->cols;
  const uint32_t block_rows   = m->block_rows;
  const uint32_t block_cols   = m->block_cols;
  const uint32_t n_block_rows = m->n_block_rows;

  if (rows == 0u || cols == 0u ||
      block_rows == 0u || block_cols == 0u ||
      n_block_rows == 0u) {
    return;
  }

  /* For each block-row, process its constituent row stripes. */
  for (uint32_t br = 0u; br < n_block_rows; ++br) {
    const uint32_t row_start = br * block_rows;
    const uint32_t row_end   = row_start + block_rows;

    /* Iterate within the block-row over actual rows (tail-safe). */
    for (uint32_t local_r = 0u; local_r < block_rows; ++local_r) {
      const uint32_t row = row_start + local_r;
      if (row >= rows) {
        break;
      }

      float acc = bias ? bias[row] : 0.0f;

      const uint32_t blk_begin = m->row_ptr[br];
      const uint32_t blk_end   = m->row_ptr[br + 1u];

      /* Accumulate contributions from all non-zero blocks in this block-row. */
      for (uint32_t bi = blk_begin; bi < blk_end; ++bi) {
        const uint32_t bc = m->col_idx[bi];
        const uint32_t col_block_start = bc * block_cols;

        const float *block = block_values_ptr(m, bi);
        const float *block_row = block + (size_t)local_r * (size_t)block_cols;

        /* Inner product of the local block row with the corresponding
         * slice of x, with clipping for tail columns. */
        for (uint32_t local_c = 0u; local_c < block_cols; ++local_c) {
          const uint32_t col = col_block_start + local_c;
          if (col >= cols) {
            break;
          }
          acc += block_row[local_c] * x[col];
        }
      }

      y[row] = acc;
    }

    (void)row_end; /* suppress unused-variable warning if not referenced */
  }
}
