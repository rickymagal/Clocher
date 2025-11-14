/**
 * @file sparse_format.h
 * @brief In-memory representation and loader interface for block-sparse matrices.
 *
 * @details
 * This header defines:
 *
 * - The in-memory layout for block-sparse matrices used by the engine
 *   (see @ref ie_block_sparse_matrix_t).
 * - A small status-code enum for sparse-related operations
 *   (see @ref ie_sparse_status_t).
 * - Utility helpers for computing block grid sizes and densities.
 * - The public loader function @ref ie_block_sparse_load, which reads a
 *   compact binary representation into an in-memory descriptor.
 * - A reference single-precision GEMV kernel
 *   (see @ref ie_gemv_block_sparse_f32).
 *
 * The chosen layout is a **block-row CSR (BSR)**:
 *
 * - The full matrix has @c rows x @c cols elements.
 * - It is partitioned into blocks of @c block_rows x @c block_cols.
 * - Block rows are numbered from 0 to @c n_block_rows-1, where
 *   @c n_block_rows = ceil(rows / block_rows).
 * - Block columns are numbered from 0 to @c n_block_cols-1, where
 *   @c n_block_cols = ceil(cols / block_cols).
 * - Only non-zero blocks are stored; zero blocks are omitted.
 *
 * Storage:
 * - @ref ie_block_sparse_matrix_t::row_ptr has size (n_block_rows + 1).
 *   For block-row @c br, the non-zero block indices are in the range:
 *   [ row_ptr[br], row_ptr[br + 1] ).
 * - @ref ie_block_sparse_matrix_t::col_idx has size @c nnzb, where each
 *   entry stores a block-column index (0-based).
 * - @ref ie_block_sparse_matrix_t::values has size:
 *   @c nnzb * block_rows * block_cols.
 *   Values for block j are laid out in row-major order:
 *   @code
 *   offset = j * (block_rows * block_cols);
 *   // element (i, k) inside the block:
 *   values[offset + i * block_cols + k]
 *   @endcode
 *
 * Notes:
 * - Tail blocks (on the last block-row/column) may extend beyond the
 *   actual matrix dimensions. Consumers must clip accesses using the
 *   @c rows and @c cols fields.
 * - The on-disk format is defined by the implementation in
 *   @c engine/src/sparse_io.c and consumed via
 *   @ref ie_block_sparse_load.
 */

#ifndef IE_SPARSE_FORMAT_H_
#define IE_SPARSE_FORMAT_H_

#include <stddef.h>
#include <stdint.h>

#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Status codes for block-sparse operations.
 *
 * @details
 * These codes are returned by functions that manipulate block-sparse
 * matrices, in particular @ref ie_block_sparse_load. They are designed
 * to be lightweight and sufficient for logging and error propagation.
 */
typedef enum ie_sparse_status_e {
  /** Operation completed successfully. */
  IE_SPARSE_OK = 0,
  /** Invalid arguments (NULL pointer, size mismatch, etc.). */
  IE_SPARSE_ERR_ARGS = -1,
  /** Failed to open a required file. */
  IE_SPARSE_ERR_OPEN = -2,
  /** Failed to read from a required file. */
  IE_SPARSE_ERR_READ = -3,
  /** Encountered an unexpected magic number or version. */
  IE_SPARSE_ERR_MAGIC = -4,
  /** Encountered an invalid or unsupported on-disk format. */
  IE_SPARSE_ERR_FORMAT = -5,
  /** Memory allocation failure. */
  IE_SPARSE_ERR_NOMEM = -6
} ie_sparse_status_t;

/**
 * @brief Human-readable description of an @ref ie_sparse_status_t code.
 *
 * @param st Status code to describe.
 * @return Pointer to a static, NUL-terminated string describing the status.
 *
 * @note The returned pointer is valid for the lifetime of the process and
 *       must not be freed by the caller.
 */
const char *ie_sparse_status_str(ie_sparse_status_t st);

/**
 * @brief In-memory representation of a block-sparse matrix.
 *
 * @details
 * This structure contains all metadata and storage required to perform
 * matrix-vector products in block-sparse form. It is owned by the caller;
 * loader functions such as @ref ie_block_sparse_load will allocate and
 * populate the arrays and can later be released using
 * @ref ie_block_sparse_release.
 *
 * Lifetime rules:
 * - All pointer fields are either NULL (when @c nnzb is zero) or point to
 *   heap-allocated memory obtained via @c malloc.
 * - After a successful load, the caller must eventually call
 *   @ref ie_block_sparse_release on the descriptor.
 * - After release, the descriptor is left in a zeroed state and may be
 *   reused or safely destroyed.
 */
typedef struct ie_block_sparse_matrix {
  /** Total number of rows in the original dense matrix. */
  uint32_t rows;
  /** Total number of columns in the original dense matrix. */
  uint32_t cols;

  /** Block height (number of rows per block). */
  uint32_t block_rows;
  /** Block width (number of columns per block). */
  uint32_t block_cols;

  /** Number of block-rows (ceil(rows / block_rows)). */
  uint32_t n_block_rows;
  /** Number of block-columns (ceil(cols / block_cols)). */
  uint32_t n_block_cols;

  /** Number of non-zero blocks stored in this matrix. */
  uint32_t nnzb;

  /**
   * @brief Block-row pointer array of length (n_block_rows + 1).
   *
   * @details
   * For block-row @c br, the non-zero blocks are found in the range:
   * [ row_ptr[br], row_ptr[br + 1] ) within @ref col_idx and
   * @ref values.
   */
  uint32_t *row_ptr;

  /**
   * @brief Block-column indices array of length @c nnzb.
   *
   * @details
   * Each entry is a 0-based block-column index referencing the block-column
   * of the corresponding non-zero block.
   */
  uint32_t *col_idx;

  /**
   * @brief Dense values for all non-zero blocks, in row-major order.
   *
   * @details
   * Size is @c nnzb * block_rows * block_cols. For block j, the data is
   * laid out as:
   *   values[j * (block_rows * block_cols) + i * block_cols + k]
   * representing the element (i, k) in the local block coordinates.
   *
   * The current implementation uses 32-bit floats. Additional data types
   * (e.g., BF16) can be added later via separate descriptors or metadata.
   */
  float *values;
} ie_block_sparse_matrix_t;

/**
 * @brief Initialize a block-sparse matrix descriptor to a safe empty state.
 *
 * @details
 * This function does not allocate any memory. It only sets all fields to
 * zero or NULL, making the descriptor safe to pass to
 * @ref ie_block_sparse_release or to any loader function that expects
 * an initially empty descriptor.
 *
 * @param m Descriptor to initialize (may be NULL, in which case this
 *          function is a no-op).
 */
void ie_block_sparse_init(ie_block_sparse_matrix_t *m);

/**
 * @brief Release all heap-allocated storage owned by a block-sparse matrix.
 *
 * @details
 * If @p m is NULL, this function is a no-op. Otherwise it:
 * - Calls @c free on @ref ie_block_sparse_matrix_t::row_ptr,
 *   @ref ie_block_sparse_matrix_t::col_idx, and
 *   @ref ie_block_sparse_matrix_t::values when non-NULL.
 * - Resets all scalar fields to zero.
 * - Sets all pointer fields to NULL.
 *
 * After this call, the descriptor is in the same state as after
 * @ref ie_block_sparse_init and may be reused.
 *
 * @param m Descriptor whose storage should be released (may be NULL).
 */
void ie_block_sparse_release(ie_block_sparse_matrix_t *m);

/**
 * @brief Compute the theoretical number of block rows for a given layout.
 *
 * @param rows       Total number of rows in the dense matrix.
 * @param block_rows Block height (rows per block, must be >0).
 * @return Number of block rows (ceil(rows / block_rows)), or 0 on invalid
 *         input.
 */
uint32_t ie_block_sparse_n_block_rows(uint32_t rows, uint32_t block_rows);

/**
 * @brief Compute the theoretical number of block columns for a given layout.
 *
 * @param cols       Total number of columns in the dense matrix.
 * @param block_cols Block width (columns per block, must be >0).
 * @return Number of block columns (ceil(cols / block_cols)), or 0 on invalid
 *         input.
 */
uint32_t ie_block_sparse_n_block_cols(uint32_t cols, uint32_t block_cols);

/**
 * @brief Compute the logical density (non-zero blocks / total blocks).
 *
 * @details
 * The return value is a floating-point fraction between 0.0 and 1.0.
 * If any of the parameters is zero, this function returns 0.0.
 *
 * @param nnzb         Number of non-zero blocks.
 * @param n_block_rows Number of block rows.
 * @param n_block_cols Number of block columns.
 * @return Density as a float in [0.0, 1.0].
 */
float ie_block_sparse_density(uint32_t nnzb,
                              uint32_t n_block_rows,
                              uint32_t n_block_cols);

/**
 * @brief Load a block-sparse matrix from a binary file.
 *
 * @details
 * This function reads the on-disk format defined by the implementation in
 * @c engine/src/sparse_io.c and populates @p out with a fully initialized
 * @ref ie_block_sparse_matrix_t instance.
 *
 * On success:
 * - @ref ie_block_sparse_matrix_t::rows and @c cols reflect the dense
 *   dimensions.
 * - @ref ie_block_sparse_matrix_t::block_rows and @c block_cols reflect
 *   the block layout.
 * - @ref ie_block_sparse_matrix_t::n_block_rows and @c n_block_cols are
 *   derived from the dense dimensions and block sizes.
 * - @ref ie_block_sparse_matrix_t::row_ptr, @c col_idx, and @c values
 *   are allocated and filled.
 *
 * On failure:
 * - @ref IE_SPARSE_ERR_* is returned.
 * - @p out is left in a zeroed / empty state (safe to reuse or release).
 *
 * @param path Filesystem path to the binary block-sparse matrix.
 * @param out  Destination descriptor (must not be NULL).
 * @return @ref IE_SPARSE_OK on success, error code otherwise.
 */
ie_sparse_status_t ie_block_sparse_load(const char *path,
                                        ie_block_sparse_matrix_t *out);

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
                              const float *bias);

#ifdef __cplusplus
}
#endif

#endif /* IE_SPARSE_FORMAT_H_ */
