/* ============================================================================
 * File: engine/src/sparse_io.c
 * ============================================================================
 */
/**
 * @file sparse_io.c
 * @brief Loader for block-sparse weight matrices from a compact binary format.
 *
 * @details
 * This translation unit implements a minimal on-disk format and corresponding
 * loader for @ref ie_block_sparse_matrix_t. It is designed to be:
 *
 * - **Compact**: only non-zero blocks and essential metadata are stored.
 * - **Portable across typical x86_64 setups**: assumes little-endian layout
 *   and 32-bit integers for sizes and indices.
 * - **Strict but forgiving**: basic validation is performed on all fields
 *   (magic, version, dimensions, counts). On failure, no partially loaded
 *   state is left in the descriptor.
 *
 * The on-disk layout is:
 *
 * @code
 * struct ie_block_sparse_header_disk {
 *   char     magic[4];      // "IEBS"
 *   uint32_t version;       // currently 1
 *   uint32_t rows;          // total dense rows
 *   uint32_t cols;          // total dense columns
 *   uint32_t block_rows;    // block height
 *   uint32_t block_cols;    // block width
 *   uint32_t nnzb;          // number of non-zero blocks
 *   uint32_t reserved[4];   // reserved for future use (zeroed)
 * };
 *
 * // Followed by:
 * uint32_t row_ptr[n_block_rows + 1];
 * uint32_t col_idx[nnzb];
 * float    values[nnzb * block_rows * block_cols];
 * @endcode
 *
 * where:
 * - @c n_block_rows = ceil(rows / block_rows)
 * - @c n_block_cols = ceil(cols / block_cols)
 *
 * The loader validates that:
 * - @c block_rows and @c block_cols are non-zero.
 * - @c rows and @c cols are non-zero.
 * - @c nnzb is consistent with the size of the @c values array.
 *
 * Error handling:
 * - Functions return @ref ie_sparse_status_t codes.
 * - On any error, the output descriptor is left in a zeroed state.
 */

#include "sparse_format.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* On-disk header                                                             */
/* -------------------------------------------------------------------------- */

/**
 * @brief On-disk header for block-sparse matrices.
 *
 * @note This type is not exposed in public headers; it is specific to the
 *       implementation of @ref ie_block_sparse_load.
 */
typedef struct ie_block_sparse_header_disk {
  char magic[4];       /**< Magic bytes, must be "IEBS". */
  uint32_t version;    /**< File format version (currently 1). */
  uint32_t rows;       /**< Total dense rows. */
  uint32_t cols;       /**< Total dense columns. */
  uint32_t block_rows; /**< Block height (rows per block). */
  uint32_t block_cols; /**< Block width (cols per block). */
  uint32_t nnzb;       /**< Number of non-zero blocks. */
  uint32_t reserved[4];/**< Reserved for future use (zeroed). */
} ie_block_sparse_header_disk_t;

/* -------------------------------------------------------------------------- */
/* Local helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Safely read @p size bytes from a stream into @p ptr.
 *
 * @details
 * This helper wraps @c fread and ensures that exactly @p size bytes are
 * read (unless the file terminates early or an error occurs).
 *
 * @param ptr   Destination buffer (must not be NULL).
 * @param size  Number of bytes to read.
 * @param f     Opened file stream (must not be NULL).
 * @return @ref IE_SPARSE_OK on success, @ref IE_SPARSE_ERR_READ on failure.
 */
static ie_sparse_status_t read_fully(void *ptr, size_t size, FILE *f) {
  if (!ptr || !f) {
    return IE_SPARSE_ERR_ARGS;
  }
  if (size == 0) {
    return IE_SPARSE_OK;
  }
  const size_t n = fread(ptr, 1, size, f);
  if (n != size) {
    return IE_SPARSE_ERR_READ;
  }
  return IE_SPARSE_OK;
}

/* -------------------------------------------------------------------------- */
/* Public helpers from sparse_format.h                                        */
/* -------------------------------------------------------------------------- */

const char *ie_sparse_status_str(ie_sparse_status_t st) {
  switch (st) {
    case IE_SPARSE_OK:         return "ok";
    case IE_SPARSE_ERR_ARGS:   return "invalid arguments";
    case IE_SPARSE_ERR_OPEN:   return "failed to open file";
    case IE_SPARSE_ERR_READ:   return "failed to read file";
    case IE_SPARSE_ERR_MAGIC:  return "invalid magic or version";
    case IE_SPARSE_ERR_FORMAT: return "invalid or unsupported format";
    case IE_SPARSE_ERR_NOMEM:  return "out of memory";
    default:                   return "unknown sparse status";
  }
}

void ie_block_sparse_init(ie_block_sparse_matrix_t *m) {
  if (!m) {
    return;
  }
  memset(m, 0, sizeof(*m));
}

void ie_block_sparse_release(ie_block_sparse_matrix_t *m) {
  if (!m) {
    return;
  }
  free(m->row_ptr);
  free(m->col_idx);
  free(m->values);
  memset(m, 0, sizeof(*m));
}

uint32_t ie_block_sparse_n_block_rows(uint32_t rows, uint32_t block_rows) {
  if (rows == 0 || block_rows == 0) {
    return 0;
  }
  return (rows + block_rows - 1u) / block_rows;
}

uint32_t ie_block_sparse_n_block_cols(uint32_t cols, uint32_t block_cols) {
  if (cols == 0 || block_cols == 0) {
    return 0;
  }
  return (cols + block_cols - 1u) / block_cols;
}

float ie_block_sparse_density(uint32_t nnzb,
                              uint32_t n_block_rows,
                              uint32_t n_block_cols) {
  const uint64_t total_blocks =
      (uint64_t)n_block_rows * (uint64_t)n_block_cols;
  if (total_blocks == 0u) {
    return 0.0f;
  }
  if (nnzb == 0u) {
    return 0.0f;
  }
  const float num = (float)nnzb;
  const float den = (float)total_blocks;
  return num / den;
}

/* -------------------------------------------------------------------------- */
/* Public loader API                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Load a block-sparse matrix from a binary file.
 *
 * @details
 * This function reads the on-disk format described in the file header
 * documentation above and populates @p out with a fully initialized
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
                                        ie_block_sparse_matrix_t *out) {
  if (!path || !out) {
    return IE_SPARSE_ERR_ARGS;
  }

  ie_block_sparse_init(out);

  FILE *f = fopen(path, "rb");
  if (!f) {
    return IE_SPARSE_ERR_OPEN;
  }

  ie_block_sparse_header_disk_t hdr;
  const ie_sparse_status_t st_hdr = read_fully(&hdr, sizeof(hdr), f);
  if (st_hdr != IE_SPARSE_OK) {
    fclose(f);
    return st_hdr;
  }

  /* Validate magic and version. */
  if (memcmp(hdr.magic, "IEBS", 4) != 0 || hdr.version != 1u) {
    fclose(f);
    return IE_SPARSE_ERR_MAGIC;
  }

  /* Basic sanity checks on dimensions. */
  if (hdr.rows == 0u || hdr.cols == 0u ||
      hdr.block_rows == 0u || hdr.block_cols == 0u) {
    fclose(f);
    return IE_SPARSE_ERR_FORMAT;
  }

  const uint32_t n_block_rows =
      ie_block_sparse_n_block_rows(hdr.rows, hdr.block_rows);
  const uint32_t n_block_cols =
      ie_block_sparse_n_block_cols(hdr.cols, hdr.block_cols);
  const uint32_t nnzb = hdr.nnzb;

  /* Guard against overflow in total values. */
  const uint64_t block_elems =
      (uint64_t)hdr.block_rows * (uint64_t)hdr.block_cols;
  const uint64_t total_vals = block_elems * (uint64_t)nnzb;

  if (n_block_rows == 0u || n_block_cols == 0u || block_elems == 0u) {
    fclose(f);
    return IE_SPARSE_ERR_FORMAT;
  }

  if (total_vals > (uint64_t)SIZE_MAX / sizeof(float)) {
    fclose(f);
    return IE_SPARSE_ERR_FORMAT;
  }

  /* Allocate arrays. */
  uint32_t *row_ptr = (uint32_t *)malloc((size_t)(n_block_rows + 1u) *
                                         sizeof(uint32_t));
  uint32_t *col_idx = (uint32_t *)malloc((size_t)nnzb * sizeof(uint32_t));
  float *values = NULL;

  if (!row_ptr || (!col_idx && nnzb > 0u)) {
    free(row_ptr);
    free(col_idx);
    fclose(f);
    return IE_SPARSE_ERR_NOMEM;
  }

  if (nnzb > 0u) {
    values = (float *)malloc((size_t)total_vals * sizeof(float));
    if (!values) {
      free(row_ptr);
      free(col_idx);
      fclose(f);
      return IE_SPARSE_ERR_NOMEM;
    }
  }

  /* Read arrays from file. */
  ie_sparse_status_t st = IE_SPARSE_OK;

  st = read_fully(row_ptr,
                  (size_t)(n_block_rows + 1u) * sizeof(uint32_t),
                  f);
  if (st != IE_SPARSE_OK) {
    goto fail;
  }

  if (nnzb > 0u) {
    st = read_fully(col_idx, (size_t)nnzb * sizeof(uint32_t), f);
    if (st != IE_SPARSE_OK) {
      goto fail;
    }

    st = read_fully(values, (size_t)total_vals * sizeof(float), f);
    if (st != IE_SPARSE_OK) {
      goto fail;
    }
  }

  fclose(f);

  /* Commit to the output descriptor. */
  out->rows        = hdr.rows;
  out->cols        = hdr.cols;
  out->block_rows  = hdr.block_rows;
  out->block_cols  = hdr.block_cols;
  out->n_block_rows = n_block_rows;
  out->n_block_cols = n_block_cols;
  out->nnzb        = nnzb;
  out->row_ptr     = row_ptr;
  out->col_idx     = col_idx;
  out->values      = values;

  return IE_SPARSE_OK;

fail:
  free(row_ptr);
  free(col_idx);
  free(values);
  fclose(f);
  ie_block_sparse_init(out);
  return st;
}
