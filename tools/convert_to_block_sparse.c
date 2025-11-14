/**
 * @file convert_to_block_sparse.c
 * @brief Offline converter from dense FP32 matrix to block-sparse binary file.
 *
 * @details
 * This tool reads a dense FP32 matrix from disk (row-major, contiguous),
 * applies a simple block pruning rule, and writes a compact block-sparse
 * representation in the format consumed by ie_block_sparse_load().
 *
 * The on-disk layout matches the header described in sparse_io.c:
 *
 * @code
 * struct ie_block_sparse_header_disk {
 *   char     magic[4];      // "IEBS"
 *   uint32_t version;       // currently 1
 *   uint32_t rows;          // total dense rows
 *   uint32_t cols;          // total dense cols
 *   uint32_t block_rows;    // block height
 *   uint32_t block_cols;    // block width
 *   uint32_t nnzb;          // non-zero blocks
 *   uint32_t reserved[4];   // zeroed
 * };
 *
 * uint32_t row_ptr[n_block_rows + 1];
 * uint32_t col_idx[nnzb];
 * float    values[nnzb * block_rows * block_cols];
 * @endcode
 *
 * Non-zero block decision:
 * - A block is kept if any element has |w| >= threshold.
 * - Otherwise the block is dropped.
 *
 * Usage:
 * @code
 *   ./convert_to_block_sparse \
 *       --in dense.bin \
 *       --out weights.iebs \
 *       --rows 4096 \
 *       --cols 4096 \
 *       --block-rows 16 \
 *       --block-cols 16 \
 *       --threshold 0.0
 * @endcode
 *
 * The input file is expected to contain `rows * cols` FP32 values in
 * row-major order with no header.
 */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparse_format.h"

/* -------------------------------------------------------------------------- */
/* On-disk header (must match sparse_io.c)                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief On-disk header for block-sparse matrices (writer side).
 */
typedef struct ie_block_sparse_header_disk {
  char magic[4];
  uint32_t version;
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  uint32_t nnzb;
  uint32_t reserved[4];
} ie_block_sparse_header_disk_t;

/* -------------------------------------------------------------------------- */
/* CLI parsing                                                                */
/* -------------------------------------------------------------------------- */

typedef struct cli_opts {
  const char *in_path;
  const char *out_path;
  uint32_t rows;
  uint32_t cols;
  uint32_t block_rows;
  uint32_t block_cols;
  float threshold;
} cli_opts_t;

/**
 * @brief Print usage message to stderr.
 */
static void usage(const char *prog) {
  fprintf(stderr,
          "Usage:\n"
          "  %s --in PATH --out PATH --rows N --cols M \\\n"
          "     --block-rows BR --block-cols BC [--threshold T]\n\n"
          "Options:\n"
          "  --in PATH         Input dense FP32 file (row-major, rows*cols floats)\n"
          "  --out PATH        Output block-sparse file (.iebs)\n"
          "  --rows N          Number of rows in dense matrix\n"
          "  --cols M          Number of columns in dense matrix\n"
          "  --block-rows BR   Block height (rows per block, >0)\n"
          "  --block-cols BC   Block width (cols per block, >0)\n"
          "  --threshold T     Keep block if any |w| >= T (default: 0.0)\n",
          prog);
}

/**
 * @brief Parse CLI arguments into cli_opts_t.
 */
static int parse_cli(int argc, char **argv, cli_opts_t *opts) {
  if (!opts) {
    return -1;
  }
  memset(opts, 0, sizeof(*opts));
  opts->threshold = 0.0f;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--in") == 0 && i + 1 < argc) {
      opts->in_path = argv[++i];
    } else if (strcmp(arg, "--out") == 0 && i + 1 < argc) {
      opts->out_path = argv[++i];
    } else if (strcmp(arg, "--rows") == 0 && i + 1 < argc) {
      opts->rows = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--cols") == 0 && i + 1 < argc) {
      opts->cols = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--block-rows") == 0 && i + 1 < argc) {
      opts->block_rows = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--block-cols") == 0 && i + 1 < argc) {
      opts->block_cols = (uint32_t)strtoul(argv[++i], NULL, 10);
    } else if (strcmp(arg, "--threshold") == 0 && i + 1 < argc) {
      opts->threshold = (float)strtod(argv[++i], NULL);
    } else {
      fprintf(stderr, "Unknown or incomplete argument: %s\n", arg);
      return -1;
    }
  }

  if (!opts->in_path || !opts->out_path ||
      opts->rows == 0u || opts->cols == 0u ||
      opts->block_rows == 0u || opts->block_cols == 0u) {
    return -1;
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Dense matrix I/O                                                           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Load a dense FP32 matrix from disk.
 *
 * The file is expected to contain exactly rows*cols floats in row-major order.
 */
static float *load_dense_matrix(const char *path,
                                uint32_t rows,
                                uint32_t cols) {
  const size_t n = (size_t)rows * (size_t)cols;
  const size_t bytes = n * sizeof(float);

  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "failed to open input file '%s': %s\n",
            path, strerror(errno));
    return NULL;
  }

  float *buf = (float *)malloc(bytes);
  if (!buf) {
    fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
    fclose(f);
    return NULL;
  }

  size_t got = fread(buf, 1, bytes, f);
  if (got != bytes) {
    fprintf(stderr,
            "short read: expected %zu bytes, got %zu bytes from '%s'\n",
            bytes, got, path);
    free(buf);
    fclose(f);
    return NULL;
  }

  fclose(f);
  return buf;
}

/* -------------------------------------------------------------------------- */
/* Dense -> block-sparse conversion                                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief Decide whether a block is non-zero under a simple max-abs rule.
 *
 * @param W          Dense matrix (row-major, rows x cols).
 * @param rows       Total rows.
 * @param cols       Total cols.
 * @param br         Block row index.
 * @param bc         Block col index.
 * @param block_rows Block height.
 * @param block_cols Block width.
 * @param threshold  Minimum |w| to consider a value "non-zero".
 * @return 1 if block is non-zero, 0 otherwise.
 */
static int block_is_nonzero(const float *W,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t br,
                            uint32_t bc,
                            uint32_t block_rows,
                            uint32_t block_cols,
                            float threshold) {
  const uint32_t row_start = br * block_rows;
  const uint32_t col_start = bc * block_cols;

  for (uint32_t lr = 0; lr < block_rows; ++lr) {
    uint32_t r = row_start + lr;
    if (r >= rows) {
      break;
    }
    const float *row = W + (size_t)r * (size_t)cols;
    for (uint32_t lc = 0; lc < block_cols; ++lc) {
      uint32_t c = col_start + lc;
      if (c >= cols) {
        break;
      }
      float v = row[c];
      if (fabsf(v) >= threshold) {
        return 1;
      }
    }
  }
  return 0;
}

/**
 * @brief Build an ie_block_sparse_matrix_t from a dense matrix in memory.
 *
 * @param W          Dense input (row-major, rows x cols).
 * @param rows       Total rows.
 * @param cols       Total cols.
 * @param block_rows Block height.
 * @param block_cols Block width.
 * @param threshold  Min |w| to keep a block.
 * @param out        Destination descriptor.
 * @return 0 on success, non-zero on failure.
 */
static int dense_to_block_sparse(const float *W,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t block_rows,
                                 uint32_t block_cols,
                                 float threshold,
                                 ie_block_sparse_matrix_t *out) {
  if (!W || !out || rows == 0u || cols == 0u ||
      block_rows == 0u || block_cols == 0u) {
    return -1;
  }

  ie_block_sparse_init(out);

  const uint32_t n_block_rows =
      ie_block_sparse_n_block_rows(rows, block_rows);
  const uint32_t n_block_cols =
      ie_block_sparse_n_block_cols(cols, block_cols);

  if (n_block_rows == 0u || n_block_cols == 0u) {
    return -1;
  }

  /* First pass: detect which blocks are non-zero. */
  const size_t total_blocks =
      (size_t)n_block_rows * (size_t)n_block_cols;
  uint8_t *nonzero = (uint8_t *)calloc(total_blocks, sizeof(uint8_t));
  if (!nonzero) {
    fprintf(stderr, "out of memory allocating nonzero flags\n");
    return -1;
  }

  uint32_t nnzb = 0u;
  for (uint32_t br = 0; br < n_block_rows; ++br) {
    for (uint32_t bc = 0; bc < n_block_cols; ++bc) {
      size_t idx = (size_t)br * (size_t)n_block_cols + (size_t)bc;
      int nz = block_is_nonzero(W, rows, cols,
                                br, bc,
                                block_rows, block_cols,
                                threshold);
      if (nz) {
        nonzero[idx] = 1u;
        ++nnzb;
      }
    }
  }

  if (nnzb == 0u) {
    fprintf(stderr,
            "warning: no non-zero blocks detected; result will be empty\n");
  }

  /* Allocate CSR-like arrays. */
  uint32_t *row_ptr =
      (uint32_t *)malloc((size_t)(n_block_rows + 1u) * sizeof(uint32_t));
  uint32_t *col_idx =
      (uint32_t *)malloc((size_t)nnzb * sizeof(uint32_t));
  float *values = NULL;

  if (!row_ptr || (!col_idx && nnzb > 0u)) {
    fprintf(stderr, "out of memory allocating row_ptr/col_idx\n");
    free(row_ptr);
    free(col_idx);
    free(nonzero);
    return -1;
  }

  if (nnzb > 0u) {
    const size_t block_elems =
        (size_t)block_rows * (size_t)block_cols;
    values = (float *)calloc((size_t)nnzb * block_elems, sizeof(float));
    if (!values) {
      fprintf(stderr, "out of memory allocating values\n");
      free(row_ptr);
      free(col_idx);
      free(nonzero);
      return -1;
    }
  }

  /* Second pass: fill row_ptr, col_idx, values. */
  uint32_t cursor = 0u;
  for (uint32_t br = 0; br < n_block_rows; ++br) {
    row_ptr[br] = cursor;
    for (uint32_t bc = 0; bc < n_block_cols; ++bc) {
      size_t idx = (size_t)br * (size_t)n_block_cols + (size_t)bc;
      if (!nonzero[idx]) {
        continue;
      }
      /* This block is non-zero: assign col_idx and copy data. */
      const uint32_t blk_id = cursor;
      col_idx[blk_id] = bc;

      const uint32_t row_start = br * block_rows;
      const uint32_t col_start = bc * block_cols;
      float *blk = values + (size_t)blk_id *
                   (size_t)block_rows * (size_t)block_cols;

      for (uint32_t lr = 0; lr < block_rows; ++lr) {
        uint32_t r = row_start + lr;
        if (r >= rows) {
          break;
        }
        const float *row = W + (size_t)r * (size_t)cols;
        for (uint32_t lc = 0; lc < block_cols; ++lc) {
          uint32_t c = col_start + lc;
          if (c >= cols) {
            break;
          }
          blk[(size_t)lr * (size_t)block_cols + (size_t)lc] = row[c];
        }
      }

      ++cursor;
    }
  }
  row_ptr[n_block_rows] = cursor;

  if (cursor != nnzb) {
    fprintf(stderr,
            "internal error: nnzb mismatch (expected %u, built %u)\n",
            nnzb, cursor);
    free(row_ptr);
    free(col_idx);
    free(values);
    free(nonzero);
    return -1;
  }

  free(nonzero);

  /* Commit to descriptor. */
  out->rows         = rows;
  out->cols         = cols;
  out->block_rows   = block_rows;
  out->block_cols   = block_cols;
  out->n_block_rows = n_block_rows;
  out->n_block_cols = n_block_cols;
  out->nnzb         = nnzb;
  out->row_ptr      = row_ptr;
  out->col_idx      = col_idx;
  out->values       = values;

  return 0;
}

/* -------------------------------------------------------------------------- */
/* Writer                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Write a block-sparse matrix to disk in IEBS format.
 *
 * @param path Output file path.
 * @param m    Fully initialized descriptor.
 * @return 0 on success, non-zero on failure.
 */
static int write_block_sparse_file(const char *path,
                                   const ie_block_sparse_matrix_t *m) {
  if (!path || !m) {
    return -1;
  }

  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "failed to open output file '%s': %s\n",
            path, strerror(errno));
    return -1;
  }

  ie_block_sparse_header_disk_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic[0] = 'I';
  hdr.magic[1] = 'E';
  hdr.magic[2] = 'B';
  hdr.magic[3] = 'S';
  hdr.version     = 1u;
  hdr.rows        = m->rows;
  hdr.cols        = m->cols;
  hdr.block_rows  = m->block_rows;
  hdr.block_cols  = m->block_cols;
  hdr.nnzb        = m->nnzb;
  /* reserved[] left zeroed */

  if (fwrite(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
    fprintf(stderr, "failed to write header to '%s'\n", path);
    fclose(f);
    return -1;
  }

  const uint32_t n_block_rows = m->n_block_rows;
  const uint32_t nnzb         = m->nnzb;
  const size_t block_elems =
      (size_t)m->block_rows * (size_t)m->block_cols;
  const size_t values_count   = (size_t)nnzb * block_elems;

  if (fwrite(m->row_ptr,
             sizeof(uint32_t),
             (size_t)(n_block_rows + 1u),
             f) != (size_t)(n_block_rows + 1u)) {
    fprintf(stderr, "failed to write row_ptr to '%s'\n", path);
    fclose(f);
    return -1;
  }

  if (nnzb > 0u) {
    if (fwrite(m->col_idx,
               sizeof(uint32_t),
               (size_t)nnzb,
               f) != (size_t)nnzb) {
      fprintf(stderr, "failed to write col_idx to '%s'\n", path);
      fclose(f);
      return -1;
    }

    if (fwrite(m->values,
               sizeof(float),
               values_count,
               f) != values_count) {
      fprintf(stderr, "failed to write values to '%s'\n", path);
      fclose(f);
      return -1;
    }
  }

  if (fclose(f) != 0) {
    fprintf(stderr, "failed to close output file '%s': %s\n",
            path, strerror(errno));
    return -1;
  }

  return 0;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
  cli_opts_t opts;
  if (parse_cli(argc, argv, &opts) != 0) {
    usage(argv[0]);
    return 1;
  }

  fprintf(stderr,
          "convert_to_block_sparse:\n"
          "  in        = %s\n"
          "  out       = %s\n"
          "  rows      = %u\n"
          "  cols      = %u\n"
          "  block_rows= %u\n"
          "  block_cols= %u\n"
          "  threshold = %g\n",
          opts.in_path,
          opts.out_path,
          opts.rows,
          opts.cols,
          opts.block_rows,
          opts.block_cols,
          (double)opts.threshold);

  float *dense = load_dense_matrix(opts.in_path,
                                   opts.rows,
                                   opts.cols);
  if (!dense) {
    return 1;
  }

  ie_block_sparse_matrix_t m;
  if (dense_to_block_sparse(dense,
                            opts.rows,
                            opts.cols,
                            opts.block_rows,
                            opts.block_cols,
                            opts.threshold,
                            &m) != 0) {
    fprintf(stderr, "failed to build block-sparse representation\n");
    free(dense);
    return 1;
  }

  free(dense);

  const float density = ie_block_sparse_density(m.nnzb,
                                                m.n_block_rows,
                                                m.n_block_cols);
  fprintf(stderr,
          "  nnzb      = %u / %u blocks (density = %.6f)\n",
          m.nnzb,
          m.n_block_rows * m.n_block_cols,
          (double)density);

  if (write_block_sparse_file(opts.out_path, &m) != 0) {
    fprintf(stderr, "failed to write output file\n");
    ie_block_sparse_release(&m);
    return 1;
  }

  ie_block_sparse_release(&m);
  fprintf(stderr, "done.\n");
  return 0;
}
