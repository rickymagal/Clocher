/**
 * @file microbench_block_sparse.c
 * @brief Microbenchmark comparing dense vs block-sparse GEMV (FP32).
 *
 * @details
 * This benchmark:
 *  - Synthesizes a large dense matrix W (rows x cols) and input vector x.
 *  - Converts W into a block-sparse representation (BSR) in-memory using
 *    the same layout as defined in sparse_format.h.
 *  - Runs repeated GEMV operations using:
 *      (a) a naive dense row-major GEMV, and
 *      (b) ie_gemv_block_sparse_f32 on the block-sparse matrix.
 *  - Reports timings, effective density, and a rough memory-traffic estimate.
 *
 * The goal is to provide a standalone way to measure the potential benefit of
 * block sparsity for large matrices under a simple max-abs block pruning rule.
 *
 * Usage:
 *   ./build/microbench_block_sparse [rows] [cols] [iters]
 *                                   [block_rows] [block_cols] [threshold]
 *
 * Defaults:
 *   rows       = 4096
 *   cols       = 4096
 *   iters      = 50
 *   block_rows = 16
 *   block_cols = 16
 *   threshold  = 1e-3
 *
 * The pruning rule is:
 *   - A block is kept if any element has |w| >= threshold.
 *   - Otherwise the block is dropped.
 *
 * Notes:
 *  - This is CPU-only and single-threaded on purpose.
 *  - It uses the same BSR format as the engine (see sparse_format.h) and
 *    calls ie_gemv_block_sparse_f32 from gemm_block_sparse.c.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparse_format.h"

/* -------------------------------------------------------------------------- */
/* Timing helpers                                                             */
/* -------------------------------------------------------------------------- */

/**
 * @brief Portable timestamp in seconds using C11 timespec_get.
 * @return Current time as double seconds.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* -------------------------------------------------------------------------- */
/* Dense helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Initialize an array with pseudo-random FP32 in [-scale, +scale].
 * @param p     Destination pointer.
 * @param n     Number of elements.
 * @param seed  Seed (modified internally).
 * @param scale Half-range for uniform distribution.
 */
static void init_rand(float *p, size_t n, unsigned *seed, float scale) {
  for (size_t i = 0; i < n; ++i) {
    *seed = *seed * 1664525u + 1013904223u;
    float u = (float)((*seed & 0xFFFFFFu) / 16777216.0f);
    p[i] = (u * 2.0f - 1.0f) * scale;
  }
}

/**
 * @brief Row-major GEMV: y = W * x.
 * @param W    Weights [rows x cols], row-major.
 * @param x    Input vector [cols].
 * @param y    Output vector [rows].
 * @param rows Number of rows.
 * @param cols Number of columns.
 */
static void gemv_rowmajor(const float *W, const float *x, float *y,
                          size_t rows, size_t cols) {
  for (size_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const float *w = W + r * cols;
    for (size_t c = 0; c < cols; ++c) {
      acc += w[c] * x[c];
    }
    y[r] = acc;
  }
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
                            size_t rows,
                            size_t cols,
                            size_t br,
                            size_t bc,
                            size_t block_rows,
                            size_t block_cols,
                            float threshold) {
  const size_t row_start = br * block_rows;
  const size_t col_start = bc * block_cols;

  for (size_t lr = 0; lr < block_rows; ++lr) {
    const size_t r = row_start + lr;
    if (r >= rows) {
      break;
    }
    const float *row = W + r * cols;
    for (size_t lc = 0; lc < block_cols; ++lc) {
      const size_t c = col_start + lc;
      if (c >= cols) {
        break;
      }
      const float v = row[c];
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

  const size_t total_blocks =
      (size_t)n_block_rows * (size_t)n_block_cols;

  unsigned char *nonzero =
      (unsigned char *)calloc(total_blocks, sizeof(unsigned char));
  if (!nonzero) {
    fprintf(stderr, "microbench_block_sparse: allocation failure (nonzero)\n");
    return -1;
  }

  uint32_t nnzb = 0u;
  for (uint32_t br = 0; br < n_block_rows; ++br) {
    for (uint32_t bc = 0; bc < n_block_cols; ++bc) {
      const size_t idx = (size_t)br * (size_t)n_block_cols + (size_t)bc;
      const int nz = block_is_nonzero(W,
                                      (size_t)rows,
                                      (size_t)cols,
                                      (size_t)br,
                                      (size_t)bc,
                                      (size_t)block_rows,
                                      (size_t)block_cols,
                                      threshold);
      if (nz) {
        nonzero[idx] = 1u;
        ++nnzb;
      }
    }
  }

  uint32_t *row_ptr =
      (uint32_t *)malloc((size_t)(n_block_rows + 1u) * sizeof(uint32_t));
  uint32_t *col_idx =
      (uint32_t *)malloc((size_t)nnzb * sizeof(uint32_t));
  float *values = NULL;

  if (!row_ptr || (!col_idx && nnzb > 0u)) {
    fprintf(stderr, "microbench_block_sparse: allocation failure (csr)\n");
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
      fprintf(stderr, "microbench_block_sparse: allocation failure (values)\n");
      free(row_ptr);
      free(col_idx);
      free(nonzero);
      return -1;
    }
  }

  uint32_t cursor = 0u;
  for (uint32_t br = 0; br < n_block_rows; ++br) {
    row_ptr[br] = cursor;
    for (uint32_t bc = 0; bc < n_block_cols; ++bc) {
      const size_t idx = (size_t)br * (size_t)n_block_cols + (size_t)bc;
      if (!nonzero[idx]) {
        continue;
      }

      const uint32_t blk_id = cursor;
      col_idx[blk_id] = bc;

      const uint32_t row_start = br * block_rows;
      const uint32_t col_start = bc * block_cols;
      float *blk = values + (size_t)blk_id *
                   (size_t)block_rows * (size_t)block_cols;

      for (uint32_t lr = 0; lr < block_rows; ++lr) {
        const uint32_t r = row_start + lr;
        if (r >= rows) {
          break;
        }
        const float *row = W + (size_t)r * (size_t)cols;
        for (uint32_t lc = 0; lc < block_cols; ++lc) {
          const uint32_t c = col_start + lc;
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
            "microbench_block_sparse: nnzb mismatch (expected %u, built %u)\n",
            nnzb, cursor);
    free(row_ptr);
    free(col_idx);
    free(values);
    free(nonzero);
    return -1;
  }

  free(nonzero);

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
/* Main benchmark                                                             */
/* -------------------------------------------------------------------------- */

int main(int argc, char **argv) {
  /* Defaults: "large" weights by default. */
  size_t rows = (argc > 1) ? (size_t)strtoul(argv[1], NULL, 10) : 4096;
  size_t cols = (argc > 2) ? (size_t)strtoul(argv[2], NULL, 10) : 4096;
  size_t iters = (argc > 3) ? (size_t)strtoul(argv[3], NULL, 10) : 50;
  uint32_t block_rows = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 16u;
  uint32_t block_cols = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 16u;
  float threshold =
      (argc > 6) ? (float)strtod(argv[6], NULL) : 1e-3f;

  printf("microbench_block_sparse:\n");
  printf("  rows       = %zu\n", rows);
  printf("  cols       = %zu\n", cols);
  printf("  iters      = %zu\n", iters);
  printf("  block_rows = %u\n", block_rows);
  printf("  block_cols = %u\n", block_cols);
  printf("  threshold  = %g\n", (double)threshold);

  const size_t nW = rows * cols;
  const size_t nx = cols;
  const size_t ny = rows;

  float *W_dense = (float *)malloc(nW * sizeof(float));
  float *x       = (float *)malloc(nx * sizeof(float));
  float *y_dense = (float *)malloc(ny * sizeof(float));
  float *y_sparse = (float *)malloc(ny * sizeof(float));

  if (!W_dense || !x || !y_dense || !y_sparse) {
    fprintf(stderr, "microbench_block_sparse: allocation failure\n");
    free(W_dense);
    free(x);
    free(y_dense);
    free(y_sparse);
    return 1;
  }

  unsigned seed = 12345u;
  init_rand(W_dense, nW, &seed, 1.0f / 32.0f);
  init_rand(x, nx, &seed, 1.0f);

  /* Convert dense -> block-sparse (this is our "large weight" conversion). */
  ie_block_sparse_matrix_t m;
  if (dense_to_block_sparse(W_dense,
                            (uint32_t)rows,
                            (uint32_t)cols,
                            block_rows,
                            block_cols,
                            threshold,
                            &m) != 0) {
    fprintf(stderr, "microbench_block_sparse: dense_to_block_sparse failed\n");
    free(W_dense);
    free(x);
    free(y_dense);
    free(y_sparse);
    return 1;
  }

  const float density = ie_block_sparse_density(m.nnzb,
                                                m.n_block_rows,
                                                m.n_block_cols);
  printf("  nnzb       = %u / %u blocks (density = %.6f)\n",
         m.nnzb,
         m.n_block_rows * m.n_block_cols,
         (double)density);

  /* Quick correctness check on a single multiply. */
  gemv_rowmajor(W_dense, x, y_dense, rows, cols);
  ie_gemv_block_sparse_f32(&m, x, y_sparse, NULL);

  double max_abs_diff = 0.0;
  for (size_t i = 0; i < ny; ++i) {
    const double d = fabs((double)y_dense[i] - (double)y_sparse[i]);
    if (d > max_abs_diff) {
      max_abs_diff = d;
    }
  }
  printf("  max_abs_diff(dense vs sparse) = %.6g\n", max_abs_diff);

  /* Warmup. */
  gemv_rowmajor(W_dense, x, y_dense, rows, cols);
  ie_gemv_block_sparse_f32(&m, x, y_sparse, NULL);

  /* Timed loop: dense. */
  double t0_dense = now_s();
  for (size_t it = 0; it < iters; ++it) {
    gemv_rowmajor(W_dense, x, y_dense, rows, cols);
  }
  double t1_dense = now_s();

  /* Timed loop: block-sparse. */
  double t0_sparse = now_s();
  for (size_t it = 0; it < iters; ++it) {
    ie_gemv_block_sparse_f32(&m, x, y_sparse, NULL);
  }
  double t1_sparse = now_s();

  const double dt_dense = t1_dense - t0_dense;
  const double dt_sparse = t1_sparse - t0_sparse;

  const double us_per_iter_dense = (dt_dense * 1e6) / (double)iters;
  const double us_per_iter_sparse = (dt_sparse * 1e6) / (double)iters;

  /* Rough memory traffic: assume read of W and x and write of y per iter. */
  const double bytes_dense =
      (double)(rows * cols + cols + rows) * sizeof(float);
  const double bytes_sparse =
      (double)m.nnzb *
          (double)m.block_rows * (double)m.block_cols * sizeof(float) +
      (double)m.cols * sizeof(float) +
      (double)m.rows * sizeof(float);

  const double gbps_dense =
      (bytes_dense * (double)iters / dt_dense) / 1e9;
  const double gbps_sparse =
      (bytes_sparse * (double)iters / dt_sparse) / 1e9;

  printf("\n=== Results ===\n");
  printf("dense:   %.3f ms total, %.2f us/iter, ~%.2f GB/s\n",
         dt_dense * 1000.0, us_per_iter_dense, gbps_dense);
  printf("sparse:  %.3f ms total, %.2f us/iter, ~%.2f GB/s\n",
         dt_sparse * 1000.0, us_per_iter_sparse, gbps_sparse);
  if (dt_sparse > 0.0) {
    printf("speedup (dense/sparse): %.3fx\n", dt_dense / dt_sparse);
  }

  ie_block_sparse_release(&m);
  free(W_dense);
  free(x);
  free(y_dense);
  free(y_sparse);

  return 0;
}
