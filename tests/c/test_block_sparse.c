/**
 * @file test_block_sparse.c
 * @brief Unit tests for block-sparse format helpers and GEMV kernel.
 */

#include "sparse_format.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compare two floats with absolute tolerance.
 *
 * @param a   First value.
 * @param b   Second value.
 * @param eps Absolute tolerance.
 * @return 1 if |a-b| <= eps, 0 otherwise.
 */
static int feq(float a, float b, float eps) {
  const float diff = (float)fabsf(a - b);
  return diff <= eps;
}

/**
 * @brief Dump a dense vector to stderr (debug helper).
 *
 * @param name Name to print.
 * @param v    Vector pointer.
 * @param n    Number of elements.
 */
static void dump_vec(const char *name, const float *v, uint32_t n) {
  fprintf(stderr, "%s:", name);
  for (uint32_t i = 0; i < n; ++i) {
    fprintf(stderr, " %.6f", v[i]);
  }
  fprintf(stderr, "\n");
}

/* -------------------------------------------------------------------------- */
/* Tiny 4x4 example in BSR(2x2)                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build a 4x4 dense matrix used for testing.
 *
 * Layout (row-major):
 *   [ 1  2  0  0 ]
 *   [ 3  4  0  0 ]
 *   [ 0  0  5  6 ]
 *   [ 0  0  7  8 ]
 *
 * This can be represented with two non-zero 2x2 blocks on the diagonal.
 *
 * @param W_out Destination array of length 16 (row-major).
 */
static void build_dense_4x4(float *W_out) {
  static const float W[16] = {
    1.0f, 2.0f, 0.0f, 0.0f,
    3.0f, 4.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 5.0f, 6.0f,
    0.0f, 0.0f, 7.0f, 8.0f
  };
  memcpy(W_out, W, sizeof(W));
}

/**
 * @brief Build block-sparse descriptor for the 4x4 test matrix.
 *
 * The layout is BSR with 2x2 blocks:
 *   - rows = 4, cols = 4
 *   - block_rows = 2, block_cols = 2
 *   - n_block_rows = 2, n_block_cols = 2
 *   - nnzb = 2 (two diagonal blocks)
 *
 * Row pointers:
 *   row_ptr = [0, 1, 2]
 * Column indices:
 *   col_idx = [0, 1]
 * Values (two 2x2 blocks, row-major each):
 *   B0 = [1 2; 3 4]
 *   B1 = [5 6; 7 8]
 *
 * @param m Descriptor to populate (must not be NULL).
 */
static void build_block_sparse_4x4(ie_block_sparse_matrix_t *m) {
  assert(m != NULL);

  ie_block_sparse_init(m);

  m->rows       = 4u;
  m->cols       = 4u;
  m->block_rows = 2u;
  m->block_cols = 2u;

  m->n_block_rows = ie_block_sparse_n_block_rows(m->rows, m->block_rows);
  m->n_block_cols = ie_block_sparse_n_block_cols(m->cols, m->block_cols);
  m->nnzb         = 2u;

  const uint32_t n_block_rows = m->n_block_rows;
  const uint32_t nnzb         = m->nnzb;
  const size_t   block_elems  =
      (size_t)m->block_rows * (size_t)m->block_cols;

  uint32_t *row_ptr = (uint32_t *)malloc((size_t)(n_block_rows + 1u) *
                                         sizeof(uint32_t));
  uint32_t *col_idx = (uint32_t *)malloc((size_t)nnzb * sizeof(uint32_t));
  float    *values  = (float *)malloc((size_t)nnzb * block_elems *
                                      sizeof(float));

  assert(row_ptr != NULL);
  assert(col_idx != NULL);
  assert(values  != NULL);

  /* Row pointers: first block-row -> block 0; second block-row -> block 1. */
  row_ptr[0] = 0u;
  row_ptr[1] = 1u;
  row_ptr[2] = 2u;

  /* Column indices: both diagonal blocks. */
  col_idx[0] = 0u;
  col_idx[1] = 1u;

  /* Values: B0 then B1, each 2x2 row-major. */
  values[0] = 1.0f; values[1] = 2.0f;
  values[2] = 3.0f; values[3] = 4.0f;

  values[4] = 5.0f; values[5] = 6.0f;
  values[6] = 7.0f; values[7] = 8.0f;

  m->row_ptr = row_ptr;
  m->col_idx = col_idx;
  m->values  = values;
}

/* -------------------------------------------------------------------------- */
/* Tests                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Test helper functions for block grid sizes and density.
 *
 * @return 0 on success, non-zero on failure.
 */
static int test_block_sparse_helpers(void) {
  const uint32_t nbr = ie_block_sparse_n_block_rows(4u, 2u);
  const uint32_t nbc = ie_block_sparse_n_block_cols(4u, 2u);

  if (nbr != 2u || nbc != 2u) {
    fprintf(stderr, "n_block_rows/cols mismatch: got (%u,%u)\n", nbr, nbc);
    return 1;
  }

  const float d = ie_block_sparse_density(2u, nbr, nbc);
  if (!feq(d, 0.5f, 1e-6f)) {
    fprintf(stderr, "density mismatch: got %f, expected 0.5\n", d);
    return 2;
  }

  const uint32_t bad_rows = ie_block_sparse_n_block_rows(0u, 2u);
  const uint32_t bad_cols = ie_block_sparse_n_block_cols(4u, 0u);
  if (bad_rows != 0u || bad_cols != 0u) {
    fprintf(stderr, "expected zero for invalid grid sizes\n");
    return 3;
  }

  return 0;
}

/**
 * @brief Test that init and release leave descriptor in a safe state.
 *
 * @return 0 on success, non-zero on failure.
 */
static int test_block_sparse_init_release(void) {
  ie_block_sparse_matrix_t m;
  ie_block_sparse_init(&m);

  if (m.rows != 0u || m.cols != 0u ||
      m.row_ptr != NULL || m.col_idx != NULL || m.values != NULL) {
    fprintf(stderr, "init did not zero descriptor\n");
    return 1;
  }

  /* Allocate some storage and then release. */
  build_block_sparse_4x4(&m);
  if (m.row_ptr == NULL || m.col_idx == NULL || m.values == NULL) {
    fprintf(stderr, "build_block_sparse_4x4 produced NULL storage\n");
    return 2;
  }

  ie_block_sparse_release(&m);
  if (m.rows != 0u || m.cols != 0u ||
      m.row_ptr != NULL || m.col_idx != NULL || m.values != NULL) {
    fprintf(stderr, "release did not zero descriptor\n");
    return 3;
  }

  return 0;
}

/**
 * @brief Test GEMV on the 4x4 example with a simple vector.
 *
 * @return 0 on success, non-zero on failure.
 */
static int test_block_sparse_gemv_basic(void) {
  ie_block_sparse_matrix_t m;
  build_block_sparse_4x4(&m);

  float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  /* Expected dense result: y = W * x (with W from build_dense_4x4). */
  float W_dense[16];
  build_dense_4x4(W_dense);

  float y_ref[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t r = 0; r < 4u; ++r) {
    float acc = 0.0f;
    const float *row = &W_dense[r * 4u];
    for (uint32_t c = 0; c < 4u; ++c) {
      acc += row[c] * x[c];
    }
    y_ref[r] = acc;
  }

  ie_gemv_block_sparse_f32(&m, x, y, b);

  int err = 0;
  for (uint32_t i = 0; i < 4u; ++i) {
    if (!feq(y[i], y_ref[i], 1e-5f)) {
      fprintf(stderr,
              "gemv mismatch at %u: got %.6f, expected %.6f\n",
              i, y[i], y_ref[i]);
      dump_vec("y", y, 4u);
      dump_vec("y_ref", y_ref, 4u);
      err = 1;
      break;
    }
  }

  ie_block_sparse_release(&m);
  return err;
}

/**
 * @brief Test that loader fails with a missing file.
 *
 * This just checks error handling and status codes.
 *
 * @return 0 on success, non-zero on failure.
 */
static int test_block_sparse_load_missing(void) {
  ie_block_sparse_matrix_t m;
  ie_block_sparse_init(&m);

  const ie_sparse_status_t st =
      ie_block_sparse_load("this_file_should_not_exist.iebs", &m);

  if (st != IE_SPARSE_ERR_OPEN) {
    fprintf(stderr, "expected IE_SPARSE_ERR_OPEN, got %d (%s)\n",
            (int)st, ie_sparse_status_str(st));
    ie_block_sparse_release(&m);
    return 1;
  }

  /* Descriptor must remain empty. */
  if (m.rows != 0u || m.cols != 0u ||
      m.row_ptr != NULL || m.col_idx != NULL || m.values != NULL) {
    fprintf(stderr, "loader left non-empty descriptor on error\n");
    ie_block_sparse_release(&m);
    return 2;
  }

  return 0;
}

/* -------------------------------------------------------------------------- */
/* Main                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief Entry point: run all block-sparse tests.
 *
 * @return 0 on success, non-zero if any subtest fails.
 */
int main(void) {
  int rc = 0;

  rc = test_block_sparse_helpers();
  if (rc != 0) {
    fprintf(stderr, "[FAIL] test_block_sparse_helpers (rc=%d)\n", rc);
    return rc;
  }

  rc = test_block_sparse_init_release();
  if (rc != 0) {
    fprintf(stderr, "[FAIL] test_block_sparse_init_release (rc=%d)\n", rc);
    return rc;
  }

  rc = test_block_sparse_gemv_basic();
  if (rc != 0) {
    fprintf(stderr, "[FAIL] test_block_sparse_gemv_basic (rc=%d)\n", rc);
    return rc;
  }

  rc = test_block_sparse_load_missing();
  if (rc != 0) {
    fprintf(stderr, "[FAIL] test_block_sparse_load_missing (rc=%d)\n", rc);
    return rc;
  }

  printf("[OK] block-sparse tests passed\n");
  return 0;
}
