/* File: engine/src/opt/pretranspose.c
 * -----------------------------------------------------------------------------
 * @file pretranspose.c
 * @brief Build and cache blocked-K weight layout for GEMV.
 *
 * This module implements a light-weight re-packing of a row-major weight
 * matrix W (rows x cols) into a per-row, contiguous "blocked-K" layout,
 * suitable for cache-friendly inner loops. The packing is stable and
 * reversible: each output row preserves the original order of columns, only
 * grouped by contiguous blocks of size blk_k.
 *
 * The module also exposes a tiny on-disk cache keyed by a filename stem,
 * a layout tag ("wxh", "woh", etc.), and the structural parameters
 * (rows, cols, blk_k). The cache is optional and atomic on POSIX via
 * rename(2).
 *
 * Thread-safety: all functions are re-entrant; no global state is modified.
 * Ownership: the returned buffer in ::ie_layout_build_blockedK belongs to the
 * caller and must be released with ::ie_layout_free (or free()).
 */

#include "ie_layout.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/*                              Public utilities                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a filename for the on-disk cache of a blocked matrix.
 *
 * The generated name is deterministic and includes the structural parameters
 * so that changes in shape or block-size naturally invalidate older caches.
 *
 * @param base_path  Path or stem to namespace the cache (may be NULL/empty).
 * @param tag        Short, human-readable tag (e.g., "wxh", "woh").
 * @param rows       Row count in the logical matrix.
 * @param cols       Column count in the logical matrix.
 * @param blk_k      Column block size used for packing.
 * @param out_buf    Destination buffer for the resulting path.
 * @param out_cap    Capacity of @p out_buf in bytes (including NUL).
 * @return 0 on success; -1 on invalid arguments or buffer too small.
 */
int ie_layout_cache_name(const char *base_path, const char *tag,
                         size_t rows, size_t cols, size_t blk_k,
                         char *out_buf, size_t out_cap) {
  if (!tag || !out_buf || out_cap < 8) return -1;
  const char *stem = (base_path && *base_path) ? base_path : "weights.bin";
  int n = snprintf(out_buf, out_cap, "%s.%s.r%zu.c%zu.bk%zu.cache.bin",
                   stem, tag, rows, cols, blk_k);
  return (n > 0 && (size_t)n < out_cap) ? 0 : -1;
}

/* -------------------------------------------------------------------------- */
/*                              Internal helpers                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Load a blocked buffer from disk cache into a freshly malloc'ed array.
 *
 * The function validates the payload length by expecting exactly rows*cols
 * floats. It does not validate content beyond size and returns the raw buffer
 * as-is.
 *
 * @param path     Cache file path (may be NULL to force a miss).
 * @param rows     Expected number of rows.
 * @param cols     Expected number of cols.
 * @param data_out On success, *data_out points to a malloc'ed float array of
 *                 length rows*cols the caller owns.
 * @return 0 on a successful load; -1 on any error (including cache miss).
 */
static int load_cache(const char *path, size_t rows, size_t cols, float **data_out) {
  if (!path || !*path || !data_out) return -1;
  FILE *f = fopen(path, "rb");
  if (!f) return -1;

  size_t count = rows * cols;
  float *buf = (float *)malloc(count * sizeof(float));
  if (!buf) {
    fclose(f);
    return -1;
  }
  size_t got = fread(buf, sizeof(float), count, f);
  fclose(f);
  if (got != count) {
    free(buf);
    return -1;
  }
  *data_out = buf;
  return 0;
}

/**
 * @brief Save a blocked buffer to disk cache using an atomic rename.
 *
 * The function writes to a temporary file "<path>.tmp" and then renames it
 * to @p path. If any step fails, the temporary file is unlinked.
 *
 * @param path Destination file path.
 * @param data Source buffer (length rows*cols floats).
 * @param rows Number of rows.
 * @param cols Number of cols.
 * @return 0 on success; -1 on error.
 */
static int save_cache_atomic(const char *path, const float *data, size_t rows, size_t cols) {
  if (!path || !*path || !data) return -1;

  char tmp[4096];
  int n = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
  if (n <= 0 || (size_t)n >= sizeof(tmp)) return -1;

  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;

  size_t count = rows * cols;
  size_t put = fwrite(data, sizeof(float), count, f);
  if (put != count) {
    fclose(f);
    (void)remove(tmp);
    return -1;
  }
  if (fclose(f) != 0) {
    (void)remove(tmp);
    return -1;
  }

  if (rename(tmp, path) != 0) {
    (void)remove(tmp);
    return -1;
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/*                               Public packing                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Build a per-row Blocked-K layout from a row-major matrix.
 *
 * Layout definition (per row):
 *   For k0 in {0, blk_k, 2*blk_k, ...} copy the contiguous slice
 *   W[r, k0 .. k0 + klen - 1] where klen = min(blk_k, cols - k0)
 * into the destination row buffer. The order of elements within a row is
 * unchanged, merely grouped by contiguous @p blk_k segments.
 *
 * Performance note:
 *  - This packing does not introduce padding; the last block is short.
 *  - Consumers can use @p blk_k as an inner-loop tile when iterating K.
 *
 * Cache interaction:
 *  - If @p cache_path is non-NULL and a readable file exists with exactly
 *    rows*cols floats, the function loads it and returns immediately.
 *  - Otherwise the function computes the layout, attempts to save it
 *    atomically to @p cache_path (if provided), and returns the new buffer.
 *
 * @param W           Pointer to row-major input matrix (rows x cols).
 * @param rows        Number of rows in @p W.
 * @param cols        Number of columns in @p W.
 * @param blk_k       Column block size (>= 1).
 * @param cache_path  Optional filename for cache read/write (may be NULL).
 * @param out         Output descriptor to be filled on success.
 * @return 0 on success (with @p out->data != NULL); -1 on invalid args or OOM.
 */
int ie_layout_build_blockedK(const float *W,
                             size_t rows, size_t cols, size_t blk_k,
                             const char *cache_path,
                             ie_wblocked_desc_t *out) {
  if (!W || !out || rows == 0 || cols == 0 || blk_k == 0) return -1;

  float *buf = NULL;
  if (cache_path && load_cache(cache_path, rows, cols, &buf) == 0) {
    out->rows = rows;
    out->cols = cols;
    out->blk_k = blk_k;
    out->data = buf;
    return 0;
  }

  buf = (float *)malloc(rows * cols * sizeof(float));
  if (!buf) return -1;

  for (size_t r = 0; r < rows; ++r) {
    const float *src_row = W + r * cols;
    float *dst_row = buf + r * cols;
    size_t written = 0;

    for (size_t k0 = 0; k0 < cols; k0 += blk_k) {
      const size_t klen = (k0 + blk_k <= cols) ? blk_k : (cols - k0);
      memcpy(dst_row + written, src_row + k0, klen * sizeof(float));
      written += klen;
    }
  }

  if (cache_path) {
    (void)save_cache_atomic(cache_path, buf, rows, cols);
  }

  out->rows = rows;
  out->cols = cols;
  out->blk_k = blk_k;
  out->data = buf;
  return 0;
}

/**
 * @brief Release a blocked descriptor and reset its fields.
 *
 * Safe to call with NULL. After the call, @p desc fields are zeroed.
 *
 * @param desc Descriptor pointer (may be NULL).
 */
void ie_layout_free(ie_wblocked_desc_t *desc) {
  if (!desc) return;
  free(desc->data);
  desc->data = NULL;
  desc->rows = 0;
  desc->cols = 0;
  desc->blk_k = 0;
}
