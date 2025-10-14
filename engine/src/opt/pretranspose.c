/**
 * @file pretranspose.c
 * @brief Build and cache blocked-K weight layout for GEMV.
 */
#include "ie_layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/**
 * @brief Create a filename for the on-disk cache of a blocked matrix.
 *
 * @param base_path  Path to original weights file (used to namespace cache).
 * @param tag        Short tag like "wxh" or "woh".
 * @param rows       Row count.
 * @param cols       Column count.
 * @param blk_k      Column block size.
 * @param out_buf    Output buffer for the resulting path string.
 * @param out_cap    Capacity of @p out_buf including NUL terminator.
 * @return 0 on success; -1 on failure.
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

/**
 * @brief Try loading a blocked buffer from disk cache.
 *
 * @param path Cache file path.
 * @param rows Rows.
 * @param cols Cols.
 * @param data_out *data malloc'ed on success.
 * @return 0 on success; -1 otherwise.
 */
static int load_cache(const char *path, size_t rows, size_t cols, float **data_out) {
  if (!path) return -1;
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  size_t count = rows * cols;
  float *buf = (float*)malloc(count * sizeof(float));
  if (!buf) { fclose(f); return -1; }
  size_t got = fread(buf, sizeof(float), count, f);
  fclose(f);
  if (got != count) { free(buf); return -1; }
  *data_out = buf;
  return 0;
}

/**
 * @brief Save a blocked buffer to disk cache (atomic via temp file).
 *
 * @param path Destination file path.
 * @param data Buffer pointer.
 * @param rows Rows.
 * @param cols Cols.
 * @return 0 on success; -1 otherwise.
 */
static int save_cache_atomic(const char *path, const float *data, size_t rows, size_t cols) {
  if (!path) return -1;
  char tmp[4096];
  int n = snprintf(tmp, sizeof(tmp), "%s.tmp", path);
  if (n <= 0 || (size_t)n >= sizeof(tmp)) return -1;

  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;
  size_t count = rows * cols;
  size_t put = fwrite(data, sizeof(float), count, f);
  if (put != count) { fclose(f); remove(tmp); return -1; }
  if (fclose(f) != 0) { remove(tmp); return -1; }

  if (rename(tmp, path) != 0) { remove(tmp); return -1; }
  return 0;
}

/**
 * @brief Build blocked-K layout from row-major W (rows x cols).
 *
 * Layout: for each row r, columns are stored in contiguous blocks of size blk_k:
 *   [W[r,0..blk_k-1] | W[r,blk_k..2*blk_k-1] | ...]
 * The last block may be short if cols is not divisible by blk_k.
 *
 * @param W           Pointer to row-major matrix (rows x cols).
 * @param rows        Row count.
 * @param cols        Column count.
 * @param blk_k       Column block size (>=1).
 * @param cache_path  Optional cache file to load/save; may be NULL.
 * @param out         Output descriptor (allocated here).
 * @return 0 on success; -1 on failure.
 */
int ie_layout_build_blockedK(const float *W,
                             size_t rows, size_t cols, size_t blk_k,
                             const char *cache_path,
                             ie_wblocked_desc_t *out) {
  if (!W || !out || rows == 0 || cols == 0 || blk_k == 0) return -1;

  float *buf = NULL;
  if (load_cache(cache_path, rows, cols, &buf) == 0) {
    out->rows = rows; out->cols = cols; out->blk_k = blk_k; out->data = buf;
    return 0;
  }

  buf = (float*)malloc(rows * cols * sizeof(float));
  if (!buf) return -1;

  for (size_t r = 0; r < rows; ++r) {
    const float *src_row = W + r * cols;
    float *dst_row = buf + r * cols;
    size_t written = 0;
    for (size_t k0 = 0; k0 < cols; k0 += blk_k) {
      size_t klen = (k0 + blk_k <= cols) ? blk_k : (cols - k0);
      memcpy(dst_row + written, src_row + k0, klen * sizeof(float));
      written += klen;
    }
  }

  if (cache_path) (void)save_cache_atomic(cache_path, buf, rows, cols);

  out->rows = rows; out->cols = cols; out->blk_k = blk_k; out->data = buf;
  return 0;
}

/**
 * @brief Free a blocked descriptor and its buffer.
 *
 * @param desc Descriptor pointer (may be NULL).
 */
void ie_layout_free(ie_wblocked_desc_t *desc) {
  if (!desc) return;
  free(desc->data);
  desc->data = NULL;
  desc->rows = desc->cols = desc->blk_k = 0;
}
