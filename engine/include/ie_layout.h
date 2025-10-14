/**
 * @file ie_layout.h
 * @brief Helpers for pretransposing and caching weight matrices in blocked-K layout.
 */
#ifndef IE_LAYOUT_H_
#define IE_LAYOUT_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Descriptor for a blocked-K layout buffer. */
typedef struct ie_wblocked_desc {
  size_t rows;       /**< Number of rows (R). */
  size_t cols;       /**< Number of columns (K). */
  size_t blk_k;      /**< Column block size (BK). */
  float *data;       /**< Pointer to blocked buffer (size rows*cols). */
} ie_wblocked_desc_t;

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
 * @return 0 on success; -1 if arguments are invalid or buffer is too small.
 */
int ie_layout_cache_name(const char *base_path, const char *tag,
                         size_t rows, size_t cols, size_t blk_k,
                         char *out_buf, size_t out_cap);

/**
 * @brief Allocate and build a blocked-K copy of row-major @p W.
 *
 * Memory is allocated with malloc() and must be freed by the caller. If a cache
 * file exists at @p cache_path, this function attempts to load it; otherwise it
 * constructs the blocked layout and saves the cache atomically.
 *
 * @param W           Pointer to source in row-major (rows x cols).
 * @param rows        Row count.
 * @param cols        Column count.
 * @param blk_k       Column block size (e.g., 64 or 128). Must divide cols or last block is short.
 * @param cache_path  Optional cache file path (may be NULL to skip disk cache).
 * @param out         Output descriptor (data pointer, rows/cols/blk_k).
 * @return 0 on success; -1 on failure.
 */
int ie_layout_build_blockedK(const float *W,
                             size_t rows, size_t cols, size_t blk_k,
                             const char *cache_path,
                             ie_wblocked_desc_t *out);

/**
 * @brief Free resources owned by a blocked descriptor.
 *
 * @param desc Descriptor to free (NULL allowed).
 */
void ie_layout_free(ie_wblocked_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif /* IE_LAYOUT_H_ */
