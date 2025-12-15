/* ============================================================================
 * File: engine/src/io/loader_mmap.c
 * ============================================================================
 */
/**
 * @file loader_mmap.c
 * @brief Portable file loader with an mmap-first strategy and safe fallbacks.
 *
 * This module provides:
 *  - Read-only memory mapping of files (Linux: `mmap`, others: buffered read).
 *  - Best-effort page advice (WILLNEED, RANDOM, SEQUENTIAL, HUGEPAGE) through
 *    the centralized mmap-tuning helpers.
 *  - Optional readahead priming and a page-touch routine for warm-up.
 *
 * Additional utilities for deduplicated weights:
 *  - Multi-file mapping helper that opens and maps N files in one call.
 *  - Best-effort prefault policy applied per file (sequential vs willneed).
 *
 * The API is intentionally minimal and self-contained; it does not depend on
 * engine-specific structures. Callers can:
 *   - use ::ie_mmap_open to obtain a RO view of a file,
 *   - optionally call ::ie_mmap_apply_advise to refine access patterns,
 *   - optionally call ::ie_mmap_touch to prefault pages,
 *   - finally call ::ie_mmap_close to release resources.
 */

#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

#include "ie_mmap_tuning.h" /* centralized best-effort madvise/fadvise/readahead */

/* ------------------------------- Public API -------------------------------- */
/**
 * @def IE_MMAP_F_READONLY
 * @brief Map file as read-only (default); always used on non-Linux fallback.
 */
#define IE_MMAP_F_READONLY   0x01
/**
 * @def IE_MMAP_F_POPULATE
 * @brief Request early population (Linux `MAP_POPULATE`) + fadvise/willneed.
 */
#define IE_MMAP_F_POPULATE   0x02
/**
 * @def IE_MMAP_F_HUGEPAGE
 * @brief Best-effort request for THP (`MADV_HUGEPAGE`) after mapping.
 */
#define IE_MMAP_F_HUGEPAGE   0x04

/**
 * @enum ie_mmap_advise_t
 * @brief Optional readahead/access pattern hint for the mapped region.
 */
typedef enum ie_mmap_advise {
  IE_MMAP_ADVISE_NONE = 0,   /**< No additional advice. */
  IE_MMAP_ADVISE_WILLNEED,   /**< Expect imminent reads. */
  IE_MMAP_ADVISE_RANDOM,     /**< Random access pattern. */
  IE_MMAP_ADVISE_SEQUENTIAL  /**< Mostly sequential reads. */
} ie_mmap_advise_t;

/**
 * @brief Map a file into memory (read-only) or fall back to a malloc buffer.
 *
 * On Linux:
 *  - Opens @p path with O_RDONLY; mmaps it with `PROT_READ`.
 *  - When @p flags includes ::IE_MMAP_F_POPULATE, it:
 *      * issues `posix_fadvise(..., WILLNEED)` (best-effort),
 *      * enables `MAP_POPULATE` if available,
 *      * calls `readahead(2)` as a further best-effort hint.
 *  - When @p flags includes ::IE_MMAP_F_HUGEPAGE, applies `MADV_HUGEPAGE`
 *    through the mmap-tuning helper after a successful `mmap`.
 *
 * On non-Linux platforms or when `mmap` fails:
 *  - Reads the whole file into a heap buffer (64-byte aligned).
 *  - Sets @p *out_is_fallback = 1 so callers may `free()` it via
 *    ::ie_mmap_close.
 *
 * @param path            File path to map.
 * @param out_addr        Output pointer to mapped or allocated region.
 * @param out_len         Output byte length of the region.
 * @param out_is_fallback Output flag set to 1 when malloc fallback is used.
 * @param flags           Bitwise OR of IE_MMAP_F_* flags.
 * @return 0 on success; negative errno-like value on failure.
 */
int ie_mmap_open(const char *path,
                 const void **out_addr, size_t *out_len,
                 int *out_is_fallback,
                 int flags);

/**
 * @brief Apply access pattern advice to a mapped region (best-effort).
 *
 * Internally forwards to `ie_mmap_advise()` from the mmap-tuning module.
 * Non-Linux platforms simply ignore the request and return 0.
 *
 * @param addr   Base address (as returned by ::ie_mmap_open).
 * @param len    Region length in bytes.
 * @param advise Advice enum value.
 * @return 0 on success (or unsupported); negative value on failure.
 */
int ie_mmap_apply_advise(const void *addr, size_t len, ie_mmap_advise_t advise);

/**
 * @brief Touch pages across a region with the given stride to enforce warm-up.
 *
 * The routine performs a volatile read at fixed strides to prevent the compiler
 * from optimizing away the loop. It forwards to
 * ::ie_mmap_prefault_touch_ro for consistency with other modules.
 *
 * @param addr     Base address.
 * @param len      Region length in bytes.
 * @param stride   Stride in bytes (use >= 1; 4096 is a reasonable default).
 * @param verify   When non-zero, returns a checksum to keep side effects.
 * @param out_sum  Optional output for the checksum (may be NULL).
 * @return 0 on success; negative value on invalid arguments.
 */
int ie_mmap_touch(const void *addr, size_t len, size_t stride,
                  int verify, uint64_t *out_sum);

/**
 * @brief Unmap or free a region previously returned by ::ie_mmap_open.
 *
 * @param addr          Base address to release.
 * @param len           Region length in bytes.
 * @param was_fallback  1 if malloc fallback was used, 0 for true mmap path.
 * @return 0 on success; negative value on failure.
 */
int ie_mmap_close(const void *addr, size_t len, int was_fallback);

/**
 * @brief Map multiple files (read-only) in one call.
 *
 * This helper is intended for deduplicated weights, where multiple blobs
 * (defaults/exceptions/masks) must be available simultaneously and should
 * share a consistent paging policy.
 *
 * Semantics:
 * - Each file is mapped using ::ie_mmap_open with @p flags.
 * - On partial failure, already-mapped files are closed and the function returns
 *   a negative error code.
 *
 * Memory ownership:
 * - On success, the caller owns the returned arrays and must free them:
 *     - `addrs[i]` and `lens[i]` must be released by calling ::ie_mmap_close
 *     - `is_fallback[i]` is used as the third parameter to ::ie_mmap_close
 *     - The arrays `addrs`, `lens`, `is_fallback` are allocated and must be free'd
 *
 * @param paths        Array of file paths of length @p n.
 * @param n            Number of files.
 * @param out_addrs    Output array of mapped addresses (allocated).
 * @param out_lens     Output array of mapped lengths (allocated).
 * @param out_fallback Output array of fallback flags (allocated).
 * @param flags        Flags passed to ::ie_mmap_open for each file.
 * @param advise       Optional advice applied to each mapping after opening.
 * @return 0 on success; negative errno-like value on failure.
 */
int ie_mmap_open_many(const char *const *paths, size_t n,
                      const void ***out_addrs, size_t **out_lens, int **out_fallback,
                      int flags, ie_mmap_advise_t advise);

/**
 * @brief Close a set of mappings created by ::ie_mmap_open_many.
 *
 * This helper is the symmetric teardown for ::ie_mmap_open_many.
 *
 * @param addrs      Array of addresses.
 * @param lens       Array of lengths.
 * @param fallback   Array of fallback flags.
 * @param n          Number of files.
 */
void ie_mmap_close_many(const void **addrs, const size_t *lens, const int *fallback, size_t n);

/* ------------------------------ Implementation ---------------------------- */

int ie_mmap_open(const char *path,
                 const void **out_addr, size_t *out_len,
                 int *out_is_fallback,
                 int flags) {
  if (!path || !out_addr || !out_len || !out_is_fallback) return -22;

#if defined(__linux__)
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return -errno;

  struct stat st;
  if (fstat(fd, &st) != 0) { int e = -errno; close(fd); return e; }
  if (st.st_size <= 0) { close(fd); return -5; }

  /* If POPULATE requested, prime the cache best-effort before mapping. */
  if (flags & IE_MMAP_F_POPULATE) {
    (void)ie_mmap_fadvise(fd, 0, (off_t)st.st_size, IE_PAGE_WILLNEED);
    (void)ie_mmap_readahead(fd, 0, (size_t)st.st_size);
  }

  int mmap_flags = MAP_PRIVATE;
  #ifdef MAP_POPULATE
  if (flags & IE_MMAP_F_POPULATE) mmap_flags |= MAP_POPULATE;
  #endif

  void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, fd, 0);
  int saved_errno = errno;
  close(fd);

  if (p == MAP_FAILED) {
    /* Fallback to buffered read. */
    FILE *f = fopen(path, "rb");
    if (!f) return -saved_errno ? -saved_errno : -errno;

    size_t n = (size_t)st.st_size;
    void *buf = NULL;
    /* 64-byte alignment tends to play well with prefetchers. */
    int aerr = posix_memalign(&buf, 64, n ? n : 1u);
    if (aerr != 0 || !buf) { fclose(f); return -12; }

    size_t rd = fread(buf, 1, n, f);
    fclose(f);
    if (rd != n) { free(buf); return -5; }

    *out_addr = buf;
    *out_len = n;
    *out_is_fallback = 1;
    return 0;
  }

  /* Best-effort THP request if flagged. */
  if (flags & IE_MMAP_F_HUGEPAGE) {
    (void)ie_mmap_advise_thp(p, (size_t)st.st_size, /*enable=*/1);
  }

  *out_addr = p;
  *out_len = (size_t)st.st_size;
  *out_is_fallback = 0;
  return 0;

#else
  /* Non-Linux: buffered read only. */
  FILE *f = fopen(path, "rb");
  if (!f) return -errno;

  if (fseek(f, 0, SEEK_END) != 0) { int e = -errno; fclose(f); return e; }
  long sz = ftell(f);
  if (sz < 0) { int e = -errno; fclose(f); return e; }
  if (fseek(f, 0, SEEK_SET) != 0) { int e = -errno; fclose(f); return e; }

  size_t n = (size_t)sz;
  void *buf = NULL;
  int aerr = posix_memalign(&buf, 64, n ? n : 1u);
  if (aerr != 0 || !buf) { fclose(f); return -12; }

  size_t rd = fread(buf, 1, n, f);
  fclose(f);
  if (rd != n) { free(buf); return -5; }

  (void)flags;
  *out_addr = buf;
  *out_len = n;
  *out_is_fallback = 1;
  return 0;
#endif
}

int ie_mmap_apply_advise(const void *addr, size_t len, ie_mmap_advise_t advise) {
  if (!addr || len == 0) return -22;

  ie_page_policy pol = IE_PAGE_NORMAL;
  switch (advise) {
    case IE_MMAP_ADVISE_NONE:       pol = IE_PAGE_NORMAL;     break;
    case IE_MMAP_ADVISE_WILLNEED:   pol = IE_PAGE_WILLNEED;   break;
    case IE_MMAP_ADVISE_RANDOM:     pol = IE_PAGE_RANDOM;     break;
    case IE_MMAP_ADVISE_SEQUENTIAL: pol = IE_PAGE_SEQUENTIAL; break;
    default: return -22;
  }
  return ie_mmap_advise((void*)addr, len, pol);
}

int ie_mmap_touch(const void *addr, size_t len, size_t stride,
                  int verify, uint64_t *out_sum) {
  if (!addr || len == 0) return -22;
  return ie_mmap_prefault_touch_ro(addr, len, stride ? stride : 64u,
                                   verify ? out_sum : NULL);
}

int ie_mmap_close(const void *addr, size_t len, int was_fallback) {
  if (!addr) return 0;
#if defined(__linux__)
  if (!was_fallback) {
    return (munmap((void*)addr, len) == 0) ? 0 : -errno;
  }
#endif
  (void)len;
  free((void*)addr);
  return 0;
}

int ie_mmap_open_many(const char *const *paths, size_t n,
                      const void ***out_addrs, size_t **out_lens, int **out_fallback,
                      int flags, ie_mmap_advise_t advise) {
  if (!paths || n == 0 || !out_addrs || !out_lens || !out_fallback) return -22;

  const void **addrs = (const void **)calloc(n, sizeof(*addrs));
  size_t *lens = (size_t *)calloc(n, sizeof(*lens));
  int *fallback = (int *)calloc(n, sizeof(*fallback));
  if (!addrs || !lens || !fallback) {
    free(addrs); free(lens); free(fallback);
    return -12;
  }

  for (size_t i = 0; i < n; i++) {
    if (!paths[i] || !*paths[i]) {
      ie_mmap_close_many(addrs, lens, fallback, n);
      return -22;
    }

    int rc = ie_mmap_open(paths[i], &addrs[i], &lens[i], &fallback[i], flags);
    if (rc != 0) {
      ie_mmap_close_many(addrs, lens, fallback, n);
      return rc;
    }

    if (advise != IE_MMAP_ADVISE_NONE) {
      (void)ie_mmap_apply_advise(addrs[i], lens[i], advise);
    }
  }

  *out_addrs = addrs;
  *out_lens = lens;
  *out_fallback = fallback;
  return 0;
}

void ie_mmap_close_many(const void **addrs, const size_t *lens, const int *fallback, size_t n) {
  if (!addrs || !lens || !fallback) return;
  for (size_t i = 0; i < n; i++) {
    if (addrs[i]) (void)ie_mmap_close(addrs[i], lens[i], fallback[i]);
  }
  free((void**)addrs);
  free((void*)lens);
  free((void*)fallback);
}

