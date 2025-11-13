/* ============================================================================
 * File: engine/include/ie_mmap_tuning.h
 * ============================================================================
 */
/**
 * @file ie_mmap_tuning.h
 * @brief Portable helpers for mmap tuning (madvise/fadvise, THP/HugeTLB, readahead).
 *
 * This module centralizes optional OS hints related to mapped model files:
 * - Page-in patterns via `madvise()` (sequential/random/willneed/dontneed)
 * - File-descriptor cache hints via `posix_fadvise()`
 * - Transparent Huge Pages (THP) advice for a mapped region
 * - Explicit readahead priming of the page cache
 *
 * All functions are **best-effort**:
 *  - They succeed with return code `0` if the hint was accepted **or**
 *    if the platform lacks support (non-fatal).
 *  - They return `-1` only on hard errors (e.g., invalid pointers/FD).
 *
 * The implementation targets Linux primarily, but it compiles on POSIX
 * systems. On non-Linux platforms, some calls devolve into harmless no-ops.
 */
#ifndef IE_MMAP_TUNING_H
#define IE_MMAP_TUNING_H

#include <stddef.h>   /* size_t */
#include <stdint.h>   /* uint64_t */
#include <sys/types.h>/* off_t */

#ifdef __cplusplus
extern "C" {
#endif

/** @enum ie_page_policy
 *  @brief High-level policy for memory access hints.
 */
typedef enum ie_page_policy {
  IE_PAGE_NORMAL = 0,     /**< Default kernel policy. */
  IE_PAGE_RANDOM = 1,     /**< Random access is expected. */
  IE_PAGE_SEQUENTIAL = 2, /**< Sequential forward access is expected. */
  IE_PAGE_WILLNEED = 3,   /**< Prefetch into page cache if possible. */
  IE_PAGE_DONTNEED = 4    /**< Page cache may be dropped soon. */
} ie_page_policy;

/**
 * @brief Apply a memory access policy to a mapped region (best-effort).
 *
 * Internally attempts `madvise()` with the nearest matching advice
 * (`MADV_NORMAL`, `MADV_RANDOM`, `MADV_SEQUENTIAL`, `MADV_WILLNEED`,
 * `MADV_DONTNEED`). If `madvise()` is unavailable, the call becomes a no-op.
 *
 * @param addr   Base address of the mapped region (page-aligned recommended).
 * @param length Length in bytes of the region (rounded up by the kernel).
 * @param policy Desired policy.
 * @return 0 on success or harmless no-op, -1 on hard error (errno set).
 */
int ie_mmap_advise(void *addr, size_t length, ie_page_policy policy);

/**
 * @brief Provide file-cache hints for a file descriptor (best-effort).
 *
 * Internally attempts `posix_fadvise()` with `POSIX_FADV_*` equivalents.
 * Not all `ie_page_policy` values have a 1:1 mapping; unsupported ones
 * degrade to a reasonable fallback or a no-op.
 *
 * @param fd     Open file descriptor for the mapped file.
 * @param offset Starting byte offset within the file.
 * @param length Number of bytes to advise (0 may mean "to EOF" on Linux).
 * @param policy Policy to hint (NORMAL/SEQUENTIAL/RANDOM/WILLNEED/DONTNEED).
 * @return 0 on success or harmless no-op, -1 on hard error (errno set).
 */
int ie_mmap_fadvise(int fd, off_t offset, off_t length, ie_page_policy policy);

/**
 * @brief Suggest Transparent Huge Pages (THP) for a mapped region (best-effort).
 *
 * Attempts `madvise(addr, len, MADV_HUGEPAGE)` when available. If `enable`
 * is `0`, attempts `MADV_NOHUGEPAGE`. If the kernel lacks these flags, this
 * function becomes a no-op and still returns 0.
 *
 * @param addr   Start of the region.
 * @param length Size of the region in bytes.
 * @param enable Non-zero to request THP; zero to explicitly disable THP.
 * @return 0 on success or harmless no-op, -1 on hard error (errno set).
 */
int ie_mmap_advise_thp(void *addr, size_t length, int enable);

/**
 * @brief Try to pre-populate page cache for a file range (best-effort).
 *
 * On Linux, first tries the `readahead(2)` system call; if unavailable,
 * falls back to `posix_fadvise(..., POSIX_FADV_WILLNEED)` as a weaker hint.
 *
 * @param fd     Open file descriptor.
 * @param offset Start offset in bytes.
 * @param length Length in bytes; use a reasonable cap (e.g., mapped size).
 * @return 0 on success or harmless no-op, -1 on hard error (errno set).
 */
int ie_mmap_readahead(int fd, off_t offset, size_t length);

/**
 * @brief Read the kernel HugeTLB default page size in bytes, if known.
 *
 * Parses `/proc/meminfo` line `Hugepagesize:` on Linux. Returns 0 if unknown.
 *
 * @return HugeTLB page size in bytes, or 0 if not detected.
 */
size_t ie_mmap_kernel_hugepagesize(void);

/**
 * @brief Quick in-place page touching to prefault a region into RSS.
 *
 * Strides by cache-line (default 64 bytes) to force-demand paging in a
 * bounded manner. Safe for read-only mappings. This is independent from
 * readahead and can be used after `mmap()` to ensure residency.
 *
 * @param addr     Base pointer of the mapped region.
 * @param length   Number of bytes to touch.
 * @param stride   Stride in bytes (use 64 or 128). Zero selects 64.
 * @param checksum Optional out pointer to receive a lightweight checksum
 *                 (may be NULL). Helps compilers keep the loop intact.
 * @return 0 on success; -1 if inputs are invalid.
 */
int ie_mmap_prefault_touch_ro(const void *addr, size_t length, size_t stride,
                              uint64_t *checksum);

#ifdef __cplusplus
}
#endif
#endif /* IE_MMAP_TUNING_H */
