/* ============================================================================
 * File: engine/src/io/mmap_tuning.c
 * ============================================================================
 */
/**
 * @file mmap_tuning.c
 * @brief Implementation of portable mmap tuning helpers.
 *
 * See @ref ie_mmap_tuning.h for the public API and behavioral contract.
 */

#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include "ie_mmap_tuning.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
  #include <sys/mman.h>
  #include <sys/syscall.h>
  #include <fcntl.h>
#endif

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Map ie_page_policy to madvise() flag where possible.
 */
static int map_policy_to_madvise(ie_page_policy p) {
#if defined(__linux__)
  switch (p) {
    case IE_PAGE_NORMAL:     return MADV_NORMAL;
    case IE_PAGE_RANDOM:     return MADV_RANDOM;
    case IE_PAGE_SEQUENTIAL: return MADV_SEQUENTIAL;
    case IE_PAGE_WILLNEED:   return MADV_WILLNEED;
    case IE_PAGE_DONTNEED:   return MADV_DONTNEED;
    default:                 return MADV_NORMAL;
  }
#else
  (void)p;
  return -1; /* not supported on this platform */
#endif
}

/**
 * @brief Safe wrapper around madvise (best-effort).
 *
 * Returns 0 if unsupported (no-op), -1 on hard misuse errors.
 */
static int best_effort_madvise(void *addr, size_t length, int advice) {
  if (!addr || length == 0) { errno = EINVAL; return -1; }
#if defined(__linux__)
  if (advice < 0) return 0; /* treat unsupported advice as no-op */
  if (madvise(addr, length, advice) != 0) {
    /* EINVAL for unknown flags/unaligned size: treat as best-effort success. */
    if (errno == EINVAL || errno == ENOSYS || errno == ENOTSUP) return 0;
    return -1;
  }
#else
  (void)addr; (void)length; (void)advice;
#endif
  return 0;
}

/* ----------------------------- public functions --------------------------- */

int ie_mmap_advise(void *addr, size_t length, ie_page_policy policy) {
  const int adv = map_policy_to_madvise(policy);
  return best_effort_madvise(addr, length, adv);
}

int ie_mmap_fadvise(int fd, off_t offset, off_t length, ie_page_policy policy) {
  if (fd < 0) { errno = EBADF; return -1; }
#if defined(__linux__) || defined(_POSIX_VERSION)
  int adv = 0;
  switch (policy) {
    case IE_PAGE_NORMAL:     adv = 0 /* POSIX_FADV_NORMAL */; break;
    case IE_PAGE_RANDOM:     adv = 1 /* POSIX_FADV_RANDOM */; break;
    case IE_PAGE_SEQUENTIAL: adv = 2 /* POSIX_FADV_SEQUENTIAL */; break;
    case IE_PAGE_WILLNEED:   adv = 3 /* POSIX_FADV_WILLNEED */; break;
    case IE_PAGE_DONTNEED:   adv = 4 /* POSIX_FADV_DONTNEED */; break;
    default:                 adv = 0; break;
  }
  /* Use macros if they exist; fall back to literal values otherwise. */
  #if defined(POSIX_FADV_NORMAL)
    if (policy == IE_PAGE_NORMAL)     adv = POSIX_FADV_NORMAL;
  #endif
  #if defined(POSIX_FADV_RANDOM)
    if (policy == IE_PAGE_RANDOM)     adv = POSIX_FADV_RANDOM;
  #endif
  #if defined(POSIX_FADV_SEQUENTIAL)
    if (policy == IE_PAGE_SEQUENTIAL) adv = POSIX_FADV_SEQUENTIAL;
  #endif
  #if defined(POSIX_FADV_WILLNEED)
    if (policy == IE_PAGE_WILLNEED)   adv = POSIX_FADV_WILLNEED;
  #endif
  #if defined(POSIX_FADV_DONTNEED)
    if (policy == IE_PAGE_DONTNEED)   adv = POSIX_FADV_DONTNEED;
  #endif

  /* posix_fadvise returns an errno-style code, not -1. */
  #if defined(__GLIBC__) || defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__)
    int rc = posix_fadvise(fd, offset, length, adv);
    if (rc == EINVAL || rc == ENOSYS || rc == ENOTSUP) return 0; /* best-effort */
    if (rc != 0) { errno = rc; return -1; }
    return 0;
  #else
    (void)offset; (void)length; (void)adv;
    return 0;
  #endif
#else
  (void)fd; (void)offset; (void)length; (void)policy;
  return 0;
#endif
}

int ie_mmap_advise_thp(void *addr, size_t length, int enable) {
#if defined(__linux__)
  /* Prefer explicit THP advice if available; otherwise no-op. */
  #if defined(MADV_HUGEPAGE) && defined(MADV_NOHUGEPAGE)
    return best_effort_madvise(addr, length, enable ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  #else
    (void)addr; (void)length; (void)enable;
    return 0;
  #endif
#else
  (void)addr; (void)length; (void)enable;
  return 0;
#endif
}

int ie_mmap_readahead(int fd, off_t offset, size_t length) {
  if (fd < 0) { errno = EBADF; return -1; }
#if defined(__linux__)
  /* Try the Linux readahead(2) syscall first. */
  #ifdef SYS_readahead
    /* NB: readahead returns 0 on success, or -1 with errno on failure. */
    ssize_t rc = (ssize_t)syscall(SYS_readahead, fd, (off64_t)offset, (size_t)length);
    if (rc == 0) return 0;
    if (errno == ENOSYS || errno == EINVAL || errno == EPERM) {
      /* Fall back to posix_fadvise WILLNEED. */
      (void)ie_mmap_fadvise(fd, offset, (off_t)length, IE_PAGE_WILLNEED);
      return 0;
    }
    return -1;
  #else
    /* No SYS_readahead: fall back to posix_fadvise WILLNEED. */
    (void)ie_mmap_fadvise(fd, offset, (off_t)length, IE_PAGE_WILLNEED);
    return 0;
  #endif
#else
  /* Non-Linux: best-effort posix_fadvise if available. */
  (void)ie_mmap_fadvise(fd, offset, (off_t)length, IE_PAGE_WILLNEED);
  return 0;
#endif
}

size_t ie_mmap_kernel_hugepagesize(void) {
#if defined(__linux__)
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kbytes = 0;
  while (fgets(line, (int)sizeof(line), f)) {
    /* Example: "Hugepagesize:       2048 kB" */
    if (strncmp(line, "Hugepagesize:", 13) == 0) {
      unsigned long v = 0;
      if (sscanf(line + 13, "%lu kB", &v) == 1) {
        kbytes = (size_t)v;
        break;
      }
    }
  }
  fclose(f);
  return kbytes ? (kbytes * 1024u) : 0u;
#else
  return 0u;
#endif
}

int ie_mmap_prefault_touch_ro(const void *addr, size_t length, size_t stride,
                              uint64_t *checksum) {
  if (!addr || length == 0) { errno = EINVAL; return -1; }
  const size_t step = (stride == 0 ? 64u : stride);
  const volatile unsigned char *p = (const volatile unsigned char *)addr;
  uint64_t acc = 0u;
  /* Bound the loop to [0, length). Use volatile loads to keep it intact. */
  for (size_t i = 0; i < length; i += step) {
    acc += p[i];
  }
  /* Touch the last byte to ensure the tail page is demand-paged too. */
  acc += p[length - 1u];
  if (checksum) *checksum = acc;
  return 0;
}
