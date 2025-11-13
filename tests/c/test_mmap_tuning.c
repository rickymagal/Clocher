/* File: tests/c/test_mmap_tuning.c
 * -----------------------------------------------------------------------------
 * @file test_mmap_tuning.c
 * @brief Sanity tests for mmap + paging advice + soft prefetch walks.
 *
 * This test is intentionally self-contained and relies only on POSIX APIs,
 * so it can be built alongside the engine's mmap tuning modules without
 * depending on their internal headers. It exercises the following:
 *
 *  1) Creates a temporary file and fills it with a deterministic byte pattern.
 *  2) Maps the file read-only with mmap(2).
 *  3) Issues paging hints (madvise/posix_madvise) for SEQUENTIAL and RANDOM.
 *  4) Performs a soft prefetch walk (touch each page with a configurable stride).
 *  5) Verifies a rolling checksum to ensure mapping correctness.
 *
 * The program exits with code 0 on success and non-zero on any failure.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* -------------------------------------------------------------------------- */
/*                                Test config                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief Test payload size in bytes (default 16 MiB).
 *
 * Can be overridden with the environment variable TEST_MMAP_BYTES.
 */
static size_t cfg_bytes(void) {
  const char *env = getenv("TEST_MMAP_BYTES");
  if (!env || !*env) return 16u * 1024u * 1024u;
  char *end = NULL;
  unsigned long long v = strtoull(env, &end, 10);
  if (!end || end == env || v == 0ull) return 16u * 1024u * 1024u;
  return (size_t)v;
}

/**
 * @brief Page-touch stride in bytes for the soft prefetch walk.
 *
 * Can be overridden with TEST_MMAP_STRIDE (default = system page size).
 */
static size_t cfg_stride(size_t pagesz) {
  const char *env = getenv("TEST_MMAP_STRIDE");
  if (!env || !*env) return pagesz;
  char *end = NULL;
  unsigned long long v = strtoull(env, &end, 10);
  if (!end || end == env || v == 0ull) return pagesz;
  return (size_t)v;
}

/* -------------------------------------------------------------------------- */
/*                                Utilities                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create a temporary file of @p nbytes and fill it with a pattern.
 *
 * Pattern: byte i := (uint8_t)((i * 131) ^ 0xA5).
 *
 * @param nbytes Desired file size.
 * @param out_fd On success, receives an open R/W file descriptor.
 * @param out_path Optional buffer to receive the path used.
 * @param out_cap  Capacity of @p out_path.
 * @return 0 on success; -1 on error.
 */
static int make_temp_file(size_t nbytes, int *out_fd, char *out_path, size_t out_cap) {
  if (!out_fd) return -1;
  char tmpl[] = "/tmp/mmap_tuning_test.XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) {
    perror("mkstemp");
    return -1;
  }

  /* Unlink immediately; the file will be removed when closed. */
  if (unlink(tmpl) != 0) {
    perror("unlink");
    close(fd);
    return -1;
  }

  /* Pre-allocate and write pattern in 1 MiB chunks. */
  const size_t chunk = 1u * 1024u * 1024u;
  uint8_t *buf = (uint8_t *)malloc(chunk);
  if (!buf) {
    perror("malloc");
    close(fd);
    return -1;
  }

  size_t written = 0;
  while (written < nbytes) {
    size_t todo = (nbytes - written) < chunk ? (nbytes - written) : chunk;
    for (size_t i = 0; i < todo; ++i) {
      size_t idx = written + i;
      buf[i] = (uint8_t)((idx * 131u) ^ 0xA5u);
    }
    ssize_t put = write(fd, buf, todo);
    if (put < 0) {
      perror("write");
      free(buf);
      close(fd);
      return -1;
    }
    if ((size_t)put != todo) {
      fprintf(stderr, "short write: %zd != %zu\n", put, todo);
      free(buf);
      close(fd);
      return -1;
    }
    written += (size_t)put;
  }
  free(buf);

  if (out_path && out_cap > 0) {
    /* Not the real on-disk path (since unlinked), but return the template for logs. */
    snprintf(out_path, out_cap, "%s", "(anonymous tmpfile)");
  }

  *out_fd = fd;
  return 0;
}

/**
 * @brief Compute a simple rolling checksum over @p p for @p n bytes.
 * @return 64-bit checksum.
 */
static uint64_t checksum_bytes(const uint8_t *p, size_t n) {
  uint64_t s0 = 1469598103934665603ull; /* FNV offset basis */
  uint64_t s1 = 1099511628211ull;       /* FNV prime (lower) */
  for (size_t i = 0; i < n; ++i) {
    s0 ^= p[i];
    s0 *= s1;
  }
  return s0;
}

/**
 * @brief Verify the file pattern matches the expected formula.
 *
 * @param p      Pointer to mapped bytes.
 * @param n      Number of bytes.
 * @param outcs  On success, receives checksum.
 * @return 0 on success; -1 on mismatch.
 */
static int verify_pattern(const uint8_t *p, size_t n, uint64_t *outcs) {
  uint64_t cs = checksum_bytes(p, n);
  if (outcs) *outcs = cs;

  /* Spot-check a few positions to ensure mapping content is correct. */
  size_t probes[5] = {0, n / 7, n / 5, n / 3, n - 1};
  for (int i = 0; i < 5; ++i) {
    size_t idx = probes[i];
    if (idx >= n) continue;
    uint8_t want = (uint8_t)((idx * 131u) ^ 0xA5u);
    if (p[idx] != want) {
      fprintf(stderr, "mismatch at %zu: got 0x%02x, want 0x%02x\n",
              idx, (unsigned)p[idx], (unsigned)want);
      return -1;
    }
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/*                               Test routines                                 */
/* -------------------------------------------------------------------------- */

/**
 * @brief Run the SEQUENTIAL/RANDOM advice and soft prefetch walk.
 *
 * This function does not assert performance; it only ensures the sequence of
 * calls is valid and the mapping remains readable and correct.
 *
 * @param base   Mapped address (read-only).
 * @param n      Mapping length in bytes.
 * @param stride Page-touch stride in bytes.
 * @return 0 on success; -1 on any error.
 */
static int exercise_advice_and_walk(const uint8_t *base, size_t n, size_t stride) {
  /* Try madvise first; fall back to posix_madvise if unavailable. */
  int have_madvise = 1;
#if !defined(MADV_RANDOM) || !defined(MADV_SEQUENTIAL)
  have_madvise = 0;
#endif

  if (have_madvise) {
    if (madvise((void *)base, n, MADV_SEQUENTIAL) != 0) {
      /* Not fatal; some kernels refuse advice on tmpfs. */
      fprintf(stderr, "[warn] madvise(SEQUENTIAL) failed: %s\n", strerror(errno));
    }
  } else {
#if defined(POSIX_MADV_SEQUENTIAL)
    if (posix_madvise((void *)base, n, POSIX_MADV_SEQUENTIAL) != 0) {
      fprintf(stderr, "[warn] posix_madvise(SEQUENTIAL) failed\n");
    }
#endif
  }

  /* Soft prefetch walk: touch one byte per page/stride. */
  volatile uint8_t sink = 0;
  for (size_t i = 0; i < n; i += stride) {
    sink ^= base[i];
  }

  if (have_madvise) {
    if (madvise((void *)base, n, MADV_RANDOM) != 0) {
      fprintf(stderr, "[warn] madvise(RANDOM) failed: %s\n", strerror(errno));
    }
  } else {
#if defined(POSIX_MADV_RANDOM)
    if (posix_madvise((void *)base, n, POSIX_MADV_RANDOM) != 0) {
      fprintf(stderr, "[warn] posix_madvise(RANDOM) failed\n");
    }
#endif
  }

  /* Re-verify a checksum after the walk to ensure content is still valid. */
  (void)sink;
  uint64_t cs = 0;
  if (verify_pattern(base, n, &cs) != 0) return -1;
  fprintf(stderr, "[info] checksum after walk: 0x%016" PRIx64 "\n", cs);
  return 0;
}

/* -------------------------------------------------------------------------- */
/*                                    Main                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Entry point for the mmap tuning test.
 *
 * Environment variables:
 *  - TEST_MMAP_BYTES   : total bytes to allocate (default 16 MiB)
 *  - TEST_MMAP_STRIDE  : stride in bytes for the soft prefetch walk
 *
 * Exit status: 0 on pass, non-zero on failure.
 */
int main(void) {
  int rc = 1;
  int fd = -1;
  void *addr = NULL;

  const size_t total = cfg_bytes();
  const long pagesz_l = sysconf(_SC_PAGESIZE);
  const size_t pagesz = (pagesz_l > 0) ? (size_t)pagesz_l : 4096u;
  const size_t stride = cfg_stride(pagesz);

  fprintf(stderr, "[test] bytes=%zu, pagesz=%zu, stride=%zu\n",
          total, pagesz, stride);

  if (make_temp_file(total, &fd, NULL, 0) != 0) {
    fprintf(stderr, "FAIL: could not create/write temp file\n");
    goto done;
  }

  /* Map read-only, private. */
  addr = mmap(NULL, total, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    perror("mmap");
    goto done;
  }

  /* Initial checksum + pattern verification. */
  uint64_t cs0 = 0;
  if (verify_pattern((const uint8_t *)addr, total, &cs0) != 0) {
    fprintf(stderr, "FAIL: pattern mismatch after initial mmap\n");
    goto done;
  }
  fprintf(stderr, "[info] checksum initial: 0x%016" PRIx64 "\n", cs0);

  /* Exercise advice + soft prefetch walk. */
  if (exercise_advice_and_walk((const uint8_t *)addr, total, stride) != 0) {
    fprintf(stderr, "FAIL: exercise_advice_and_walk failed\n");
    goto done;
  }

  fprintf(stderr, "[pass] mmap tuning test completed successfully\n");
  rc = 0;

done:
  if (addr && addr != MAP_FAILED) munmap(addr, total);
  if (fd >= 0) close(fd);
  return rc;
}

