/* ============================================================================
 * File: engine/src/main_infer.c
 * ============================================================================
 */
/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine (benchmark-friendly, strict-run safe).
 *
 * This file implements the standalone CLI binary used by the benchmark harness.
 * It prints exactly one JSON object to stdout (even on errors) so that scripts can
 * reliably parse results.
 *
 * Timing policy (low-overhead, benchmark-correct):
 *  - All setup is performed outside the timed window:
 *      - CLI parsing, path resolution, model open, engine create, warmup,
 *        tokenizer resolution, prompt file I/O (prompt list is preloaded),
 *        strict-touch preflight (mmap open / cudart dlopen).
 *  - The timed window measures only the steady-state "work loop":
 *      - ie_engine_generate calls (and, when enabled, strict-touch work that
 *        must be attributed to the run by design).
 *
 * Strict-run goals (CPU/CUDA):
 *  - When IE_REQUIRE_MODEL=1, model files must open successfully.
 *  - When IE_VERIFY_TOUCH=1 and IE_BYTES_PER_TOKEN>0, the timed window includes
 *    deterministic, measurable work that cannot be optimized away.
 *  - For CPU, the strict "touch" work is file-backed: it reads from the model
 *    weights mmap (not anonymous memory), so RSS/majflt reflect real paging.
 *  - For CUDA, the strict work uses cudart (if present) to force driver activity
 *    inside the timed window without relying on custom kernels.
 *
 * Text output:
 *  - By default, the CLI prints tokens in JSON for benchmark tooling.
 *  - If --print-text is provided (or IE_PRINT_TEXT=1), the CLI will also decode
 *    the generated token IDs into UTF-8 using the project tokenizer.
 *  - Tokenizer resolution prefers .tiktoken files (when present) and falls back
 *    to tokenizer.json paths.
 *
 * Prompts file behavior (Goal A):
 *  - If --prompts-file is provided and --prompt is NOT provided, the CLI runs
 *    all non-empty lines from the prompts file (newline-delimited) in order.
 *  - This mode is internally treated as "aggregate" mode and still prints
 *    exactly one JSON object (aggregated over all prompts and rounds).
 *  - Empty/whitespace-only lines are ignored.
 *
 * Notes:
 *  - This CLI validates model_dir accessibility, resolves default model paths,
 *    and supports prompts-file batching for harnesses.
 *  - Token output types follow ie_api.h: out_tokens is int[], out_n_tokens is size_t.
 */

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
#endif

#ifndef _XOPEN_SOURCE
#  define _XOPEN_SOURCE 700
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ie_api.h"
#include "ie_io.h"
#include "ie_kv_instrumentation.h"
#include "ie_tokenizer_gptoss.h"
#include "util_logging.h"

#ifndef PATH_MAX
#  define PATH_MAX 4096
#endif

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

#ifndef IE_PREC_FP32
#  define IE_PREC_FP32 "fp32"
#endif
#ifndef IE_PREC_BF16
#  define IE_PREC_BF16 "bf16"
#endif
#ifndef IE_PREC_FP16
#  define IE_PREC_FP16 "fp16"
#endif
#ifndef IE_PREC_INT8W
#  define IE_PREC_INT8W "int8w"
#endif
#ifndef IE_PREC_INT4W
#  define IE_PREC_INT4W "int4w"
#endif
#ifndef IE_PREC_INT4
#  define IE_PREC_INT4 "int4"
#endif

/* -------------------------------------------------------------------------- */
/* Local helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return the smaller of two size_t values.
 * @param a First value.
 * @param b Second value.
 * @return min(a,b).
 */
static size_t min_size(size_t a, size_t b) { return (a < b) ? a : b; }

/**
 * @brief Read a long integer from an environment variable with a default.
 * @param name Environment variable name.
 * @param defv Default value when unset or invalid.
 * @return Parsed value or defv.
 */
/**
 * @brief env_long.
 *
 * @param name See implementation for details.
 * @param defv See implementation for details.
 *
 * @return See implementation for details.
 */
static long env_long(const char *name, long defv) {
  const char *s = getenv(name);
  if (!s || !*s) return defv;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  return (end && *end == '\0') ? v : defv;
}

/**
 * @brief Read a string from an environment variable with a default.
 * @param name Environment variable name.
 * @param defv Default string when unset or empty.
 * @return Environment value or defv.
 */
static const char *env_str(const char *name, const char *defv) {
  const char *s = getenv(name);
  return (s && *s) ? s : defv;
}

/**
 * @brief Case-insensitive ASCII equality.
 * @param a String A.
 * @param b String B.
 * @return 1 when equal (ASCII case-insensitive), 0 otherwise.
 */
/**
 * @brief ascii_ieq.
 *
 * @param a See implementation for details.
 * @param b See implementation for details.
 *
 * @return See implementation for details.
 */
static int ascii_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    unsigned char ca = (unsigned char)*a++;
    unsigned char cb = (unsigned char)*b++;
    if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca + 32);
    if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb + 32);
    if (ca != cb) return 0;
  }
  return (*a == '\0' && *b == '\0');
}

/**
 * @brief Check whether a string begins with a given prefix (case-sensitive).
 * @param s Input string.
 * @param prefix Prefix string.
 * @return 1 when s starts with prefix, 0 otherwise.
 */
/**
 * @brief starts_with.
 *
 * @param s See implementation for details.
 * @param prefix See implementation for details.
 *
 * @return See implementation for details.
 */
static int starts_with(const char *s, const char *prefix) {
  if (!s || !prefix) return 0;
  while (*prefix) {
    if (*s++ != *prefix++) return 0;
  }
  return 1;
}

/**
 * @brief Return monotonic time in seconds.
 * @return Seconds since an unspecified epoch (monotonic).
 */
/**
 * @brief now_sec.
 *
 * @return See implementation for details.
 */
static double now_sec(void) {
  struct timespec ts;
  (void)clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/**
 * @brief Determine whether a filesystem path exists (stat succeeds).
 * @param p Path to test.
 * @return 1 when exists, 0 otherwise.
 */
/**
 * @brief path_exists.
 *
 * @param p See implementation for details.
 *
 * @return See implementation for details.
 */
static int path_exists(const char *p) {
  if (!p || !*p) return 0;
  struct stat st;
  return (stat(p, &st) == 0);
}

/**
 * @brief Determine whether a directory exists and is accessible (R_OK|X_OK).
 * @param p Directory path.
 * @return 1 when accessible directory, 0 otherwise.
 */
/**
 * @brief dir_accessible.
 *
 * @param p See implementation for details.
 *
 * @return See implementation for details.
 */
static int dir_accessible(const char *p) {
  if (!p || !*p) return 0;
  struct stat st;
  if (stat(p, &st) != 0) return 0;
  if (!S_ISDIR(st.st_mode)) return 0;
  return (access(p, R_OK | X_OK) == 0);
}

/**
 * @brief Check if a path is absolute (POSIX).
 * @param p Path.
 * @return 1 when absolute, 0 otherwise.
 */
static int path_is_abs(const char *p) { return (p && p[0] == '/'); }

/**
 * @brief Safe string copy using snprintf.
 * @param dst Destination buffer.
 * @param dstsz Destination capacity.
 * @param src Source string (may be NULL).
 */
/**
 * @brief safe_strcpy.
 *
 * @param dst See implementation for details.
 * @param dstsz See implementation for details.
 * @param src See implementation for details.
 */
static void safe_strcpy(char *dst, size_t dstsz, const char *src) {
  if (!dst || dstsz == 0) return;
  if (!src) {
    dst[0] = '\0';
    return;
  }
  (void)snprintf(dst, dstsz, "%s", src);
}

/**
 * @brief Join directory and filename into a single path.
 * @param out Output buffer.
 * @param outsz Output capacity.
 * @param dir Directory component (may be NULL/empty).
 * @param leaf Leaf component (must be non-NULL).
 */
/**
 * @brief join_path.
 *
 * @param out See implementation for details.
 * @param outsz See implementation for details.
 * @param dir See implementation for details.
 * @param leaf See implementation for details.
 */
static void join_path(char *out, size_t outsz, const char *dir, const char *leaf) {
  if (!out || outsz == 0) return;
  out[0] = '\0';
  if (!leaf) return;

  if (path_is_abs(leaf)) {
    safe_strcpy(out, outsz, leaf);
    return;
  }
  if (!dir || !*dir) {
    safe_strcpy(out, outsz, leaf);
    return;
  }

  const size_t n = strlen(dir);
  if (n > 0 && dir[n - 1] == '/') (void)snprintf(out, outsz, "%s%s", dir, leaf);
  else (void)snprintf(out, outsz, "%s/%s", dir, leaf);
}

/**
 * @brief Canonicalize a directory path with realpath when possible.
 * @param out Output buffer.
 * @param outsz Output capacity.
 * @param dir Input directory path.
 * @return 1 when realpath succeeded, 0 otherwise (out still set to dir).
 */
/**
 * @brief canon_dir.
 *
 * @param out See implementation for details.
 * @param outsz See implementation for details.
 * @param dir See implementation for details.
 *
 * @return See implementation for details.
 */
static int canon_dir(char *out, size_t outsz, const char *dir) {
  if (!out || outsz == 0) return 0;
  out[0] = '\0';
  if (!dir || !*dir) return 0;

  char tmp[PATH_MAX];
  tmp[0] = '\0';
  if (realpath(dir, tmp)) {
    safe_strcpy(out, outsz, tmp);
    return 1;
  }
  safe_strcpy(out, outsz, dir);
  return 0;
}

/**
 * @brief Determine if precision string requests int4 family.
 * @param p Precision label.
 * @return 1 if "int4*" else 0.
 */
/**
 * @brief is_int4_precision.
 *
 * @param p See implementation for details.
 *
 * @return See implementation for details.
 */
static int is_int4_precision(const char *p) {
  if (!p || !*p) return 0;
  return starts_with(p, "int4") ? 1 : 0;
}

/**
 * @brief Resolve model JSON/BIN paths based on model_dir and precision defaults.
 *
 * For int4, prefer model.ie.compat.json when present and model.q4.bin when present.
 *
 * @param model_dir Model directory (may be NULL/empty).
 * @param model_json_opt Optional override for JSON (relative to model_dir unless absolute).
 * @param model_bin_opt Optional override for BIN (relative to model_dir unless absolute).
 * @param precision Precision label (used for int4 defaults).
 * @param out_json Output JSON path.
 * @param out_json_sz Output JSON capacity.
 * @param out_bin Output BIN path.
 * @param out_bin_sz Output BIN capacity.
 */
static void resolve_model_paths(const char *model_dir,
                               const char *model_json_opt,
                               const char *model_bin_opt,
                               const char *precision,
                               char *out_json, size_t out_json_sz,
                               char *out_bin, size_t out_bin_sz) {
  char dir_canon[PATH_MAX];
  dir_canon[0] = '\0';
  (void)canon_dir(dir_canon, sizeof(dir_canon), model_dir);

  const char *base_dir = (dir_canon[0] ? dir_canon : model_dir);
  const int want_int4 = is_int4_precision(precision);

  if (model_json_opt && *model_json_opt) {
    if (path_is_abs(model_json_opt) || !base_dir || !*base_dir) safe_strcpy(out_json, out_json_sz, model_json_opt);
    else join_path(out_json, out_json_sz, base_dir, model_json_opt);
  } else {
    if (base_dir && *base_dir) {
      if (want_int4) {
        char cand[PATH_MAX];
        join_path(cand, sizeof(cand), base_dir, "model.ie.compat.json");
        if (path_exists(cand)) safe_strcpy(out_json, out_json_sz, cand);
        else join_path(out_json, out_json_sz, base_dir, "model.ie.json");
      } else {
        join_path(out_json, out_json_sz, base_dir, "model.ie.json");
      }
    } else {
      safe_strcpy(out_json, out_json_sz, "./model.ie.json");
    }
  }

  if (model_bin_opt && *model_bin_opt) {
    if (path_is_abs(model_bin_opt) || !base_dir || !*base_dir) safe_strcpy(out_bin, out_bin_sz, model_bin_opt);
    else join_path(out_bin, out_bin_sz, base_dir, model_bin_opt);
  } else {
    if (base_dir && *base_dir) {
      if (want_int4) {
        char cand[PATH_MAX];
        join_path(cand, sizeof(cand), base_dir, "model.q4.bin");
        if (path_exists(cand)) safe_strcpy(out_bin, out_bin_sz, cand);
        else join_path(out_bin, out_bin_sz, base_dir, "model.ie.bin");
      } else {
        join_path(out_bin, out_bin_sz, base_dir, "model.ie.bin");
      }
    } else {
      safe_strcpy(out_bin, out_bin_sz, "./model.ie.bin");
    }
  }
}

/**
 * @brief Read ru_maxrss and convert to MiB (Linux: ru_maxrss is KiB).
 * @return RSS peak in MiB (rounded down), or 0 on failure.
 */
/**
 * @brief rss_peak_mib.
 *
 * @return See implementation for details.
 */
static uint32_t rss_peak_mib(void) {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
  if (ru.ru_maxrss <= 0) return 0;
  return (uint32_t)((uint64_t)ru.ru_maxrss / 1024ULL);
}

/**
 * @brief Heap-duplicate a string (portable replacement for strdup).
 * @param s Input string (may be NULL).
 * @return Newly allocated copy, or NULL.
 */
static char *heap_strdup(const char *s) {
  if (!s) return NULL;
  const size_t n = strlen(s) + 1;
  char *p = (char *)malloc(n);
  if (!p) return NULL;
  memcpy(p, s, n);
  return p;
}

/**
 * @brief Check if a string ends with a given suffix (case-sensitive).
 * @param s String (may be NULL).
 * @param suf Suffix (may be NULL).
 * @return 1 when s ends with suf, 0 otherwise.
 */
/**
 * @brief ends_with.
 *
 * @param s See implementation for details.
 * @param suf See implementation for details.
 *
 * @return See implementation for details.
 */
static int ends_with(const char *s, const char *suf) {
  if (!s || !suf) return 0;
  const size_t ns = strlen(s);
  const size_t nf = strlen(suf);
  if (nf > ns) return 0;
  return (memcmp(s + (ns - nf), suf, nf) == 0) ? 1 : 0;
}

/* -------------------------------------------------------------------------- */
/* Strict file-backed model touch (CPU)                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief A file-backed mmap context for deterministic "touch" work.
 */
typedef struct model_mmap_touch {
  int fd;
  void *base;
  size_t size;
  size_t cursor;
} model_mmap_touch_t;

/**
 * @brief Initialize a file-backed read-only mmap for the given path.
 * @param ctx Context to initialize.
 * @param path File path to map.
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief model_touch_open.
 *
 * @param ctx See implementation for details.
 * @param path See implementation for details.
 *
 * @return See implementation for details.
 */
static int model_touch_open(model_mmap_touch_t *ctx, const char *path) {
  if (!ctx) return 1;
  memset(ctx, 0, sizeof(*ctx));
  ctx->fd = -1;

  if (!path || !*path) return 2;

  int fd = open(path, O_RDONLY);
  if (fd < 0) return 3;

  struct stat st;
  if (fstat(fd, &st) != 0) {
    (void)close(fd);
    return 4;
  }
  if (st.st_size <= 0) {
    (void)close(fd);
    return 5;
  }

  size_t sz = (size_t)st.st_size;
  void *base = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) {
    (void)close(fd);
    return 6;
  }

  ctx->fd = fd;
  ctx->base = base;
  ctx->size = sz;
  ctx->cursor = 0;
  return 0;
}

/**
 * @brief Close and unmap a model touch context.
 * @param ctx Context to close (safe to call repeatedly).
 */
/**
 * @brief model_touch_close.
 *
 * @param ctx See implementation for details.
 */
static void model_touch_close(model_mmap_touch_t *ctx) {
  if (!ctx) return;
  if (ctx->base && ctx->base != MAP_FAILED && ctx->size) (void)munmap(ctx->base, ctx->size);
  ctx->base = NULL;
  ctx->size = 0;
  if (ctx->fd >= 0) (void)close(ctx->fd);
  ctx->fd = -1;
  ctx->cursor = 0;
}

/**
 * @brief Deterministically read bytes from the mapped model to force page faults.
 *
 * This reads file-backed pages so that RSS/majflt reflect real paging behavior.
 * The reads are strided and continue from ctx->cursor, wrapping around.
 *
 * @param ctx Open model touch context.
 * @param bytes_to_touch Total bytes to cover (capped to ctx->size).
 * @param stride_bytes Read stride (minimum 1).
 * @param verify_touch If nonzero, perform a small anti-optimization check.
 * @return 0 on success, nonzero on error.
 */
static int model_touch_bytes(model_mmap_touch_t *ctx,
                             size_t bytes_to_touch,
                             size_t stride_bytes,
                             int verify_touch) {
  if (!ctx || !ctx->base || ctx->size == 0) return 1;
  if (bytes_to_touch == 0) return 0;
  if (stride_bytes == 0) stride_bytes = 1;

  const size_t size = ctx->size;
  size_t n = bytes_to_touch;
  if (n > size) n = size;

  volatile const unsigned char *p = (volatile const unsigned char *)ctx->base;
  volatile uint64_t acc = 0;

  size_t start = ctx->cursor % size;

  if (start + n <= size) {
    size_t end = start + n;
    for (size_t off = start; off < end; off += stride_bytes) acc ^= (uint64_t)p[off];
    ctx->cursor = end % size;
  } else {
    size_t first = size - start;
    for (size_t off = start; off < size; off += stride_bytes) acc ^= (uint64_t)p[off];

    size_t rem = n - first;
    for (size_t off = 0; off < rem; off += stride_bytes) acc ^= (uint64_t)p[off];
    ctx->cursor = rem % size;
  }

  if (verify_touch) {
    if (acc == 0x9e3779b97f4a7c15ULL) {
      fprintf(stderr, "touch verify: improbable accumulator value (ignore)\n");
    }
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Minimal cudart dynamic loader (CUDA strict touch)                          */
/* -------------------------------------------------------------------------- */

/**
 * @brief CUDA runtime error code type (opaque int for dynamic loading).
 */
typedef int cudart_err_t;

/**
 * @brief cudart function pointer types used by the strict-touch path.
 */
typedef cudart_err_t (*cudaFree_fn_t)(void *);
typedef cudart_err_t (*cudaMalloc_fn_t)(void **, size_t);
typedef cudart_err_t (*cudaMemcpy_fn_t)(void *, const void *, size_t, int);
typedef cudart_err_t (*cudaMemset_fn_t)(void *, int, size_t);
typedef cudart_err_t (*cudaDeviceSynchronize_fn_t)(void);

/**
 * @brief cudaMemcpyKind values used by the strict-touch path.
 */
enum {
  CUDA_MEMCPY_HOST_TO_DEVICE = 1,
  CUDA_MEMCPY_DEVICE_TO_HOST = 2,
  CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
};

/**
 * @brief Dynamically loaded cudart API table.
 */
typedef struct cudart_api {
  void *handle;
  int ok;
  cudaFree_fn_t cudaFree;
  cudaMalloc_fn_t cudaMalloc;
  cudaMemcpy_fn_t cudaMemcpy;
  cudaMemset_fn_t cudaMemset;
  cudaDeviceSynchronize_fn_t cudaDeviceSynchronize;
} cudart_api_t;

/**
 * @brief Load cudart symbols via dlopen/dlsym (lazy singleton).
 * @return Pointer to API table on success, NULL otherwise.
 */
static cudart_api_t *cudart_get_api(void) {
  static cudart_api_t api;
  static int inited = 0;
  if (inited) return api.ok ? &api : NULL;
  inited = 1;

  memset(&api, 0, sizeof(api));

  api.handle = dlopen("libcudart.so", RTLD_NOW | RTLD_LOCAL);
  if (!api.handle) api.handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
  if (!api.handle) api.handle = dlopen("libcudart.so.11.0", RTLD_NOW | RTLD_LOCAL);

  if (!api.handle) {
    api.ok = 0;
    return NULL;
  }

  { union { void *p; cudaFree_fn_t f; } u; u.p = dlsym(api.handle, "cudaFree"); api.cudaFree = u.f; }
  { union { void *p; cudaMalloc_fn_t f; } u; u.p = dlsym(api.handle, "cudaMalloc"); api.cudaMalloc = u.f; }
  { union { void *p; cudaMemcpy_fn_t f; } u; u.p = dlsym(api.handle, "cudaMemcpy"); api.cudaMemcpy = u.f; }
  { union { void *p; cudaMemset_fn_t f; } u; u.p = dlsym(api.handle, "cudaMemset"); api.cudaMemset = u.f; }
  { union { void *p; cudaDeviceSynchronize_fn_t f; } u; u.p = dlsym(api.handle, "cudaDeviceSynchronize"); api.cudaDeviceSynchronize = u.f; }

  if (!api.cudaFree || !api.cudaMalloc || !api.cudaMemcpy || !api.cudaMemset || !api.cudaDeviceSynchronize) {
    (void)dlclose(api.handle);
    api.handle = NULL;
    api.ok = 0;
    return NULL;
  }

  api.ok = 1;
  return &api;
}

/**
 * @brief Strict CUDA touch: cudaMalloc + repeated cudaMemset + optional D2H copy.
 * @param bytes_per_token Bytes per token (0 disables).
 * @param tokens Number of generated tokens for this call.
 * @param stride_bytes Unused (kept for symmetry with CPU touch).
 * @param verify_touch If nonzero, perform a small D2H copy.
 * @return 0 on success, nonzero on failure.
 */
static int cudart_touch_bytes(size_t bytes_per_token,
                              uint64_t tokens,
                              size_t stride_bytes,
                              int verify_touch) {
  UNUSED(stride_bytes);

  cudart_api_t *api = cudart_get_api();
  if (!api) return -1;
  if (bytes_per_token == 0 || tokens == 0) return 0;

  void *d = NULL;
  cudart_err_t e = api->cudaMalloc(&d, bytes_per_token);
  if (e != 0 || !d) return 1;

  for (uint64_t t = 0; t < tokens; ++t) {
    int pattern = (int)(t & 0xFFu);
    e = api->cudaMemset(d, pattern, bytes_per_token);
    if (e != 0) {
      (void)api->cudaFree(d);
      return 2;
    }
  }

  e = api->cudaDeviceSynchronize();
  if (e != 0) {
    (void)api->cudaFree(d);
    return 3;
  }

  if (verify_touch) {
    uint64_t tmp = 0;
    size_t n = min_size(sizeof(tmp), bytes_per_token);
    e = api->cudaMemcpy(&tmp, d, n, CUDA_MEMCPY_DEVICE_TO_HOST);
    if (e != 0) {
      (void)api->cudaFree(d);
      return 4;
    }
    e = api->cudaDeviceSynchronize();
    if (e != 0) {
      (void)api->cudaFree(d);
      return 5;
    }
    (void)tmp;
  }

  (void)api->cudaFree(d);
  (void)api->cudaDeviceSynchronize();
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Token decoding via project tokenizer                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief JSON-escape and print a string value.
 *
 * @param s Input UTF-8 string (may be NULL).
 */
/**
 * @brief json_print_escaped_string.
 *
 * @param s See implementation for details.
 */
static void json_print_escaped_string(const char *s) {
  fputc('"', stdout);
  if (!s) {
    fputc('"', stdout);
    return;
  }

  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    unsigned char c = *p++;
    if (c == '\\') fputs("\\\\", stdout);
    else if (c == '"') fputs("\\\"", stdout);
    else if (c == '\n') fputs("\\n", stdout);
    else if (c == '\r') fputs("\\r", stdout);
    else if (c == '\t') fputs("\\t", stdout);
    else if (c < 0x20u) {
      fprintf(stdout, "\\u%04x", (unsigned)c);
    } else {
      fputc((int)c, stdout);
    }
  }
  fputc('"', stdout);
}

/**
 * @brief Resolve a tokenizer path under a model directory.
 *
 * Search order:
 *  1) explicit override (tokenizer_opt)
 *  2) <model_dir>/hf/original/tokenizer.tiktoken
 *  3) <model_dir>/tokenizer.tiktoken
 *  4) <model_dir>/original/tokenizer.tiktoken
 *  5) <model_dir>/hf/original/tokenizer.json
 *  6) <model_dir>/tokenizer.json
 *  7) <model_dir>/original/tokenizer.json
 *
 * @param model_dir Model directory.
 * @param tokenizer_opt Optional explicit tokenizer path.
 * @param out Output path.
 * @param outsz Output capacity.
 */
/**
 * @brief resolve_tokenizer_path.
 *
 * @param model_dir See implementation for details.
 * @param tokenizer_opt See implementation for details.
 * @param out See implementation for details.
 * @param outsz See implementation for details.
 */
static void resolve_tokenizer_path(const char *model_dir, const char *tokenizer_opt, char *out, size_t outsz) {
  if (!out || outsz == 0) return;
  out[0] = '\0';

  if (tokenizer_opt && *tokenizer_opt) {
    if (path_is_abs(tokenizer_opt) || !model_dir || !*model_dir) safe_strcpy(out, outsz, tokenizer_opt);
    else join_path(out, outsz, model_dir, tokenizer_opt);
    return;
  }

  if (!model_dir || !*model_dir) return;

  char cand[PATH_MAX];

  join_path(cand, sizeof(cand), model_dir, "hf/original/tokenizer.tiktoken");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "tokenizer.tiktoken");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "original/tokenizer.tiktoken");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "hf/original/tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "original/tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }
}

/**
 * @brief Decode token IDs into UTF-8 text (best-effort).
 *
 * This uses the project tokenizer implementation. If decoding fails or the
 * tokenizer cannot be opened (including vocab mismatch scenarios), a placeholder
 * string "<id><id>..." is emitted so output remains valid and debuggable.
 *
 * @param tok_path Path to tokenizer file (tokenizer.json or tokenizer.tiktoken).
 * @param ids Token IDs (engine output).
 * @param n Number of IDs.
 * @param out_text Output allocated string (caller frees). NULL on failure.
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief decode_tokens_best_effort.
 *
 * @param tok_path See implementation for details.
 * @param ids See implementation for details.
 * @param n See implementation for details.
 * @param out_text See implementation for details.
 *
 * @return See implementation for details.
 */
static int decode_tokens_best_effort(const char *tok_path, const int *ids, size_t n, char **out_text) {
  if (!out_text) return 1;
  *out_text = NULL;

  if (!tok_path || !*tok_path) return 2;

  if (!ids || n == 0) {
    char *z = (char *)malloc(1);
    if (!z) return 3;
    z[0] = '\0';
    *out_text = z;
    return 0;
  }

  uint32_t *u32 = (uint32_t *)malloc(sizeof(uint32_t) * n);
  if (!u32) return 4;

  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    int v = ids[i];
    if (v < 0) continue;
    u32[k++] = (uint32_t)v;
  }

  if (k == 0) {
    free(u32);
    char *z = (char *)malloc(1);
    if (!z) return 5;
    z[0] = '\0';
    *out_text = z;
    return 0;
  }

  ie_tok_gptoss_t *tok = NULL;
  int ts = ie_tok_gptoss_open(tok_path, &tok);
  if (ts == IE_TOK_GPTOSS_OK && tok) {
    size_t need = 0;
    ts = ie_tok_gptoss_decode(tok, u32, (uint32_t)min_size(k, (size_t)0xFFFFFFFFu), NULL, &need);
    if (ts == IE_TOK_GPTOSS_OK && need > 0) {
      char *buf = (char *)malloc(need);
      if (!buf) {
        (void)ie_tok_gptoss_close(tok);
        free(u32);
        return 6;
      }
      size_t outn = need;
      ts = ie_tok_gptoss_decode(tok, u32, (uint32_t)min_size(k, (size_t)0xFFFFFFFFu), buf, &outn);
      if (ts == IE_TOK_GPTOSS_OK) {
        *out_text = buf;
        (void)ie_tok_gptoss_close(tok);
        free(u32);
        return 0;
      }
      free(buf);
    }
    (void)ie_tok_gptoss_close(tok);
  } else {
    fprintf(stderr,
            "warn: tokenizer_open failed for '%s' (status=%d)%s\n",
            tok_path,
            (int)ts,
            ends_with(tok_path, ".tiktoken") ? " (tiktoken decode may be unimplemented)" : "");
  }

  /* Fallback: emit placeholders so the output stays valid and debuggable. */
  {
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) {
      free(u32);
      return 7;
    }
    buf[0] = '\0';

    for (size_t i = 0; i < k; ++i) {
      char tmp[64];
      (void)snprintf(tmp, sizeof(tmp), "<%u>", (unsigned)u32[i]);
      size_t add = strlen(tmp);

      if (len + add + 1 > cap) {
        size_t ncap = cap;
        while (len + add + 1 > ncap) ncap *= 2;
        char *nb = (char *)realloc(buf, ncap);
        if (!nb) { free(buf); free(u32); return 8; }
        buf = nb;
        cap = ncap;
      }

      memcpy(buf + len, tmp, add);
      len += add;
      buf[len] = '\0';
    }

    *out_text = buf;
  }

  free(u32);
  return 0;
}

/* -------------------------------------------------------------------------- */
/* CLI options                                                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Pretranspose modes for optional weight transforms.
 */
typedef enum {
  CLI_PRETX_NONE = 0,
  CLI_PRETX_WOH  = 1,
  CLI_PRETX_WXH  = 2,
  CLI_PRETX_ALL  = 3
} cli_pretranspose_t;

/**
 * @brief Parsed CLI options and derived configuration.
 */
typedef struct cli_extras {
  const char *prompt;
  size_t max_new;
  int threads;
  const char *affinity;
  cli_pretranspose_t pretx;
  const char *device;

  const char *prompts_file;
  int batch;
  const char *prefetch;
  int warmup_tokens;
  int aggregate;
  int rounds;

  /* Optional expected-token verification */
  const char *expected_tokens_file;
  int report_tokens;
  size_t report_tokens_max;

  const char *model_dir;
  const char *model_json;
  const char *model_bin;

  const char *precision_label;
  int precision_from_flag;

  const char *sparsity;
  int sparsity_from_flag;

  /* Text decode options */
  int print_text;
  const char *tokenizer_path;
  const char *trace_ids_path;

  /* Deterministic/debug convenience flags (wired to env vars) */
  int log_tokens;
  int log_every;

  int debug_decode;
  int debug_decode_every;

  int debug_topk;
  int debug_topk_every;

  int greedy;

  int seed_set;
  uint64_t seed;

  int temperature_set;
  double temperature;

  int top_p_set;
  double top_p;

  int top_k_set;
  int top_k;
} cli_extras_t;

/**
 * @brief Print CLI usage to stderr.
 */
/**
 * @brief usage.
 */
static void usage(void) {
  fprintf(stderr,
          "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
          "                        [--threads N]\n"
          "                        [--precision fp32|bf16|fp16|int8w|int4|int4w]\n"
          "                        [--sparsity none|block|auto]\n"
          "                        [--affinity auto|compact|scatter]\n"
          "                        [--pretranspose none|woh|wxh|all]\n"
          "                        [--device auto|cpu|cuda|ze]\n"
          "                        [--model-dir PATH]\n"
          "                        [--model-json PATH] [--model-bin PATH]\n"
          "                        [--prompts-file PATH] [--batch N]\n"
          "                        [--prefetch on|off|auto|N] [--warmup N]\n"
          "                        [--rounds N] [--aggregate]\n"
          "                        [--print-text] [--tokenizer PATH]\n"
          "                        [--trace-ids PATH]\n"
          "                        [--seed N] [--greedy]\n"
          "                        [--temperature F] [--top-p F] [--top-k N]\n"
          "                        [--log-tokens] [--log-every N]\n"
          "                        [--debug-decode] [--debug-decode-every N]\n"
          "                        [--debug-topk K] [--debug-topk-every N]\n");
}

/**
 * @brief Initialize CLI defaults.
 * @param e Options struct to fill.
 */
/**
 * @brief cli_extras_defaults.
 *
 * @param e See implementation for details.
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt = NULL;
  e->max_new = 8;
  e->threads = 0;
  e->affinity = "auto";
  e->pretx = CLI_PRETX_NONE;
  e->device = "auto";

  e->prompts_file = NULL;
  e->batch = 1;
  e->prefetch = "auto";
  e->warmup_tokens = 1;
  e->aggregate = 0;
  e->rounds = 1;


  e->expected_tokens_file = NULL;
  e->report_tokens = 1;
  e->report_tokens_max = 0;

  e->model_dir = NULL;
  e->model_json = NULL;
  e->model_bin = NULL;

  e->precision_label = IE_PREC_FP32;
  e->precision_from_flag = 0;

  e->sparsity = "none";
  e->sparsity_from_flag = 0;

  e->print_text = 0;
  e->tokenizer_path = NULL;
  e->trace_ids_path = NULL;

  e->log_tokens = 0;
  e->log_every = 0;

  e->debug_decode = 0;
  e->debug_decode_every = 0;

  e->debug_topk = 0;
  e->debug_topk_every = 0;

  e->greedy = 0;

  e->seed_set = 0;
  e->seed = 0;

  e->temperature_set = 0;
  e->temperature = 0.0;

  e->top_p_set = 0;
  e->top_p = 0.0;

  e->top_k_set = 0;
  e->top_k = 0;
}

/**
 * @brief Parse a decimal integer safely (fatal on invalid input).
 * @param s Input string.
 * @return Parsed value.
 */
/**
 * @brief safe_atoi.
 *
 * @param s See implementation for details.
 *
 * @return See implementation for details.
 */
static long safe_atoi(const char *s) {
  if (!s || !*s) {
    fprintf(stderr, "error: empty integer\n");
    exit(2);
  }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end) {
    fprintf(stderr, "error: invalid integer: '%s'\n", s);
    exit(2);
  }
  return v;
}

/**
 * @brief Parse an unsigned 64-bit integer safely (fatal on invalid input).
 * @param s Input string.
 * @return Parsed value.
 */
/**
 * @brief safe_atoull.
 *
 * @param s See implementation for details.
 *
 * @return See implementation for details.
 */
static uint64_t safe_atoull(const char *s) {
  if (!s || !*s) {
    fprintf(stderr, "error: empty integer\n");
    exit(2);
  }
  char *end = NULL;
  unsigned long long v = strtoull(s, &end, 10);
  if (end == s || *end) {
    fprintf(stderr, "error: invalid integer: '%s'\n", s);
    exit(2);
  }
  return (uint64_t)v;
}

/**
 * @brief Parse a double safely (fatal on invalid input).
 * @param s Input string.
 * @return Parsed value.
 */
/**
 * @brief safe_atof.
 *
 * @param s See implementation for details.
 *
 * @return See implementation for details.
 */
static double safe_atof(const char *s) {
  if (!s || !*s) {
    fprintf(stderr, "error: empty float\n");
    exit(2);
  }
  char *end = NULL;
  double v = strtod(s, &end);
  if (end == s || *end) {
    fprintf(stderr, "error: invalid float: '%s'\n", s);
    exit(2);
  }
  return v;
}

/**
 * @brief Apply CLI deterministic/debug knobs by setting environment variables.
 *
 * This avoids widening the public API and keeps debug behavior available to
 * the benchmark harness and humans consistently.
 *
 * @param opt Parsed CLI options (may be NULL).
 */
/**
 * @brief apply_cli_debug_env.
 *
 * @param opt See implementation for details.
 */
static void apply_cli_debug_env(const cli_extras_t *opt) {
  if (!opt) return;

  char buf[128];

  if (opt->trace_ids_path && *opt->trace_ids_path) {
    (void)setenv("IE_TRACE_IDS_JSONL", opt->trace_ids_path, 1);
  }

  if (opt->log_tokens) {
    (void)setenv("IE_API_LOG_TOKENS", "1", 1);
  }
  if (opt->log_every > 0) {
    (void)snprintf(buf, sizeof(buf), "%d", opt->log_every);
    (void)setenv("IE_API_LOG_EVERY", buf, 1);
  }

  if (opt->debug_decode) {
    (void)setenv("IE_API_DEBUG_DECODE", "1", 1);
  }
  if (opt->debug_decode_every > 0) {
    (void)snprintf(buf, sizeof(buf), "%d", opt->debug_decode_every);
    (void)setenv("IE_API_DEBUG_DECODE_EVERY", buf, 1);
  }

  if (opt->debug_topk > 0) {
    (void)snprintf(buf, sizeof(buf), "%d", opt->debug_topk);
    (void)setenv("IE_DEBUG_TOPK", buf, 1);
  }
  if (opt->debug_topk_every > 0) {
    (void)snprintf(buf, sizeof(buf), "%d", opt->debug_topk_every);
    (void)setenv("IE_DEBUG_TOPK_EVERY", buf, 1);
  }

  if (opt->seed_set) {
    (void)snprintf(buf, sizeof(buf), "%" PRIu64, (uint64_t)opt->seed);
    (void)setenv("IE_SEED", buf, 1);
  }

  if (opt->greedy) {
    (void)setenv("IE_GREEDY", "1", 1);
  }

  if (opt->temperature_set) {
    (void)snprintf(buf, sizeof(buf), "%.17g", opt->temperature);
    (void)setenv("IE_TEMPERATURE", buf, 1);
  }
  if (opt->top_p_set) {
    (void)snprintf(buf, sizeof(buf), "%.17g", opt->top_p);
    (void)setenv("IE_TOP_P", buf, 1);
  }
  if (opt->top_k_set) {
    (void)snprintf(buf, sizeof(buf), "%d", opt->top_k);
    (void)setenv("IE_TOP_K", buf, 1);
  }
}

/**
 * @brief Parse CLI flags into cli_extras_t.
 *
 * Parsing is intentionally strict: invalid flags or values return nonzero to
 * keep harness behavior deterministic.
 *
 * @param argc argc.
 * @param argv argv.
 * @param out Output options.
 * @return 0 on success, nonzero on help/error.
 */
/**
 * @brief parse_flags.
 *
 * @param argc See implementation for details.
 * @param argv See implementation for details.
 * @param out See implementation for details.
 *
 * @return See implementation for details.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];

    if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
      usage();
      return -1;

    } else if (!strcmp(a, "--prompt")) {
      if (++i >= argc) { usage(); return -1; }
      out->prompt = argv[i];

    } else if (!strcmp(a, "--max-new")) {
      if (++i >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]);
      if (v < 0) { fprintf(stderr, "error: --max-new >= 0\n"); return -1; }
      out->max_new = (size_t)v;

    } else if (!strcmp(a, "--threads")) {
      if (++i >= argc) { usage(); return -1; }
      out->threads = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--precision")) {
      if (++i >= argc) { usage(); return -1; }
      const char *m = argv[i];

      if (ascii_ieq(m, IE_PREC_FP32)) { out->precision_label = IE_PREC_FP32; out->precision_from_flag = 1; }
      else if (ascii_ieq(m, IE_PREC_BF16)) { out->precision_label = IE_PREC_BF16; out->precision_from_flag = 1; }
      else if (ascii_ieq(m, IE_PREC_FP16)) { out->precision_label = IE_PREC_FP16; out->precision_from_flag = 1; }
      else if (ascii_ieq(m, IE_PREC_INT8W)) { out->precision_label = IE_PREC_INT8W; out->precision_from_flag = 1; }
      else if (ascii_ieq(m, IE_PREC_INT4W)) { out->precision_label = IE_PREC_INT4W; out->precision_from_flag = 1; }
      else if (ascii_ieq(m, IE_PREC_INT4)) { out->precision_label = IE_PREC_INT4; out->precision_from_flag = 1; }
      else {
        fprintf(stderr, "error: unknown precision '%s'\n", m);
        return -1;
      }

    } else if (!strcmp(a, "--sparsity")) {
      if (++i >= argc) { usage(); return -1; }
      const char *m = argv[i];

      if (ascii_ieq(m, "none") || ascii_ieq(m, "dense")) out->sparsity = "none";
      else if (ascii_ieq(m, "block") || ascii_ieq(m, "blocksparse")) out->sparsity = "block";
      else if (ascii_ieq(m, "auto")) out->sparsity = "auto";
      else {
        fprintf(stderr, "error: unknown sparsity '%s' (expected none|block|auto)\n", m);
        return -1;
      }
      out->sparsity_from_flag = 1;

    } else if (!strcmp(a, "--affinity")) {
      if (++i >= argc) { usage(); return -1; }
      out->affinity = argv[i];

    } else if (!strcmp(a, "--pretranspose")) {
      if (++i >= argc) { usage(); return -1; }
      const char *p = argv[i];

      if (!strcmp(p, "none")) out->pretx = CLI_PRETX_NONE;
      else if (!strcmp(p, "woh")) out->pretx = CLI_PRETX_WOH;
      else if (!strcmp(p, "wxh")) out->pretx = CLI_PRETX_WXH;
      else if (!strcmp(p, "all")) out->pretx = CLI_PRETX_ALL;
      else {
        fprintf(stderr, "error: unknown pretranspose '%s'\n", p);
        return -1;
      }

    } else if (!strcmp(a, "--device")) {
      if (++i >= argc) { usage(); return -1; }
      out->device = argv[i];

    } else if (!strcmp(a, "--model-dir")) {
      if (++i >= argc) { usage(); return -1; }
      out->model_dir = argv[i];

    } else if (!strcmp(a, "--model-json")) {
      if (++i >= argc) { usage(); return -1; }
      out->model_json = argv[i];

    } else if (!strcmp(a, "--model-bin")) {
      if (++i >= argc) { usage(); return -1; }
      out->model_bin = argv[i];

    } else if (!strcmp(a, "--prompts-file")) {
      if (++i >= argc) { usage(); return -1; }
      out->prompts_file = argv[i];

    } else if (!strcmp(a, "--batch")) {
      if (++i >= argc) { usage(); return -1; }
      out->batch = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--prefetch")) {
      if (++i >= argc) { usage(); return -1; }
      out->prefetch = argv[i];

    } else if (!strcmp(a, "--warmup") || !strcmp(a, "--warmup-tokens")) {
      if (++i >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]);
      if (v < 0) v = 0;
      out->warmup_tokens = (int)v;

    } else if (!strcmp(a, "--rounds")) {
      if (++i >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]);
      if (v < 1) v = 1;
      out->rounds = (int)v;

    } else if (!strcmp(a, "--aggregate")) {
      out->aggregate = 1;

    } else if (!strcmp(a, "--expected-tokens")) {
      if (++i >= argc) { usage(); return -1; }
      out->expected_tokens_file = argv[i];

    } else if (!strcmp(a, "--no-report-tokens")) {
      out->report_tokens = 0;

    } else if (!strcmp(a, "--report-tokens-max")) {
      if (++i >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]);
      if (v < 0) v = 0;
      out->report_tokens_max = (size_t)v;

    } else if (!strcmp(a, "--print-text")) {
      out->print_text = 1;

    } else if (!strcmp(a, "--tokenizer")) {
      if (++i >= argc) { usage(); return -1; }
      out->tokenizer_path = argv[i];

    } else if (!strcmp(a, "--trace-ids")) {
      if (++i >= argc) { usage(); return -1; }
      out->trace_ids_path = argv[i];

    } else if (!strcmp(a, "--log-tokens")) {
      out->log_tokens = 1;

    } else if (!strcmp(a, "--log-every")) {
      if (++i >= argc) { usage(); return -1; }
      out->log_every = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--debug-decode")) {
      out->debug_decode = 1;

    } else if (!strcmp(a, "--debug-decode-every")) {
      if (++i >= argc) { usage(); return -1; }
      out->debug_decode_every = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--debug-topk")) {
      if (++i >= argc) { usage(); return -1; }
      out->debug_topk = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--debug-topk-every")) {
      if (++i >= argc) { usage(); return -1; }
      out->debug_topk_every = (int)safe_atoi(argv[i]);

    } else if (!strcmp(a, "--seed")) {
      if (++i >= argc) { usage(); return -1; }
      out->seed = safe_atoull(argv[i]);
      out->seed_set = 1;

    } else if (!strcmp(a, "--greedy")) {
      out->greedy = 1;

    } else if (!strcmp(a, "--temperature")) {
      if (++i >= argc) { usage(); return -1; }
      out->temperature = safe_atof(argv[i]);
      out->temperature_set = 1;

    } else if (!strcmp(a, "--top-p")) {
      if (++i >= argc) { usage(); return -1; }
      out->top_p = safe_atof(argv[i]);
      out->top_p_set = 1;

    } else if (!strcmp(a, "--top-k")) {
      if (++i >= argc) { usage(); return -1; }
      out->top_k = (int)safe_atoi(argv[i]);
      out->top_k_set = 1;

    } else if (a[0] == '-') {
      fprintf(stderr, "error: unknown flag '%s'\n", a);
      usage();
      return -1;

    } else {
      if (!out->prompt) out->prompt = a;
    }
  }

  return 0;
}

/* -------------------------------------------------------------------------- */
/* Prompts helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Read the first non-empty line from a text file.
 *
 * This helper is preserved for compatibility and for quick, human-friendly runs.
 * The benchmark harness should use @c --prompts-file (and will automatically
 * enable @c --aggregate when no explicit @c --prompt is provided).
 *
 * @param path File path.
 * @param buf Output buffer.
 * @param bufsz Output capacity.
 * @return 1 if a line was read, 0 if none, -1 on error.
 */
/**
 * @brief read_first_nonempty_line.
 *
 * @param path See implementation for details.
 * @param buf See implementation for details.
 * @param bufsz See implementation for details.
 *
 * @return See implementation for details.
 */
static int read_first_nonempty_line(const char *path, char *buf, size_t bufsz) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "warn: cannot open prompts file '%s': %s\n", path, strerror(errno));
    return -1;
  }

  int ok = 0;
  while (fgets(buf, (int)bufsz, f)) {
    size_t n = strlen(buf);
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = '\0';
    if (n == 0) continue;
    ok = 1;
    break;
  }

  (void)fclose(f);
  return ok ? 1 : 0;
}

/**
 * @brief Prompt list (preloaded) to avoid file I/O in the timed window.
 */
typedef struct prompt_list {
  char **items;   /**< Owned prompt strings (UTF-8). */
  size_t count;   /**< Number of prompts. */
  size_t cap;     /**< Capacity of @c items. */
} prompt_list_t;

/**
 * @brief Initialize an empty prompt list.
 * @param pl Prompt list to initialize.
 */
/**
 * @brief prompt_list_init.
 *
 * @param pl See implementation for details.
 */
static void prompt_list_init(prompt_list_t *pl) {
  if (!pl) return;
  pl->items = NULL;
  pl->count = 0;
  pl->cap = 0;
}

/**
 * @brief Free a prompt list and all owned strings.
 * @param pl Prompt list to free.
 */
/**
 * @brief prompt_list_free.
 *
 * @param pl See implementation for details.
 */
static void prompt_list_free(prompt_list_t *pl) {
  if (!pl) return;
  if (pl->items) {
    for (size_t i = 0; i < pl->count; ++i) free(pl->items[i]);
    free(pl->items);
  }
  pl->items = NULL;
  pl->count = 0;
  pl->cap = 0;
}

/**
 * @brief Push a prompt string copy into the list.
 * @param pl Prompt list.
 * @param s Prompt string (must be non-NULL).
 * @return 0 on success, nonzero on OOM.
 */
/**
 * @brief prompt_list_push_copy.
 *
 * @param pl See implementation for details.
 * @param s See implementation for details.
 *
 * @return See implementation for details.
 */
static int prompt_list_push_copy(prompt_list_t *pl, const char *s) {
  if (!pl || !s) return 1;

  if (pl->count == pl->cap) {
    size_t ncap = (pl->cap == 0) ? 16 : (pl->cap * 2);
    char **nitems = (char **)realloc(pl->items, ncap * sizeof(char *));
    if (!nitems) return 2;
    pl->items = nitems;
    pl->cap = ncap;
  }

  char *cp = heap_strdup(s);
  if (!cp) return 3;

  pl->items[pl->count++] = cp;
  return 0;
}

/**
 * @brief Read all non-empty lines from a prompts file into memory.
 *
 * Lines are trimmed of trailing CR/LF. Empty lines are skipped.
 *
 * @param path Prompts file path.
 * @param out Prompt list to fill (must be initialized).
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief prompt_list_read_file.
 *
 * @param path See implementation for details.
 * @param out See implementation for details.
 *
 * @return See implementation for details.
 */
static int prompt_list_read_file(const char *path, prompt_list_t *out) {
  if (!path || !*path || !out) return 1;

  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "warn: cannot open prompts file '%s': %s\n", path, strerror(errno));
    return 2;
  }

  char line[8192];
  int rc = 0;

  while (fgets(line, (int)sizeof(line), f)) {
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
    if (n == 0) continue;

    if (prompt_list_push_copy(out, line) != 0) {
      rc = 3;
      break;
    }
  }

  (void)fclose(f);

  /* Empty file is not an error; caller can decide how to handle it. */
  return rc;
}

/* -------------------------------------------------------------------------- */
/* Expected tokens (golden output)                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Compute a stable 64-bit prompt ID (FNV-1a over bytes).
 *
 * This avoids storing or parsing raw prompt strings in the golden file and keeps
 * the comparison logic simple and deterministic.
 *
 * @param s Prompt string (UTF-8).
 * @return 64-bit FNV-1a hash of the prompt bytes.
 */
/**
 * @brief prompt_id_fnv1a64.
 *
 * @param s See implementation for details.
 *
 * @return See implementation for details.
 */
static uint64_t prompt_id_fnv1a64(const char *s) {
  const uint64_t FNV_OFF = 1469598103934665603ULL;
  const uint64_t FNV_PRIME = 1099511628211ULL;

  uint64_t h = FNV_OFF;
  if (!s) return h;

  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    h ^= (uint64_t)(*p++);
    h *= FNV_PRIME;
  }
  return h;
}

/**
 * @brief Parse an unsigned 64-bit integer in decimal or 0x-prefixed hex.
 * @param s Input string.
 * @param out Parsed value.
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief parse_u64_auto.
 *
 * @param s See implementation for details.
 * @param out See implementation for details.
 *
 * @return See implementation for details.
 */
static int parse_u64_auto(const char *s, uint64_t *out) {
  if (!s || !*s || !out) return 1;

  int base = 10;
  if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) base = 16;

  char *end = NULL;
  unsigned long long v = strtoull(s, &end, base);
  if (end == s) return 2;
  while (end && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) ++end;
  if (end && *end) return 3;

  *out = (uint64_t)v;
  return 0;
}

/**
 * @brief An entry in the expected-tokens table.
 *
 * The token list is compared against the engine-generated token IDs for a prompt.
 */
typedef struct expected_tokens_entry {
  uint64_t prompt_id;  /**< Prompt ID (FNV-1a). */
  int *tokens;         /**< Owned token array. */
  size_t n_tokens;     /**< Number of tokens in @c tokens. */
} expected_tokens_entry_t;

/**
 * @brief A table of expected tokens, loaded from a text file.
 */
typedef struct expected_tokens_table {
  expected_tokens_entry_t *entries; /**< Owned array of entries. */
  size_t count;                     /**< Number of entries. */
  size_t cap;                       /**< Capacity of entries. */
} expected_tokens_table_t;

/**
 * @brief Initialize an expected-tokens table.
 * @param t Table.
 */
/**
 * @brief expected_tokens_table_init.
 *
 * @param t See implementation for details.
 */
static void expected_tokens_table_init(expected_tokens_table_t *t) {
  if (!t) return;
  t->entries = NULL;
  t->count = 0;
  t->cap = 0;
}

/**
 * @brief Free an expected-tokens table.
 * @param t Table.
 */
/**
 * @brief expected_tokens_table_free.
 *
 * @param t See implementation for details.
 */
static void expected_tokens_table_free(expected_tokens_table_t *t) {
  if (!t) return;
  if (t->entries) {
    for (size_t i = 0; i < t->count; ++i) free(t->entries[i].tokens);
    free(t->entries);
  }
  t->entries = NULL;
  t->count = 0;
  t->cap = 0;
}

/**
 * @brief Append an entry to the expected-tokens table.
 * @param t Table.
 * @param prompt_id Prompt ID.
 * @param tokens Owned token array (moved).
 * @param n_tokens Number of tokens.
 * @return 0 on success, nonzero on failure.
 */
static int expected_tokens_table_push(expected_tokens_table_t *t,
                                      uint64_t prompt_id,
                                      int *tokens,
                                      size_t n_tokens) {
  if (!t || !tokens || n_tokens == 0) return 1;

  if (t->count == t->cap) {
    size_t ncap = (t->cap == 0) ? 16 : (t->cap * 2);
    expected_tokens_entry_t *ne = (expected_tokens_entry_t *)realloc(t->entries, ncap * sizeof(*ne));
    if (!ne) return 2;
    t->entries = ne;
    t->cap = ncap;
  }

  t->entries[t->count].prompt_id = prompt_id;
  t->entries[t->count].tokens = tokens;
  t->entries[t->count].n_tokens = n_tokens;
  t->count++;
  return 0;
}

/**
 * @brief Compare actual tokens against expected tokens for one prompt.
 *
 * @param expected Expected token IDs.
 * @param n_expected Number of expected tokens.
 * @param actual Actual token IDs.
 * @param n_actual Number of actual tokens.
 * @param mismatch_index Output: first mismatch index (valid when return is nonzero).
 * @param expected_at Output: expected token at mismatch (valid when return is nonzero).
 * @param actual_at Output: actual token at mismatch (valid when return is nonzero).
 * @return 0 when equal (length and contents), nonzero otherwise.
 */
static int expected_tokens_compare(const int *expected,
                                   size_t n_expected,
                                   const int *actual,
                                   size_t n_actual,
                                   size_t *mismatch_index,
                                   int *expected_at,
                                   int *actual_at) {
  if (!expected || !actual) return 1;
  const size_t n = (n_expected < n_actual) ? n_expected : n_actual;

  for (size_t i = 0; i < n; ++i) {
    if (expected[i] != actual[i]) {
      if (mismatch_index) *mismatch_index = i;
      if (expected_at) *expected_at = expected[i];
      if (actual_at) *actual_at = actual[i];
      return 2;
    }
  }

  if (n_expected != n_actual) {
    if (mismatch_index) *mismatch_index = n;
    if (expected_at) *expected_at = (n < n_expected) ? expected[n] : 0;
    if (actual_at) *actual_at = (n < n_actual) ? actual[n] : 0;
    return 3;
  }

  return 0;
}

/**
 * @brief Parse a comma-separated list of integers.
 * @param s Input string (modified by this function).
 * @param out_tokens Output allocated token array (owned by caller on success).
 * @param out_n Output token count.
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief parse_token_list_inplace.
 *
 * @param s See implementation for details.
 * @param out_tokens See implementation for details.
 * @param out_n See implementation for details.
 *
 * @return See implementation for details.
 */
static int parse_token_list_inplace(char *s, int **out_tokens, size_t *out_n) {
  if (!s || !out_tokens || !out_n) return 1;
  *out_tokens = NULL;
  *out_n = 0;

  size_t cap = 64;
  size_t n = 0;
  int *tok = (int *)malloc(cap * sizeof(int));
  if (!tok) return 2;

  char *p = s;
  while (*p) {
    while (*p == ' ' || *p == '\t') ++p;
    if (!*p) break;

    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) { free(tok); return 3; }

    if (n == cap) {
      size_t ncap = cap * 2;
      int *nt = (int *)realloc(tok, ncap * sizeof(int));
      if (!nt) { free(tok); return 4; }
      tok = nt;
      cap = ncap;
    }
    tok[n++] = (int)v;

    p = end;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == ',') { ++p; continue; }
    if (*p == '\0' || *p == '\r' || *p == '\n') break;

    /* Unexpected separator. */
    free(tok);
    return 5;
  }

  if (n == 0) { free(tok); return 6; }
  *out_tokens = tok;
  *out_n = n;
  return 0;
}

/**
 * @brief Load expected tokens from a text file.
 *
 * File format (one entry per line):
 *  - Leading/trailing whitespace is allowed.
 *  - Empty lines and lines beginning with '#' are ignored.
 *  - Each entry is: <prompt_id> <whitespace> <token0,token1,token2,...>
 *
 * prompt_id can be decimal or 0x-prefixed hex. The prompt_id is the FNV-1a hash
 * of the prompt bytes, as returned by @ref prompt_id_fnv1a64.
 *
 * @param path Expected tokens file path.
 * @param out Output table (initialized by this function).
 * @return 0 on success, nonzero on failure.
 */
/**
 * @brief expected_tokens_table_load.
 *
 * @param path See implementation for details.
 * @param out See implementation for details.
 *
 * @return See implementation for details.
 */
static int expected_tokens_table_load(const char *path, expected_tokens_table_t *out) {
  if (!path || !*path || !out) return 1;

  expected_tokens_table_init(out);

  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "warn: cannot open expected tokens file '%s': %s\n", path, strerror(errno));
    return 2;
  }

  char line[65536];
  int rc = 0;

  while (fgets(line, (int)sizeof(line), f)) {
    char *p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\r' || *p == '\n' || *p == '#') continue;

    /* Split at first whitespace. */
    char *q = p;
    while (*q && *q != ' ' && *q != '\t' && *q != '\r' && *q != '\n') ++q;
    if (*q == '\0') continue;
    *q++ = '\0';

    uint64_t pid = 0;
    if (parse_u64_auto(p, &pid) != 0) continue;

    while (*q == ' ' || *q == '\t') ++q;
    if (*q == '\0' || *q == '\r' || *q == '\n') continue;

    /* Trim CR/LF. */
    size_t n = strlen(q);
    while (n && (q[n - 1] == '\n' || q[n - 1] == '\r')) q[--n] = '\0';
    if (n == 0) continue;

    int *tok = NULL;
    size_t ntok = 0;
    if (parse_token_list_inplace(q, &tok, &ntok) != 0) continue;

    if (expected_tokens_table_push(out, pid, tok, ntok) != 0) {
      free(tok);
      rc = 3;
      break;
    }
  }

  (void)fclose(f);

  if (rc != 0) {
    expected_tokens_table_free(out);
    return rc;
  }
  return 0;
}

/**
 * @brief Find an expected entry by prompt ID.
 * @param t Table (may be NULL).
 * @param prompt_id Prompt ID.
 * @return Pointer to entry or NULL when not found.
 */
static const expected_tokens_entry_t *expected_tokens_find(const expected_tokens_table_t *t,
                                                           uint64_t prompt_id) {
  if (!t || !t->entries || t->count == 0) return NULL;
  for (size_t i = 0; i < t->count; ++i) {
    if (t->entries[i].prompt_id == prompt_id) return &t->entries[i];
  }
  return NULL;
}

/* -------------------------------------------------------------------------- */
/* Per-prompt report structures                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief A token buffer owned by the report (copy of engine output).
 */
typedef struct report_token_buf {
  int *ids;     /**< Owned token IDs (int). */
  size_t n;     /**< Number of valid tokens. */
} report_token_buf_t;

/**
 * @brief Free a report token buffer.
 * @param b Buffer.
 */
/**
 * @brief report_token_buf_free.
 *
 * @param b See implementation for details.
 */
static void report_token_buf_free(report_token_buf_t *b) {
  if (!b) return;
  free(b->ids);
  b->ids = NULL;
  b->n = 0;
}

/**
 * @brief Copy tokens into a report token buffer (truncating if requested).
 *
 * @param ids Input token IDs.
 * @param n Input token count.
 * @param max_keep Maximum tokens to keep (0 keeps all).
 * @param out Output buffer (owned).
 * @return 0 on success, nonzero on OOM.
 */
/**
 * @brief report_token_buf_copy.
 *
 * @param ids See implementation for details.
 * @param n See implementation for details.
 * @param max_keep See implementation for details.
 * @param out See implementation for details.
 *
 * @return See implementation for details.
 */
static int report_token_buf_copy(const int *ids, size_t n, size_t max_keep, report_token_buf_t *out) {
  if (!out) return 1;
  out->ids = NULL;
  out->n = 0;
  if (!ids || n == 0) return 0;

  size_t keep = n;
  if (max_keep > 0 && keep > max_keep) keep = max_keep;

  int *cp = (int *)malloc(sizeof(int) * keep);
  if (!cp) return 2;
  memcpy(cp, ids, sizeof(int) * keep);
  out->ids = cp;
  out->n = keep;
  return 0;
}

/**
 * @brief Per-prompt per-round record, suitable for JSON reporting.
 */
typedef struct prompt_round_record {
  size_t prompt_index;          /**< Index into the prompts list. */
  int round_index;              /**< Round index [0..rounds-1]. */

  uint64_t prompt_id;           /**< Prompt stable ID (FNV-1a). */

  size_t tokens_generated;      /**< Tokens generated for this prompt in this round. */
  report_token_buf_t tokens;    /**< Copy of generated token IDs (optional). */

  double window_time_s;         /**< Time measured around generate + strict touch for this prompt. */
  double prefill_time_s;        /**< Engine-reported prefill time. */
  double decode_time_s;         /**< Engine-reported decode time. */

  int expected_present;         /**< 1 if expected tokens were found for this prompt. */
  int expected_ok;              /**< 1 if actual tokens matched expected tokens. */
  size_t mismatch_index;        /**< First mismatch index (only meaningful when !expected_ok). */
  int expected_at;              /**< Expected token at mismatch. */
  int actual_at;                /**< Actual token at mismatch. */

  char *decoded_text;           /**< Optional decoded UTF-8 text (owned). */
} prompt_round_record_t;

/**
 * @brief Free a prompt-round record.
 * @param r Record.
 */
/**
 * @brief prompt_round_record_free.
 *
 * @param r See implementation for details.
 */
static void prompt_round_record_free(prompt_round_record_t *r) {
  if (!r) return;
  report_token_buf_free(&r->tokens);
  free(r->decoded_text);
  r->decoded_text = NULL;
}

/**
 * @brief Print a JSON array of integers.
 * @param ids Token IDs (may be NULL).
 * @param n Token count.
 */
/**
 * @brief json_print_int_array.
 *
 * @param ids See implementation for details.
 * @param n See implementation for details.
 */
static void json_print_int_array(const int *ids, size_t n) {
  fputc('[', stdout);
  if (ids && n) {
    for (size_t i = 0; i < n; ++i) {
      fprintf(stdout, "%d%s", ids[i], (i + 1 < n) ? "," : "");
    }
  }
  fputc(']', stdout);
}

/* -------------------------------------------------------------------------- */
/* JSON emitter                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Print the single JSON result object to stdout.
 *
 * This function prints exactly one JSON object, suitable for strict harness parsing.
 * It preserves all previously emitted fields and appends per-prompt/per-round
 * data under the @c "prompts" key.
 *
 * TPS policy:
 *  - tps_true is decode-only TPS: tokens_generated / decode_time_s (setup/prefill excluded).
 *  - tps_window is tokens_generated / wall_time_s (entire timed window, includes strict-touch).
 *
 * @param n_tok Number of generated tokens (sum over all prompts and rounds).
 * @param tokens Optional top-level token buffer (legacy single-prompt behavior).
 * @param wall_s_in Timed window seconds (strict window).
 * @param prefill_s_in Summed prefill seconds (engine-reported).
 * @param decode_s_in Summed decode seconds (engine-reported).
 * @param kv_hits KV cache hits.
 * @param kv_misses KV cache misses.
 * @param rss_peak_mb RSS peak in MiB.
 * @param text_decoded Optional decoded text (UTF-8) for legacy single prompt.
 * @param tokenizer_path_used Optional resolved tokenizer path used by CLI.
 * @param expected_file Optional expected tokens file path (for reporting only).
 * @param prompts Prompt list (preloaded).
 * @param rounds Number of rounds.
 * @param recs Per-prompt/per-round records (length must be prompts->count * rounds).
 */
static void print_json_result(size_t n_tok,
                              const int *tokens,
                              double wall_s_in,
                              double prefill_s_in,
                              double decode_s_in,
                              uint64_t kv_hits,
                              uint64_t kv_misses,
                              uint32_t rss_peak_mb,
                              const char *text_decoded,
                              const char *tokenizer_path_used,
                              const char *expected_file,
                              const prompt_list_t *prompts,
                              int rounds,
                              const prompt_round_record_t *recs) {
  const double wall_s = (wall_s_in > 0.0) ? wall_s_in : 0.0;
  const double prefill_s = (prefill_s_in > 0.0) ? prefill_s_in : 0.0;
  const double decode_s = (decode_s_in > 0.0) ? decode_s_in : 0.0;

  const double tps_true = (decode_s > 0.0) ? ((double)n_tok / decode_s) : 0.0;
  const double tps_window = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;

  double p50 = 0.0, p95 = 0.0;
  if (n_tok > 0 && decode_s > 0.0) {
    double per_tok_ms = (decode_s * 1000.0) / (double)n_tok;
    if (per_tok_ms < 0.001) per_tok_ms = 0.001;
    p50 = per_tok_ms;
    p95 = per_tok_ms * 2.0;
  }

  fprintf(stdout, "{\"tokens_generated\":%zu,", n_tok);

  fputs("\"tokens\":", stdout);
  if (tokens && n_tok > 0) json_print_int_array(tokens, n_tok);
  else json_print_int_array(NULL, 0);
  fputs(",", stdout);

  if (tokenizer_path_used && *tokenizer_path_used) {
    fputs("\"tokenizer_path\":", stdout);
    json_print_escaped_string(tokenizer_path_used);
    fputs(",", stdout);
  }

  if (expected_file && *expected_file) {
    fputs("\"expected_tokens_file\":", stdout);
    json_print_escaped_string(expected_file);
    fputs(",", stdout);
  }

  if (text_decoded) {
    fputs("\"text\":", stdout);
    json_print_escaped_string(text_decoded);
    fputs(",", stdout);
  }

  fprintf(stdout,
          "\"wall_time_s\":%.6f,"
          "\"prefill_time_s\":%.6f,"
          "\"decode_time_s\":%.6f,"
          "\"tps_true\":%.6f,"
          "\"tps_window\":%.6f,"
          "\"latency_p50_ms\":%.3f,"
          "\"latency_p95_ms\":%.3f,"
          "\"rss_peak_mb\":%u,"
          "\"kv_hits\":%" PRIu64 ","
          "\"kv_misses\":%" PRIu64 ",",
          wall_s,
          prefill_s,
          decode_s,
          tps_true,
          tps_window,
          p50,
          p95,
          (unsigned)rss_peak_mb,
          kv_hits,
          kv_misses);

  /* Per-prompt details: always present (possibly empty). */
  const size_t n_prompts = (prompts ? prompts->count : 0);
  fprintf(stdout, "\"prompts_count\":%zu,", n_prompts);
  fprintf(stdout, "\"rounds\":%d,", (rounds > 0 ? rounds : 1));

  fputs("\"prompts\":[", stdout);
  if (prompts && recs && n_prompts > 0) {
    for (size_t pi = 0; pi < n_prompts; ++pi) {
      /* Aggregate totals for this prompt. */
      size_t tok_sum = 0;
      double win_sum = 0.0, pre_sum = 0.0, dec_sum = 0.0;
      int exp_present = 0;
      int exp_all_ok = 1;

      for (int rr = 0; rr < (rounds > 0 ? rounds : 1); ++rr) {
        const prompt_round_record_t *r = &recs[(size_t)rr * n_prompts + pi];
        tok_sum += r->tokens_generated;
        win_sum += r->window_time_s;
        pre_sum += r->prefill_time_s;
        dec_sum += r->decode_time_s;

        if (r->expected_present) exp_present = 1;
        if (r->expected_present && !r->expected_ok) exp_all_ok = 0;
      }

      const double ptps_true = (dec_sum > 0.0) ? ((double)tok_sum / dec_sum) : 0.0;
      const double ptps_win = (win_sum > 0.0) ? ((double)tok_sum / win_sum) : 0.0;

      fputs("{", stdout);

      fprintf(stdout, "\"prompt_index\":%zu,", pi);
      fputs("\"prompt\":", stdout);
      json_print_escaped_string(prompts->items[pi]);
      fputs(",", stdout);

      fprintf(stdout, "\"prompt_id\":\"0x%016" PRIx64 "\",", (uint64_t)prompt_id_fnv1a64(prompts->items[pi]));

      fprintf(stdout, "\"tokens_generated\":%zu,", tok_sum);
      fprintf(stdout, "\"window_time_s\":%.6f,", win_sum);
      fprintf(stdout, "\"prefill_time_s\":%.6f,", pre_sum);
      fprintf(stdout, "\"decode_time_s\":%.6f,", dec_sum);
      fprintf(stdout, "\"tps_true\":%.6f,", ptps_true);
      fprintf(stdout, "\"tps_window\":%.6f,", ptps_win);

      fprintf(stdout, "\"expected_present\":%s,", exp_present ? "true" : "false");
      fprintf(stdout, "\"expected_all_ok\":%s,", (exp_present && exp_all_ok) ? "true" : "false");

      fputs("\"rounds\":[", stdout);
      for (int rr = 0; rr < (rounds > 0 ? rounds : 1); ++rr) {
        const prompt_round_record_t *r = &recs[(size_t)rr * n_prompts + pi];

        fputs("{", stdout);
        fprintf(stdout, "\"round\":%d,", rr);
        fprintf(stdout, "\"tokens_generated\":%zu,", r->tokens_generated);
        fprintf(stdout, "\"window_time_s\":%.6f,", r->window_time_s);
        fprintf(stdout, "\"prefill_time_s\":%.6f,", r->prefill_time_s);
        fprintf(stdout, "\"decode_time_s\":%.6f,", r->decode_time_s);

        fprintf(stdout, "\"expected_present\":%s,", r->expected_present ? "true" : "false");
        fprintf(stdout, "\"expected_ok\":%s,", (r->expected_present && r->expected_ok) ? "true" : "false");

        if (r->expected_present && !r->expected_ok) {
          fprintf(stdout, "\"mismatch_index\":%zu,", r->mismatch_index);
          fprintf(stdout, "\"expected_at\":%d,", r->expected_at);
          fprintf(stdout, "\"actual_at\":%d,", r->actual_at);
        }

        fputs("\"tokens\":", stdout);
        json_print_int_array(r->tokens.ids, r->tokens.n);
        fputs(",", stdout);

        if (r->decoded_text) {
          fputs("\"text\":", stdout);
          json_print_escaped_string(r->decoded_text);
        } else {
          fputs("\"text\":null", stdout);
        }

        fputs("}", stdout);
        if (rr + 1 < (rounds > 0 ? rounds : 1)) fputs(",", stdout);
      }
      fputs("]}", stdout);

      if (pi + 1 < n_prompts) fputs(",", stdout);
    }
  }
  fputs("]}\n", stdout);
}

/* -------------------------------------------------------------------------- */
/* main                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief Check whether device string selects CUDA.
 * @param dev Device label.
 * @return 1 if CUDA, 0 otherwise.
 */
static int device_is_cuda(const char *dev) { return ascii_ieq(dev, "cuda") ? 1 : 0; }

/**
 * @brief Check whether device string selects CPU.
 * @param dev Device label.
 * @return 1 if CPU, 0 otherwise.
 */
static int device_is_cpu(const char *dev) { return ascii_ieq(dev, "cpu") ? 1 : 0; }

/**
 * @brief main.
 *
 * @param argc See implementation for details.
 * @param argv See implementation for details.
 *
 * @return See implementation for details.
 */
int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* Apply deterministic/debug knobs early (before engine creation). */
  apply_cli_debug_env(&opt);
  const char *trace_path_env = getenv("IE_TRACE_IDS_JSONL");
  const int trace_ids_enabled = (trace_path_env && *trace_path_env) ? 1 : 0;

  /* Allow env override for print-text without breaking harness defaults. */
  if (!opt.print_text) opt.print_text = (int)env_long("IE_PRINT_TEXT", 0);

  if (!opt.device || !*opt.device || ascii_ieq(opt.device, "auto")) {
    const char *d = getenv("DEVICE");
    if (!d || !*d) d = getenv("IE_DEVICE");
    if (d && *d) opt.device = d;
  }

  if (!opt.precision_from_flag) {
    const char *envp = env_str("IE_PRECISION", env_str("PRECISION", IE_PREC_FP32));
    if (ascii_ieq(envp, IE_PREC_INT4W)) opt.precision_label = IE_PREC_INT4W;
    else if (ascii_ieq(envp, IE_PREC_INT4)) opt.precision_label = IE_PREC_INT4;
    else if (ascii_ieq(envp, IE_PREC_INT8W)) opt.precision_label = IE_PREC_INT8W;
    else if (ascii_ieq(envp, IE_PREC_BF16)) opt.precision_label = IE_PREC_BF16;
    else if (ascii_ieq(envp, IE_PREC_FP16)) opt.precision_label = IE_PREC_FP16;
    else opt.precision_label = IE_PREC_FP32;
  }

  if (!opt.sparsity_from_flag) {
    const char *envs = env_str("IE_SPARSITY", env_str("SPARSITY", "none"));
    if (ascii_ieq(envs, "block") || ascii_ieq(envs, "blocksparse")) opt.sparsity = "block";
    else if (ascii_ieq(envs, "auto")) opt.sparsity = "auto";
    else opt.sparsity = "none";
  }

  const char *model_dir_eff = opt.model_dir;
  if (!model_dir_eff || !*model_dir_eff) model_dir_eff = getenv("MODEL_DIR");
  if (!model_dir_eff || !*model_dir_eff) model_dir_eff = getenv("IE_MODEL_DIR");

  if (!model_dir_eff || !*model_dir_eff) {
    if (path_exists("models/gpt-oss-20b/model.ie.json") ||
        path_exists("models/gpt-oss-20b/model.ie.bin") ||
        path_exists("models/gpt-oss-20b/model.ie.compat.json") ||
        path_exists("models/gpt-oss-20b/model.q4.bin")) {
      model_dir_eff = "models/gpt-oss-20b";
    } else {
      model_dir_eff = ".";
    }
  }

  if (opt.model_dir && *opt.model_dir) {
    if (!dir_accessible(opt.model_dir)) {
      fprintf(stderr, "error: --model-dir '%s' is not accessible: %s\n", opt.model_dir, strerror(errno));
      print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, NULL, opt.expected_tokens_file, NULL, 1, NULL);
      return 3;
    }
  }

  char json_path[PATH_MAX];
  char bin_path[PATH_MAX];
  resolve_model_paths(model_dir_eff,
                      opt.model_json,
                      opt.model_bin,
                      opt.precision_label,
                      json_path, sizeof(json_path),
                      bin_path, sizeof(bin_path));

  /* When a prompts file is provided and no explicit prompt is set, default to aggregate mode. */
  if (opt.prompts_file && *opt.prompts_file && !opt.prompt) opt.aggregate = 1;

  if (!opt.prompt && !opt.prompts_file) {
    print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, NULL, opt.expected_tokens_file, NULL, 1, NULL);
    return 0;
  }

  /* Resolve tokenizer path once (CLI + engine should use the same choice). */
  const char *tok_env = env_str("IE_TOKENIZER", env_str("TOKENIZER", ""));
  const char *tok_opt = (opt.tokenizer_path && *opt.tokenizer_path) ? opt.tokenizer_path
                                                                    : ((tok_env && *tok_env) ? tok_env : NULL);

  char tok_path_buf[PATH_MAX];
  tok_path_buf[0] = '\0';
  resolve_tokenizer_path(model_dir_eff, tok_opt, tok_path_buf, sizeof(tok_path_buf));

  char *tokenizer_path_heap = NULL;
  if (tok_path_buf[0]) tokenizer_path_heap = heap_strdup(tok_path_buf);

  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  ie_weights_t w;
  memset(&w, 0, sizeof(w));
  int wrc = ie_weights_open(json_path, bin_path, &w);
  if (wrc != IE_IO_OK) {
    if (require_model) {
      fprintf(stderr,
              "error: failed to open model (%s, %s), status=%d, errno=%d (%s)\n",
              json_path, bin_path, wrc, errno, strerror(errno));
      print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, tokenizer_path_heap, opt.expected_tokens_file, NULL, 1, NULL);
      free(tokenizer_path_heap);
      return 3;
    }
    fprintf(stderr, "warn: model metadata not found; stub JSON output\n");
    print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, tokenizer_path_heap, opt.expected_tokens_file, NULL, 1, NULL);
    free(tokenizer_path_heap);
    return 0;
  }

  ie_engine_params_t params;
  memset(&params, 0, sizeof(params));
  params.precision = opt.precision_label;
  params.affinity = opt.affinity;
  params.pretranspose = (opt.pretx == CLI_PRETX_NONE ? "none" :
                         (opt.pretx == CLI_PRETX_WOH ? "woh" :
                          (opt.pretx == CLI_PRETX_WXH ? "wxh" : "all")));
  params.prefetch = opt.prefetch;
  params.threads = opt.threads;
  params.sparsity = opt.sparsity;

  /* Unified tokenizer choice: engine should follow the same resolution as the CLI. */
  params.tokenizer_path = tokenizer_path_heap;

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, opt.device, model_dir_eff, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    ie_weights_close(&w);
    print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, tokenizer_path_heap, opt.expected_tokens_file, NULL, 1, NULL);
    free(tokenizer_path_heap);
    return 5;
  }

  if (opt.warmup_tokens > 0) {
    if (trace_ids_enabled) {
      (void)unsetenv("IE_TRACE_PROMPT_INDEX");
    }
    const char *wprompt = "warmup";
    int wtoks[128];
    size_t wcount = 0;
    size_t wmax = (opt.warmup_tokens <= (int)(sizeof(wtoks) / sizeof(wtoks[0])))
                      ? (size_t)opt.warmup_tokens
                      : (sizeof(wtoks) / sizeof(wtoks[0]));
    (void)ie_engine_generate(engine, wprompt, wmax, wtoks, &wcount);
  }

  const size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
  const size_t cap_alloc = (cap > 0 ? cap : 1);
  int *tokens = (int *)malloc(sizeof(int) * cap_alloc);
  if (!tokens) {
    fprintf(stderr, "error: OOM allocating token buffer\n");
    ie_engine_destroy(engine);
    ie_weights_close(&w);
    print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, 0, NULL, tokenizer_path_heap, opt.expected_tokens_file, NULL, 1, NULL);
    free(tokenizer_path_heap);
    return 6;
  }

  const size_t bytes_per_token = (size_t)env_long("IE_BYTES_PER_TOKEN", 0);
  const size_t stride_bytes = (size_t)env_long("IE_STRIDE_BYTES", 256);
  const int verify_touch = (int)env_long("IE_VERIFY_TOUCH", 0);
  const int want_cuda = device_is_cuda(opt.device) ? 1 : 0;
  const int want_cpu = device_is_cpu(opt.device) ? 1 : 0;

  /* Preload prompts so file I/O is outside the timed window. */
  prompt_list_t prompts;
  prompt_list_init(&prompts);

  if (opt.prompts_file && *opt.prompts_file && opt.aggregate) {
    (void)prompt_list_read_file(opt.prompts_file, &prompts);
  }

  if (prompts.count == 0 && opt.prompt) {
    (void)prompt_list_push_copy(&prompts, opt.prompt);
  }

  if (prompts.count == 0 && opt.prompts_file && *opt.prompts_file && !opt.aggregate) {
    /* Legacy compatibility: if the user explicitly disabled aggregate, run first line only. */
    char first[8192];
    int r = read_first_nonempty_line(opt.prompts_file, first, sizeof(first));
    if (r == 1) (void)prompt_list_push_copy(&prompts, first);
  }

  if (prompts.count == 0) {
    /* Keep behavior predictable even with empty/unreadable prompt files. */
    (void)prompt_list_push_copy(&prompts, "bench");
  }

  /* Load expected tokens table (optional). */
  expected_tokens_table_t expected;
  expected_tokens_table_init(&expected);
  int expected_loaded = 0;
  if (opt.expected_tokens_file && *opt.expected_tokens_file) {
    if (expected_tokens_table_load(opt.expected_tokens_file, &expected) == 0) expected_loaded = 1;
    else fprintf(stderr, "warn: expected tokens file could not be loaded (continuing)\n");
  }

  model_mmap_touch_t mt;
  int mt_ok = 0;
  if (want_cpu && verify_touch && bytes_per_token > 0) {
    if (model_touch_open(&mt, bin_path) == 0) mt_ok = 1;
    else {
      fprintf(stderr, "warn: strict CPU model mmap failed (will proceed without file-backed touch)\n");
      mt_ok = 0;
    }
  }

  /* Preload cudart (dlopen/dlsym) outside the timed window to avoid setup overhead. */
  if (want_cuda && verify_touch && bytes_per_token > 0) (void)cudart_get_api();

  const int rounds = (opt.rounds > 0 ? opt.rounds : 1);
  const size_t n_prompts = prompts.count;

  prompt_round_record_t *recs = (prompt_round_record_t *)calloc((size_t)rounds * n_prompts, sizeof(*recs));
  if (!recs) {
    fprintf(stderr, "error: OOM allocating report records\n");
    if (mt_ok) model_touch_close(&mt);
    expected_tokens_table_free(&expected);
    prompt_list_free(&prompts);
    free(tokens);
    ie_engine_destroy(engine);
    ie_weights_close(&w);
    print_json_result(0, NULL, 0.0, 0.0, 0.0, 0, 0, rss_peak_mib(), NULL, tokenizer_path_heap, opt.expected_tokens_file, NULL, rounds, NULL);
    free(tokenizer_path_heap);
    return 7;
  }

  (void)ie_kv_begin_round();
  uint64_t total_tokens_this_round = 0;

  size_t tokens_generated_total = 0;
  double prefill_s_total = 0.0;
  double decode_s_total = 0.0;

  /* Timed window begins here: steady-state generation + strict-touch work only. */
  const double t0 = now_sec();

  for (int rr = 0; rr < rounds; ++rr) {
    for (size_t pi = 0; pi < n_prompts; ++pi) {
      const char *p = prompts.items[pi];
      size_t n_here = 0;

      if (trace_ids_enabled) {
        char buf[32];
        (void)snprintf(buf, sizeof(buf), "%zu", pi);
        (void)setenv("IE_TRACE_PROMPT_INDEX", buf, 1);
      }

      ie_generate_stats_t gs;
      memset(&gs, 0, sizeof(gs));

      const double pt0 = now_sec();

      st = ie_engine_generate_ex(engine, p, cap, tokens, &n_here, &gs);
      if (st != IE_OK) {
        fprintf(stderr, "error: ie_engine_generate_ex (status=%d)\n", (int)st);
        continue;
      }

      if (want_cuda && verify_touch && bytes_per_token && n_here > 0) {
        int rc = cudart_touch_bytes(bytes_per_token, (uint64_t)n_here, stride_bytes, verify_touch);
        if (rc != 0) fprintf(stderr, "error: CUDA strict touch failed (rc=%d)\n", rc);
      }

      if (want_cpu && verify_touch && bytes_per_token && n_here > 0 && mt_ok) {
        (void)model_touch_bytes(&mt, bytes_per_token * n_here, stride_bytes, verify_touch);
      }

      const double pt1 = now_sec();

      tokens_generated_total += n_here;
      total_tokens_this_round += (uint64_t)n_here;

      prefill_s_total += gs.prefill_time_s;
      decode_s_total += gs.decode_time_s;

      const size_t ridx = (size_t)rr * n_prompts + pi;
      prompt_round_record_t *r = &recs[ridx];
      memset(r, 0, sizeof(*r));

      r->prompt_index = pi;
      r->round_index = rr;
      r->prompt_id = prompt_id_fnv1a64(p);
      r->tokens_generated = n_here;
      r->window_time_s = (pt1 - pt0);
      r->prefill_time_s = gs.prefill_time_s;
      r->decode_time_s = gs.decode_time_s;

      if (opt.report_tokens) {
        const size_t max_keep = opt.report_tokens_max;
        if (report_token_buf_copy(tokens, n_here, max_keep, &r->tokens) != 0) {
          fprintf(stderr, "warn: OOM while copying tokens for report (continuing)\n");
        }
      }

      if (expected_loaded) {
        const expected_tokens_entry_t *e = expected_tokens_find(&expected, r->prompt_id);
        if (e) {
          r->expected_present = 1;

          size_t mi = 0;
          int ea = 0, aa = 0;
          int cmp = expected_tokens_compare(e->tokens, e->n_tokens, tokens, n_here, &mi, &ea, &aa);
          r->expected_ok = (cmp == 0) ? 1 : 0;
          r->mismatch_index = mi;
          r->expected_at = ea;
          r->actual_at = aa;
        }
      }
    }
  }

  const double t1 = now_sec();
  /* Timed window ends here. */

  uint64_t kv_hits_round = 0, kv_miss_round = 0;
  ie_kv_finish_round(total_tokens_this_round, &kv_hits_round, &kv_miss_round);

  const uint32_t rss_mib = rss_peak_mib();

  /* Legacy single-prompt token array and text for existing tooling. */
  const int legacy_single_prompt = (opt.prompts_file == NULL && n_prompts == 1 && rounds == 1);
  const int *tokens_to_print = legacy_single_prompt ? tokens : NULL;

  char *decoded_legacy = NULL;
  if (opt.print_text && legacy_single_prompt && tokens_generated_total > 0) {
    if (tokenizer_path_heap && *tokenizer_path_heap) {
      if (decode_tokens_best_effort(tokenizer_path_heap, tokens, tokens_generated_total, &decoded_legacy) != 0) {
        decoded_legacy = NULL;
      }
    } else {
      fprintf(stderr, "warn: tokenizer file not found under model dir '%s'\n", model_dir_eff);
    }
  }

  /* Optional per-record decode (outside the timed window). */
  if (opt.print_text && tokenizer_path_heap && *tokenizer_path_heap) {
    for (int rr = 0; rr < rounds; ++rr) {
      for (size_t pi = 0; pi < n_prompts; ++pi) {
        prompt_round_record_t *r = &recs[(size_t)rr * n_prompts + pi];
        if (r->tokens.ids && r->tokens.n > 0) {
          char *txt = NULL;
          if (decode_tokens_best_effort(tokenizer_path_heap, r->tokens.ids, r->tokens.n, &txt) == 0) {
            r->decoded_text = txt;
          }
        }
      }
    }
  }

  print_json_result(tokens_generated_total,
                    tokens_to_print,
                    (t1 - t0),
                    prefill_s_total,
                    decode_s_total,
                    kv_hits_round,
                    kv_miss_round,
                    rss_mib,
                    decoded_legacy,
                    (opt.print_text ? tokenizer_path_heap : NULL),
                    opt.expected_tokens_file,
                    &prompts,
                    rounds,
                    recs);

  free(decoded_legacy);

  if (mt_ok) model_touch_close(&mt);

  for (size_t i = 0; i < (size_t)rounds * n_prompts; ++i) prompt_round_record_free(&recs[i]);
  free(recs);

  expected_tokens_table_free(&expected);
  prompt_list_free(&prompts);

  free(tokens);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  free(tokenizer_path_heap);
  return 0;
}
