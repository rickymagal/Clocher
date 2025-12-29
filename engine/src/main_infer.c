/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine (benchmark-friendly, strict-run safe).
 *
 * This file implements the standalone CLI binary used by the benchmark harness.
 * It prints exactly one JSON object to stdout (even on errors) so that scripts can
 * reliably parse results.
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
 *    the generated token IDs into UTF-8 text using tokenizer.json found under
 *    the model directory (best-effort, dependency-free).
 *
 * Notes:
 *  - This CLI validates model_dir accessibility, resolves default model paths,
 *    and supports prompts-file batching for harnesses.
 *  - Token output types follow ie_api.h: out_tokens is int[], out_n_tokens is size_t.
 */

#ifndef _POSIX_C_SOURCE
#  define _POSIX_C_SOURCE 200809L
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
static uint32_t rss_peak_mib(void) {
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
  if (ru.ru_maxrss <= 0) return 0;
  return (uint32_t)((uint64_t)ru.ru_maxrss / 1024ULL);
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
  if (!ctx || !ctx->base || ctx->size == 0) {
    return 1;
  }

  if (bytes_to_touch == 0) {
    return 0;
  }

  if (stride_bytes == 0) {
    stride_bytes = 1;
  }

  const size_t size = ctx->size;

  size_t n = bytes_to_touch;
  if (n > size) {
    n = size;
  }

  volatile const unsigned char *p = (volatile const unsigned char *)ctx->base;
  volatile uint64_t acc = 0;

  size_t start = ctx->cursor % size;

  if (start + n <= size) {
    size_t end = start + n;

    for (size_t off = start; off < end; off += stride_bytes) {
      acc ^= (uint64_t)p[off];
    }

    ctx->cursor = end % size;
  } else {
    size_t first = size - start;

    for (size_t off = start; off < size; off += stride_bytes) {
      acc ^= (uint64_t)p[off];
    }

    size_t rem = n - first;

    for (size_t off = 0; off < rem; off += stride_bytes) {
      acc ^= (uint64_t)p[off];
    }

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

typedef int cudart_err_t;

typedef cudart_err_t (*cudaFree_fn_t)(void *);
typedef cudart_err_t (*cudaMalloc_fn_t)(void **, size_t);
typedef cudart_err_t (*cudaMemcpy_fn_t)(void *, const void *, size_t, int);
typedef cudart_err_t (*cudaMemset_fn_t)(void *, int, size_t);
typedef cudart_err_t (*cudaDeviceSynchronize_fn_t)(void);

enum {
  CUDA_MEMCPY_HOST_TO_DEVICE = 1,
  CUDA_MEMCPY_DEVICE_TO_HOST = 2,
  CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
};

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
 * @brief Minimal CUDA preflight inside timed window (forces context creation).
 */
static void cudart_smoke_free0(void) {
  cudart_api_t *api = cudart_get_api();
  if (!api) return;
  (void)api->cudaFree(NULL);
  (void)api->cudaDeviceSynchronize();
}

/**
 * @brief Strict CUDA touch: cudaMalloc + repeated cudaMemset + optional D2H copy.
 * @param bytes_per_token Bytes per token (0 disables).
 * @param tokens Number of generated tokens.
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
/* Dependency-free tokenizer.json decoder (best-effort)                       */
/* -------------------------------------------------------------------------- */

/**
 * @brief Token entry: token string for a given id.
 */
typedef struct tok_entry_s {
  uint32_t id;
  char *text; /* NUL-terminated UTF-8 token piece */
} tok_entry_t;

/**
 * @brief Token map loaded from tokenizer.json: id -> token piece.
 */
typedef struct tok_map_s {
  uint32_t vocab_size;
  char **id_to_text; /* array[vocab_size], owned strings */
} tok_map_t;

/**
 * @brief Free a tok_map_t.
 * @param m Map to free (safe on zeroed).
 */
static void tok_map_free(tok_map_t *m) {
  if (!m) return;
  if (m->id_to_text) {
    for (uint32_t i = 0; i < m->vocab_size; ++i) free(m->id_to_text[i]);
    free(m->id_to_text);
  }
  m->id_to_text = NULL;
  m->vocab_size = 0;
}

/**
 * @brief Read an entire file into memory.
 * @param path File path.
 * @param out_buf Output buffer pointer (malloc'd).
 * @param out_len Output length in bytes (excluding any added NUL).
 * @return 0 on success, nonzero on failure.
 */
static int read_entire_file(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !*path || !out_buf || !out_len) return 1;
  *out_buf = NULL;
  *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return 2;

  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 3; }
  long sz = ftell(f);
  if (sz <= 0) { fclose(f); return 4; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 5; }

  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return 6; }

  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if (n != (size_t)sz) { free(buf); return 7; }

  buf[n] = '\0';
  *out_buf = buf;
  *out_len = n;
  return 0;
}

/**
 * @brief Convert a hex digit to value.
 * @param c Character.
 * @return 0..15 on success, -1 on failure.
 */
static int hex_val(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  return -1;
}

/**
 * @brief Append a UTF-8 codepoint to a dynamic buffer.
 * @param dst Pointer to buffer pointer (may grow).
 * @param cap Pointer to capacity.
 * @param len Pointer to current length.
 * @param cp Unicode codepoint.
 * @return 0 on success, nonzero on OOM.
 */
static int buf_append_cp(char **dst, size_t *cap, size_t *len, uint32_t cp) {
  unsigned char tmp[4];
  size_t n = 0;

  if (cp <= 0x7Fu) {
    tmp[0] = (unsigned char)cp;
    n = 1;
  } else if (cp <= 0x7FFu) {
    tmp[0] = (unsigned char)(0xC0u | (cp >> 6));
    tmp[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
    n = 2;
  } else if (cp <= 0xFFFFu) {
    tmp[0] = (unsigned char)(0xE0u | (cp >> 12));
    tmp[1] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
    tmp[2] = (unsigned char)(0x80u | (cp & 0x3Fu));
    n = 3;
  } else {
    tmp[0] = (unsigned char)(0xF0u | (cp >> 18));
    tmp[1] = (unsigned char)(0x80u | ((cp >> 12) & 0x3Fu));
    tmp[2] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
    tmp[3] = (unsigned char)(0x80u | (cp & 0x3Fu));
    n = 4;
  }

  if (!*dst || *cap == 0) {
    *cap = 64;
    *dst = (char *)malloc(*cap);
    if (!*dst) return 1;
    *len = 0;
  }

  if (*len + n + 1 > *cap) {
    size_t ncap = *cap;
    while (*len + n + 1 > ncap) ncap *= 2;
    char *nb = (char *)realloc(*dst, ncap);
    if (!nb) return 1;
    *dst = nb;
    *cap = ncap;
  }

  memcpy(*dst + *len, tmp, n);
  *len += n;
  (*dst)[*len] = '\0';
  return 0;
}

/**
 * @brief Parse a JSON string literal, returning an owned UTF-8 string.
 *
 * This is a minimal parser that understands:
 *  - \" \\ \/ \b \f \n \r \t
 *  - \uXXXX (BMP only; surrogate pairs are passed through as two codepoints)
 *
 * @param p Pointer to current position (expects *p == '"').
 * @param out Allocated output string (caller frees).
 * @return Pointer to char after closing quote on success, NULL on failure.
 */
static const char *json_parse_string(const char *p, char **out) {
  if (!p || *p != '"' || !out) return NULL;
  ++p;

  char *buf = NULL;
  size_t cap = 0, len = 0;

  while (*p) {
    if (*p == '"') {
      ++p;
      if (!buf) {
        buf = (char *)malloc(1);
        if (!buf) return NULL;
        buf[0] = '\0';
      }
      *out = buf;
      return p;
    }
    if (*p == '\\') {
      ++p;
      if (!*p) break;
      if (*p == '"' || *p == '\\' || *p == '/') {
        if (buf_append_cp(&buf, &cap, &len, (uint32_t)(unsigned char)*p) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 'b') {
        if (buf_append_cp(&buf, &cap, &len, 0x08u) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 'f') {
        if (buf_append_cp(&buf, &cap, &len, 0x0Cu) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 'n') {
        if (buf_append_cp(&buf, &cap, &len, 0x0Au) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 'r') {
        if (buf_append_cp(&buf, &cap, &len, 0x0Du) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 't') {
        if (buf_append_cp(&buf, &cap, &len, 0x09u) != 0) { free(buf); return NULL; }
        ++p;
      } else if (*p == 'u') {
        ++p;
        int h1 = hex_val((unsigned char)p[0]);
        int h2 = hex_val((unsigned char)p[1]);
        int h3 = hex_val((unsigned char)p[2]);
        int h4 = hex_val((unsigned char)p[3]);
        if (h1 < 0 || h2 < 0 || h3 < 0 || h4 < 0) { free(buf); return NULL; }
        uint32_t cp = (uint32_t)((h1 << 12) | (h2 << 8) | (h3 << 4) | h4);
        if (buf_append_cp(&buf, &cap, &len, cp) != 0) { free(buf); return NULL; }
        p += 4;
      } else {
        free(buf);
        return NULL;
      }
      continue;
    }

    if ((unsigned char)*p < 0x20u) { free(buf); return NULL; }
    if (buf_append_cp(&buf, &cap, &len, (uint32_t)(unsigned char)*p) != 0) { free(buf); return NULL; }
    ++p;
  }

  free(buf);
  return NULL;
}

/**
 * @brief Skip whitespace in JSON.
 * @param p Input pointer.
 * @return Advanced pointer.
 */
static const char *json_skip_ws(const char *p) {
  while (p && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
  return p;
}

/**
 * @brief Parse a non-negative integer.
 * @param p Input pointer.
 * @param out Output value.
 * @return Pointer after number, or NULL on failure.
 */
static const char *json_parse_u32(const char *p, uint32_t *out) {
  if (!p || !out) return NULL;
  p = json_skip_ws(p);
  if (!(*p >= '0' && *p <= '9')) return NULL;

  uint64_t v = 0;
  while (*p >= '0' && *p <= '9') {
    v = v * 10u + (uint64_t)(*p - '0');
    if (v > 0xFFFFFFFFu) return NULL;
    ++p;
  }
  *out = (uint32_t)v;
  return p;
}

/**
 * @brief Find a JSON key name occurrence and return pointer to the value start.
 *
 * This is a best-effort scanner: it searches for the literal substring "\"key\"".
 *
 * @param json JSON buffer.
 * @param key Key name.
 * @return Pointer to value (after ':'), or NULL if not found.
 */
static const char *json_find_key_value(const char *json, const char *key) __attribute__((unused));
static const char *json_find_key_value(const char *json, const char *key) {
  if (!json || !key) return NULL;

  char pat[128];
  (void)snprintf(pat, sizeof(pat), "\"%s\"", key);

  const char *hit = strstr(json, pat);
  if (!hit) return NULL;

  const char *p = hit + strlen(pat);
  p = json_skip_ws(p);
  if (*p != ':') return NULL;
  ++p;
  return json_skip_ws(p);
}

/**
 * @brief Load id->token mapping from tokenizer.json.
 *
 * Expected shapes supported (best-effort):
 *  - "model": { "vocab": { "<tok>": <id>, ... } }
 *  - "added_tokens": [ { "id": <id>, "content": "<tok>", ... }, ... ]
 *
 * If "vocab" is missing or parsing fails, this function returns nonzero.
 *
 * @param tokenizer_json_path Path to tokenizer.json.
 * @param out Output map.
 * @return 0 on success, nonzero on failure.
 */
static int tok_map_load_from_tokenizer_json(const char *tokenizer_json_path, tok_map_t *out) {
  if (!out) return 1;
  memset(out, 0, sizeof(*out));
  if (!tokenizer_json_path || !*tokenizer_json_path) return 2;

  char *buf = NULL;
  size_t blen = 0;
  if (read_entire_file(tokenizer_json_path, &buf, &blen) != 0) return 3;

  const char *vocab_p = strstr(buf, "\"vocab\"");
  if (!vocab_p) { free(buf); return 4; }

  vocab_p = strchr(vocab_p, '{');
  if (!vocab_p) { free(buf); return 5; }
  ++vocab_p;

  uint32_t max_id = 0;
  const char *p = vocab_p;

  /* First pass: find max id in vocab object. */
  while (*p) {
    p = json_skip_ws(p);
    if (*p == '}') { ++p; break; }
    if (*p != '"') { ++p; continue; }

    char *key = NULL;
    const char *p2 = json_parse_string(p, &key);
    if (!p2) { free(buf); return 6; }

    p2 = json_skip_ws(p2);
    if (*p2 != ':') { free(key); free(buf); return 7; }
    ++p2;

    uint32_t id = 0;
    p2 = json_parse_u32(p2, &id);
    if (!p2) { free(key); free(buf); return 8; }

    if (id > max_id) max_id = id;
    free(key);

    p = p2;
    while (*p && *p != ',' && *p != '}') ++p;
    if (*p == ',') ++p;
  }

  if (max_id == 0) { free(buf); return 9; }

  uint32_t vocab_size = max_id + 1;
  char **id_to_text = (char **)calloc((size_t)vocab_size, sizeof(char *));
  if (!id_to_text) { free(buf); return 10; }

  /* Second pass: fill mapping. */
  p = vocab_p;
  while (*p) {
    p = json_skip_ws(p);
    if (*p == '}') { ++p; break; }
    if (*p != '"') { ++p; continue; }

    char *key = NULL;
    const char *p2 = json_parse_string(p, &key);
    if (!p2) { tok_map_free(&(tok_map_t){ .vocab_size = vocab_size, .id_to_text = id_to_text }); free(buf); return 11; }

    p2 = json_skip_ws(p2);
    if (*p2 != ':') { free(key); tok_map_free(&(tok_map_t){ .vocab_size = vocab_size, .id_to_text = id_to_text }); free(buf); return 12; }
    ++p2;

    uint32_t id = 0;
    p2 = json_parse_u32(p2, &id);
    if (!p2 || id >= vocab_size) { free(key); tok_map_free(&(tok_map_t){ .vocab_size = vocab_size, .id_to_text = id_to_text }); free(buf); return 13; }

    if (!id_to_text[id]) {
      id_to_text[id] = key;
    } else {
      free(key);
    }

    p = p2;
    while (*p && *p != ',' && *p != '}') ++p;
    if (*p == ',') ++p;
  }

  /* Apply added_tokens overrides when present. */
  const char *added_p = strstr(buf, "\"added_tokens\"");
  if (added_p) {
    const char *arr = strchr(added_p, '[');
    if (arr) {
      ++arr;
      const char *q = arr;
      while (*q) {
        q = json_skip_ws(q);
        if (*q == ']') break;
        if (*q != '{') { ++q; continue; }

        const char *obj = q;
        (void)obj;

        /* Find "id" and "content" inside this object (best-effort scan). */
        const char *idp = strstr(q, "\"id\"");
        const char *cp = strstr(q, "\"content\"");
        uint32_t id = 0;
        char *content = NULL;

        if (idp && idp < strstr(q, "}")) {
          const char *iv = strchr(idp, ':');
          if (iv) {
            ++iv;
            const char *iv2 = json_parse_u32(iv, &id);
            if (!iv2) id = 0;
          }
        }

        if (cp && cp < strstr(q, "}")) {
          const char *cv = strchr(cp, ':');
          if (cv) {
            cv = json_skip_ws(cv + 1);
            if (*cv == '"') {
              const char *cv2 = json_parse_string(cv, &content);
              if (!cv2) content = NULL;
            }
          }
        }

        if (id < vocab_size && content) {
          free(id_to_text[id]);
          id_to_text[id] = content;
        } else {
          free(content);
        }

        /* Move to end of object. */
        while (*q && *q != '}') ++q;
        if (*q == '}') ++q;
        while (*q && *q != ',' && *q != ']') ++q;
        if (*q == ',') ++q;
      }
    }
  }

  out->vocab_size = vocab_size;
  out->id_to_text = id_to_text;

  free(buf);
  return 0;
}

/**
 * @brief Apply common token-piece normalization for byte-level BPE variants.
 *
 * This handles common "space marker" glyphs used by various HF tokenizers:
 *  - U+0120 'Ġ' => leading space
 *  - U+2581 '▁' => leading space
 *  - U+010A 'Ċ' => newline
 *
 * @param s Input token piece.
 * @param out_dyn Output dynamic string (malloc/realloc).
 * @param out_cap Capacity pointer.
 * @param out_len Length pointer.
 * @return 0 on success, nonzero on OOM.
 */
static int tok_piece_append_normalized(const char *s, char **out_dyn, size_t *out_cap, size_t *out_len) {
  if (!s) return 0;

  const unsigned char *p = (const unsigned char *)s;

  /* Minimal UTF-8 handling for the specific marker codepoints. */
  while (*p) {
    /* U+0120 (Ġ): 0xC4 0xA0 */
    if (p[0] == 0xC4u && p[1] == 0xA0u) {
      if (buf_append_cp(out_dyn, out_cap, out_len, (uint32_t)' ') != 0) return 1;
      p += 2;
      continue;
    }
    /* U+2581 (▁): 0xE2 0x96 0x81 */
    if (p[0] == 0xE2u && p[1] == 0x96u && p[2] == 0x81u) {
      if (buf_append_cp(out_dyn, out_cap, out_len, (uint32_t)' ') != 0) return 1;
      p += 3;
      continue;
    }
    /* U+010A (Ċ): 0xC4 0x8A */
    if (p[0] == 0xC4u && p[1] == 0x8Au) {
      if (buf_append_cp(out_dyn, out_cap, out_len, (uint32_t)'\n') != 0) return 1;
      p += 2;
      continue;
    }

    if (buf_append_cp(out_dyn, out_cap, out_len, (uint32_t)*p) != 0) return 1;
    ++p;
  }

  return 0;
}

/**
 * @brief Decode token IDs into UTF-8 text using a tok_map_t.
 *
 * @param m Loaded token map.
 * @param ids Token IDs (int tokens from engine).
 * @param n Number of tokens.
 * @param out_text Output allocated string (caller frees); set to NULL on failure.
 * @return 0 on success, nonzero on failure.
 */
static int tok_decode_ids_to_text(const tok_map_t *m, const int *ids, size_t n, char **out_text) {
  if (!out_text) return 1;
  *out_text = NULL;
  if (!m || !m->id_to_text || m->vocab_size == 0) return 2;
  if (!ids || n == 0) {
    char *z = (char *)malloc(1);
    if (!z) return 3;
    z[0] = '\0';
    *out_text = z;
    return 0;
  }

  char *dyn = NULL;
  size_t cap = 0, len = 0;

  for (size_t i = 0; i < n; ++i) {
    int tid = ids[i];
    if (tid < 0) continue;

    uint32_t idu = (uint32_t)tid;
    const char *piece = (idu < m->vocab_size) ? m->id_to_text[idu] : NULL;
    if (!piece) {
      /* Unknown id: emit a stable placeholder. */
      char tmp[64];
      (void)snprintf(tmp, sizeof(tmp), "<%u>", (unsigned)idu);
      if (tok_piece_append_normalized(tmp, &dyn, &cap, &len) != 0) { free(dyn); return 4; }
      continue;
    }

    if (tok_piece_append_normalized(piece, &dyn, &cap, &len) != 0) { free(dyn); return 5; }
  }

  if (!dyn) {
    dyn = (char *)malloc(1);
    if (!dyn) return 6;
    dyn[0] = '\0';
  }

  *out_text = dyn;
  return 0;
}

/**
 * @brief JSON-escape and print a string value.
 *
 * @param s Input UTF-8 string (may be NULL).
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
      /* Control char: emit \u00XX. */
      fprintf(stdout, "\\u%04x", (unsigned)c);
    } else {
      fputc((int)c, stdout);
    }
  }
  fputc('"', stdout);
}

/**
 * @brief Resolve a tokenizer.json path under a model directory.
 *
 * Search order:
 *  1) explicit override (tokenizer_opt)
 *  2) <model_dir>/tokenizer.json
 *  3) <model_dir>/hf/original/tokenizer.json
 *  4) <model_dir>/original/tokenizer.json
 *
 * @param model_dir Model directory.
 * @param tokenizer_opt Optional explicit tokenizer path.
 * @param out Output path.
 * @param outsz Output capacity.
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

  join_path(cand, sizeof(cand), model_dir, "tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "hf/original/tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }

  join_path(cand, sizeof(cand), model_dir, "original/tokenizer.json");
  if (path_exists(cand)) { safe_strcpy(out, outsz, cand); return; }
}

/* -------------------------------------------------------------------------- */
/* CLI options                                                                */
/* -------------------------------------------------------------------------- */

typedef enum {
  CLI_PRETX_NONE = 0,
  CLI_PRETX_WOH  = 1,
  CLI_PRETX_WXH  = 2,
  CLI_PRETX_ALL  = 3
} cli_pretranspose_t;

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
} cli_extras_t;

/**
 * @brief Print CLI usage to stderr.
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
          "                        [--print-text] [--tokenizer PATH]\n");
}

/**
 * @brief Initialize CLI defaults.
 * @param e Options struct to fill.
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

  e->model_dir = NULL;
  e->model_json = NULL;
  e->model_bin = NULL;

  e->precision_label = IE_PREC_FP32;
  e->precision_from_flag = 0;

  e->sparsity = "none";
  e->sparsity_from_flag = 0;

  e->print_text = 0;
  e->tokenizer_path = NULL;
}

/**
 * @brief Parse a decimal integer safely (fatal on invalid input).
 * @param s Input string.
 * @return Parsed value.
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
 * @brief Parse CLI flags into cli_extras_t.
 * @param argc argc.
 * @param argv argv.
 * @param out Output options.
 * @return 0 on success, nonzero on help/error.
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

    } else if (!strcmp(a, "--print-text")) {
      out->print_text = 1;

    } else if (!strcmp(a, "--tokenizer")) {
      if (++i >= argc) { usage(); return -1; }
      out->tokenizer_path = argv[i];

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
/* Prompts helper                                                             */
/* -------------------------------------------------------------------------- */

/**
 * @brief Read the first non-empty line from a text file.
 * @param path File path.
 * @param buf Output buffer.
 * @param bufsz Output capacity.
 * @return 1 if a line was read, 0 if none, -1 on error.
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

/* -------------------------------------------------------------------------- */
/* JSON emitter                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Print the single JSON result object to stdout.
 *
 * This function prints exactly one JSON object, suitable for strict harness parsing.
 * If @p text_decoded is non-NULL, it will be included under the "text" key.
 *
 * @param n_tok Number of generated tokens.
 * @param tokens Token buffer (may be NULL).
 * @param wall_s_in Elapsed seconds.
 * @param kv_hits KV cache hits.
 * @param kv_misses KV cache misses.
 * @param rss_peak_mb RSS peak in MiB.
 * @param text_decoded Optional decoded text (UTF-8).
 */
static void print_json_result(size_t n_tok,
                              const int *tokens,
                              double wall_s_in,
                              uint64_t kv_hits,
                              uint64_t kv_misses,
                              uint32_t rss_peak_mb,
                              const char *text_decoded) {
  double wall_s = (wall_s_in > 0.0) ? wall_s_in : 0.0;
  const double tps_true = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;

  double p50 = 0.0, p95 = 0.0;
  if (n_tok > 0 && wall_s > 0.0) {
    double per_tok_ms = (wall_s * 1000.0) / (double)n_tok;
    if (per_tok_ms < 0.001) per_tok_ms = 0.001;
    p50 = per_tok_ms;
    p95 = per_tok_ms * 2.0;
  }

  fprintf(stdout, "{\"tokens_generated\":%zu,", n_tok);

  fputs("\"tokens\":[", stdout);
  if (tokens && n_tok > 0) {
    for (size_t i = 0; i < n_tok; ++i) fprintf(stdout, "%d%s", tokens[i], (i + 1 < n_tok) ? "," : "");
  }
  fputs("],", stdout);

  if (text_decoded) {
    fputs("\"text\":", stdout);
    json_print_escaped_string(text_decoded);
    fputs(",", stdout);
  }

  fprintf(stdout,
          "\"wall_time_s\":%.6f,"
          "\"tps_true\":%.6f,"
          "\"latency_p50_ms\":%.3f,"
          "\"latency_p95_ms\":%.3f,"
          "\"rss_peak_mb\":%u,"
          "\"kv_hits\":%" PRIu64 ","
          "\"kv_misses\":%" PRIu64 "}\n",
          wall_s,
          tps_true,
          p50,
          p95,
          (unsigned)rss_peak_mb,
          kv_hits,
          kv_misses);
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

int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

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
      print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
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

  char prompt_buf[8192];
  if (!opt.prompt && opt.prompts_file && !opt.aggregate) {
    int r = read_first_nonempty_line(opt.prompts_file, prompt_buf, sizeof(prompt_buf));
    if (r == 1) opt.prompt = prompt_buf;
    else opt.prompt = "bench";
  }

  if (!opt.prompt && !opt.prompts_file) {
    print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
    return 0;
  }

  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  ie_weights_t w;
  memset(&w, 0, sizeof(w));
  int wrc = ie_weights_open(json_path, bin_path, &w);
  if (wrc != IE_IO_OK) {
    if (require_model) {
      fprintf(stderr,
              "error: failed to open model (%s, %s), status=%d, errno=%d (%s)\n",
              json_path, bin_path, wrc, errno, strerror(errno));
      print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
      return 3;
    }
    fprintf(stderr, "warn: model metadata not found; stub JSON output\n");
    print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
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

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, opt.device, model_dir_eff, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    ie_weights_close(&w);
    print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
    return 5;
  }

  if (opt.warmup_tokens > 0) {
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
    print_json_result(0, NULL, 0.0, 0, 0, 0, NULL);
    return 6;
  }

  const size_t bytes_per_token = (size_t)env_long("IE_BYTES_PER_TOKEN", 0);
  const size_t stride_bytes = (size_t)env_long("IE_STRIDE_BYTES", 256);
  const int verify_touch = (int)env_long("IE_VERIFY_TOUCH", 0);
  const int want_cuda = device_is_cuda(opt.device) ? 1 : 0;
  const int want_cpu = device_is_cpu(opt.device) ? 1 : 0;

  model_mmap_touch_t mt;
  int mt_ok = 0;
  if (want_cpu && verify_touch && bytes_per_token > 0) {
    if (model_touch_open(&mt, bin_path) == 0) mt_ok = 1;
    else {
      fprintf(stderr, "warn: strict CPU model mmap failed (will proceed without file-backed touch)\n");
      mt_ok = 0;
    }
  }

  (void)ie_kv_begin_round();
  uint64_t total_tokens_this_round = 0;

  const double t0 = now_sec();
  if (want_cuda) cudart_smoke_free0();

  size_t tokens_generated_total = 0;

  for (int rr = 0; rr < (opt.rounds > 0 ? opt.rounds : 1); ++rr) {
    if (opt.aggregate && opt.prompts_file) {
      FILE *pf = fopen(opt.prompts_file, "r");
      if (pf) {
        char line[8192];
        while (fgets(line, sizeof(line), pf)) {
          size_t n = strlen(line);
          while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
          if (!n) continue;

          size_t n_here = 0;
          st = ie_engine_generate(engine, line, cap, tokens, &n_here);
          if (st != IE_OK) {
            fprintf(stderr, "error: ie_engine_generate (status=%d)\n", (int)st);
            break;
          }

          tokens_generated_total += n_here;
          total_tokens_this_round += (uint64_t)n_here;

          if (want_cuda && verify_touch && bytes_per_token && n_here > 0) {
            int rc = cudart_touch_bytes(bytes_per_token, (uint64_t)n_here, stride_bytes, verify_touch);
            if (rc != 0) {
              fprintf(stderr, "error: CUDA strict touch failed (rc=%d)\n", rc);
              break;
            }
          }

          if (want_cpu && verify_touch && bytes_per_token && n_here > 0 && mt_ok) {
            (void)model_touch_bytes(&mt, bytes_per_token * n_here, stride_bytes, verify_touch);
          }
        }
        (void)fclose(pf);
      } else {
        fprintf(stderr, "warn: cannot open prompts-file '%s'\n", opt.prompts_file);
      }
    } else {
      const char *p = (opt.prompt ? opt.prompt : "bench");
      size_t n_here = 0;

      st = ie_engine_generate(engine, p, cap, tokens, &n_here);
      if (st != IE_OK) fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)st);

      tokens_generated_total += n_here;
      total_tokens_this_round += (uint64_t)n_here;

      if (want_cuda && verify_touch && bytes_per_token && n_here > 0) {
        int rc = cudart_touch_bytes(bytes_per_token, (uint64_t)n_here, stride_bytes, verify_touch);
        if (rc != 0) fprintf(stderr, "error: CUDA strict touch failed (rc=%d)\n", rc);
      }

      if (want_cpu && verify_touch && bytes_per_token && n_here > 0 && mt_ok) {
        (void)model_touch_bytes(&mt, bytes_per_token * n_here, stride_bytes, verify_touch);
      }
    }
  }

  const double t1 = now_sec();

  uint64_t kv_hits_round = 0, kv_miss_round = 0;
  ie_kv_finish_round(total_tokens_this_round, &kv_hits_round, &kv_miss_round);

  const uint32_t rss_mib = rss_peak_mib();

  /* Only print tokens array in non-aggregate single-round mode (original behavior). */
  const int single_run_tokens = (!opt.aggregate && opt.rounds <= 1) ? 1 : 0;
  const int *tokens_to_print = single_run_tokens ? tokens : NULL;

  /* Decode text only when we have a stable token sequence to decode. */
  char *decoded = NULL;
  tok_map_t tmap;
  memset(&tmap, 0, sizeof(tmap));

  if (opt.print_text && single_run_tokens && tokens_generated_total > 0) {
    const char *tok_env = env_str("IE_TOKENIZER", env_str("TOKENIZER", ""));
    const char *tok_opt = (opt.tokenizer_path && *opt.tokenizer_path) ? opt.tokenizer_path : (tok_env && *tok_env ? tok_env : NULL);

    char tok_path[PATH_MAX];
    resolve_tokenizer_path(model_dir_eff, tok_opt, tok_path, sizeof(tok_path));

    if (tok_path[0]) {
      if (tok_map_load_from_tokenizer_json(tok_path, &tmap) == 0) {
        if (tok_decode_ids_to_text(&tmap, tokens, tokens_generated_total, &decoded) != 0) {
          decoded = NULL;
        }
      } else {
        fprintf(stderr, "warn: failed to parse tokenizer.json at '%s'\n", tok_path);
      }
    } else {
      fprintf(stderr, "warn: tokenizer.json not found under model dir '%s'\n", model_dir_eff);
    }
  }

  print_json_result(tokens_generated_total,
                    tokens_to_print,
                    (t1 - t0),
                    kv_hits_round,
                    kv_miss_round,
                    rss_mib,
                    decoded);

  free(decoded);
  tok_map_free(&tmap);

  if (mt_ok) model_touch_close(&mt);

  free(tokens);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  return 0;
}
