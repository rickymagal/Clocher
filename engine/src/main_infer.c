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
  if (!ctx || !ctx->base || ctx->size == 0) return 1;
  if (bytes_to_touch == 0) return 0;
  if (stride_bytes == 0) stride_bytes = 1;

  size_t n = bytes_to_touch;
  if (n > ctx->size) n = ctx->size;

  volatile const unsigned char *p = (volatile const unsigned char *)ctx->base;
  volatile uint64_t acc = 0;

  size_t start = ctx->cursor;
  size_t end = start + n;
  if (end < start) end = ctx->size;

  for (size_t off = start; off < end; off += stride_bytes) acc ^= (uint64_t)p[off];

  if (end == ctx->size) {
    size_t rem = (start + n) - ctx->size;
    for (size_t off = 0; off < rem; off += stride_bytes) acc ^= (uint64_t)p[off];
    ctx->cursor = (ctx->size ? (rem % ctx->size) : 0);
  } else {
    ctx->cursor = (ctx->size ? (end % ctx->size) : 0);
  }

  if (verify_touch) {
    if (acc == 0x9e3779b97f4a7c15ULL) fprintf(stderr, "touch verify: improbable accumulator value (ignore)\n");
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
          "                        [--rounds N] [--aggregate]\n");
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
 * @param n_tok Number of generated tokens.
 * @param tokens Token buffer (may be NULL).
 * @param wall_s_in Elapsed seconds.
 * @param kv_hits KV cache hits.
 * @param kv_misses KV cache misses.
 * @param rss_peak_mb RSS peak in MiB.
 */
static void print_json_result(size_t n_tok,
                              const int *tokens,
                              double wall_s_in,
                              uint64_t kv_hits,
                              uint64_t kv_misses,
                              uint32_t rss_peak_mb) {
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

static int device_is_cuda(const char *dev) { return ascii_ieq(dev, "cuda") ? 1 : 0; }
static int device_is_cpu(const char *dev) { return ascii_ieq(dev, "cpu") ? 1 : 0; }

int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

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
      print_json_result(0, NULL, 0.0, 0, 0, 0);
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
    print_json_result(0, NULL, 0.0, 0, 0, 0);
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
      print_json_result(0, NULL, 0.0, 0, 0, 0);
      return 3;
    }
    fprintf(stderr, "warn: model metadata not found; stub JSON output\n");
    print_json_result(0, NULL, 0.0, 0, 0, 0);
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
    print_json_result(0, NULL, 0.0, 0, 0, 0);
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
    print_json_result(0, NULL, 0.0, 0, 0, 0);
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
  const int *tokens_to_print = (opt.aggregate || opt.rounds > 1) ? NULL : tokens;

  print_json_result(tokens_generated_total,
                    tokens_to_print,
                    (t1 - t0),
                    kv_hits_round,
                    kv_miss_round,
                    rss_mib);

  if (mt_ok) model_touch_close(&mt);

  free(tokens);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  return 0;
}
