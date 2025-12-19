/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine (benchmark-friendly).
 *
 * This translation unit implements the standalone CLI binary used by the
 * benchmark harness. It is designed to always emit a single JSON object on
 * stdout (even in stub mode), so that external scripts can reliably parse
 * per-run results.
 *
 * Key behaviors:
 *  - CI/Stub mode (default): does not require model files; emits valid JSON.
 *  - Strict mode (IE_REQUIRE_MODEL=1): requires model files to open.
 *  - Benchmark timing window includes only:
 *      - ie_engine_generate(...)
 *      - optional per-token "work-touch" loop
 *
 * Model path resolution:
 *  - --model-json / --model-bin override everything.
 *    If relative and --model-dir is provided, they are resolved relative to it.
 *  - Otherwise, defaults are chosen from --model-dir (or inferred model dir):
 *      - If effective precision is int4/int4w:
 *          prefer model.ie.compat.json + model.q4.bin (if present),
 *          otherwise fall back to model.ie.json + model.ie.bin.
 *      - Otherwise:
 *          use model.ie.json + model.ie.bin.
 *
 * Runtime knobs (exported as env vars):
 *  - GEMV prefetch/streaming:
 *      --pf-w N          -> IE_GEMV_PFDIST_W
 *      --pf-x N          -> IE_GEMV_PFDIST_X
 *      --force-ntw 0|1   -> IE_GEMV_FORCE_NTW
 *
 *  - Sparsity:
 *      --sparsity none|block|auto -> IE_SPARSITY (when provided)
 *
 *  - Dedup (loader/runtime consumes these; main_infer.c only exports knobs):
 *      --dedup 0|1               -> IE_DEDUP
 *      --dedup-policy ...        -> IE_DEDUP_POLICY
 *      --dedup-cache-mb N        -> IE_DEDUP_CACHE_MB
 *      --dedup-hot-bytes N       -> IE_DEDUP_HOT_BYTES
 *      --dedup-hot-list STR      -> IE_DEDUP_HOT_LIST
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ie_engine.h"
#include "ie_io.h"
#include "ie_kv_instrumentation.h"
#include "util_logging.h"
#include "util_metrics.h"

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
/* Time utilities                                                             */
/* -------------------------------------------------------------------------- */
/**
 * @brief Get a monotonic wall-clock timestamp in seconds.
 * @return Seconds as a double, suitable for interval measurements.
 */
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* -------------------------------------------------------------------------- */
/* Env helpers                                                                */
/* -------------------------------------------------------------------------- */
/**
 * @brief Parse an environment variable as a long with a default fallback.
 * @param name Environment variable name (e.g., "IE_STRIDE_BYTES").
 * @param defv Default value when unset or invalid.
 * @return Parsed value, or @p defv on error/unset.
 */
static long env_long(const char *name, long defv) {
  const char *s = getenv(name);
  if (!s || !*s) return defv;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  return (end && *end == '\0') ? v : defv;
}

/**
 * @brief Get environment variable as string with a default fallback.
 * @param name Environment variable name.
 * @param defv Default string when unset/empty.
 * @return Pointer to environment value or @p defv.
 */
static const char *env_str(const char *name, const char *defv) {
  const char *s = getenv(name);
  return (s && *s) ? s : defv;
}

/**
 * @brief Check whether a filesystem path exists.
 *
 * This is a tiny convenience wrapper around stat(2). It is intentionally
 * conservative: it returns 1 if `stat()` succeeds, and 0 otherwise.
 *
 * @param p Null-terminated path.
 * @return 1 if the path exists, 0 otherwise.
 */
static int path_exists(const char *p) {
  if (!p || !*p) return 0;
  struct stat st;
  return (stat(p, &st) == 0);
}

/**
 * @brief Check whether a filesystem path is accessible as a directory.
 * @param p Path to check.
 * @return 1 if accessible and is a directory, 0 otherwise.
 */
static int dir_accessible(const char *p) {
  if (!p || !*p) return 0;
  struct stat st;
  if (stat(p, &st) != 0) return 0;
  if (!S_ISDIR(st.st_mode)) return 0;
  return (access(p, R_OK | X_OK) == 0);
}

/* -------------------------------------------------------------------------- */
/* ASCII utilities                                                            */
/* -------------------------------------------------------------------------- */
/**
 * @brief Case-insensitive ASCII equality check (NULL-safe).
 * @param a First string (may be NULL).
 * @param b Second string (may be NULL).
 * @return 1 if equal ignoring ASCII case; 0 otherwise.
 */
static int ascii_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    unsigned char ca = (unsigned char)*a;
    unsigned char cb = (unsigned char)*b;
    unsigned char la = (ca >= 'A' && ca <= 'Z') ? (unsigned char)(ca + 32) : ca;
    unsigned char lb = (cb >= 'A' && cb <= 'Z') ? (unsigned char)(cb + 32) : cb;
    if (la != lb) return 0;
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

/**
 * @brief ASCII prefix check.
 * @param s Input string (may be NULL).
 * @param prefix Prefix string (must be non-NULL).
 * @return 1 if @p s starts with @p prefix, 0 otherwise.
 */
static int starts_with(const char *s, const char *prefix) {
  if (!s) return 0;
  while (*prefix) {
    if (*s++ != *prefix++) return 0;
  }
  return 1;
}

/* -------------------------------------------------------------------------- */
/* Path helpers                                                               */
/* -------------------------------------------------------------------------- */
/**
 * @brief Check whether a filesystem path is absolute (POSIX).
 * @param p Path string (may be NULL).
 * @return 1 if absolute, 0 otherwise.
 */
static int path_is_abs(const char *p) {
  return (p && p[0] == '/');
}

/**
 * @brief Safely copy a string into a fixed buffer (always NUL-terminates).
 * @param dst Destination buffer.
 * @param dstsz Destination size in bytes.
 * @param src Source string (may be NULL).
 */
static void safe_strcpy(char *dst, size_t dstsz, const char *src) {
  if (!dst || dstsz == 0) return;
  if (!src) {
    dst[0] = '\0';
    return;
  }
  snprintf(dst, dstsz, "%s", src);
}

/**
 * @brief Join a directory and a leaf name into a single path.
 *
 * Rules:
 *  - If @p leaf is absolute, it is copied as-is.
 *  - If @p dir is NULL/empty, the result is just @p leaf.
 *  - Otherwise result is "dir/leaf" (single slash).
 *
 * @param out Output buffer.
 * @param outsz Output buffer size in bytes.
 * @param dir Directory prefix (may be NULL/empty).
 * @param leaf File name or relative path (must be non-NULL).
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

  size_t n = strlen(dir);
  if (n > 0 && dir[n - 1] == '/') {
    snprintf(out, outsz, "%s%s", dir, leaf);
  } else {
    snprintf(out, outsz, "%s/%s", dir, leaf);
  }
}

/**
 * @brief Attempt to canonicalize a directory with realpath().
 *
 * If realpath() fails, this function falls back to copying the input.
 *
 * @param out Output buffer (PATH_MAX recommended).
 * @param outsz Output size.
 * @param dir Input directory path (may be NULL).
 * @return 1 if realpath() succeeded, 0 otherwise.
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
 * @brief Check if the precision implies int4-family weights.
 * @param p Precision string (may be NULL).
 * @return 1 if int4 or int4w, 0 otherwise.
 */
static int is_int4_precision(const char *p) {
  if (!p || !*p) return 0;
  return starts_with(p, "int4") ? 1 : 0;
}

/**
 * @brief Resolve model JSON/BIN paths using CLI overrides, model_dir, and precision.
 *
 * Precedence:
 *  1) Explicit --model-json / --model-bin:
 *     - If absolute, used as-is.
 *     - If relative and model_dir exists, resolved under model_dir.
 *     - If relative and no model_dir, used relative to CWD.
 *  2) Otherwise choose defaults under model_dir (or CWD if none):
 *     - If precision is int4/int4w:
 *         prefer model.ie.compat.json + model.q4.bin if both exist.
 *         otherwise fall back to model.ie.json + model.ie.bin.
 *     - Else:
 *         model.ie.json + model.ie.bin.
 *
 * @param model_dir Optional model directory (may be NULL).
 * @param model_json_opt Optional JSON override from CLI (may be NULL).
 * @param model_bin_opt Optional BIN override from CLI (may be NULL).
 * @param precision Effective precision string (e.g., "fp32", "int4", "int4w").
 * @param out_json Output JSON path buffer.
 * @param out_json_sz Output JSON buffer size.
 * @param out_bin Output BIN path buffer.
 * @param out_bin_sz Output BIN buffer size.
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

  /* CLI overrides first. */
  if (model_json_opt && *model_json_opt) {
    if (path_is_abs(model_json_opt) || !base_dir || !*base_dir) {
      safe_strcpy(out_json, out_json_sz, model_json_opt);
    } else {
      join_path(out_json, out_json_sz, base_dir, model_json_opt);
    }
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
    if (path_is_abs(model_bin_opt) || !base_dir || !*base_dir) {
      safe_strcpy(out_bin, out_bin_sz, model_bin_opt);
    } else {
      join_path(out_bin, out_bin_sz, base_dir, model_bin_opt);
    }
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

/* -------------------------------------------------------------------------- */
/* CLI options                                                                */
/* -------------------------------------------------------------------------- */
/** @enum cli_precision_t
 *  @brief Floating-point precision flags for convenience.
 */
typedef enum {
  CLI_PREC_FP32 = 0,
  CLI_PREC_BF16 = 1,
  CLI_PREC_FP16 = 2
} cli_precision_t;

/** @enum cli_pretranspose_t
 *  @brief Pretranspose policy hint.
 */
typedef enum {
  CLI_PRETX_NONE = 0,
  CLI_PRETX_WOH = 1,
  CLI_PRETX_WXH = 2,
  CLI_PRETX_ALL = 3
} cli_pretranspose_t;

/**
 * @struct cli_extras_t
 * @brief Parsed CLI options container (benchmark/harness friendly).
 */
typedef struct cli_extras {
  const char *prompt;
  size_t max_new;
  int threads;
  cli_precision_t prec;
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

  size_t pf_w_bytes;
  size_t pf_x_bytes;
  int force_ntw;

  int dedup_enable;        /**< -1=unset, 0=disable, 1=enable. */
  const char *dedup_policy;/**< NULL if unset, else string. */
  long dedup_cache_mb;     /**< -1 unset, else >=0. */
  long dedup_hot_bytes;    /**< -1 unset, else >=0. */
  const char *dedup_hot_list; /**< NULL if unset. */
} cli_extras_t;

/* -------------------------------------------------------------------------- */
/* CLI parsing                                                                */
/* -------------------------------------------------------------------------- */
/**
 * @brief Print CLI usage information to stderr.
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
          "                        [--pf-w BYTES] [--pf-x BYTES] [--force-ntw 0|1]\n"
          "                        [--dedup 0|1]\n"
          "                        [--dedup-policy off|on_demand|cache|replicate_hot]\n"
          "                        [--dedup-cache-mb N]\n"
          "                        [--dedup-hot-bytes N]\n"
          "                        [--dedup-hot-list STR]\n"
          "\n"
          "Env overrides: IE_PRECISION/PRECISION; IE_SPARSITY/SPARSITY;\n"
          "               IE_GEMV_PFDIST_W; IE_GEMV_PFDIST_X; IE_GEMV_FORCE_NTW;\n"
          "               IE_DEDUP; IE_DEDUP_POLICY; IE_DEDUP_CACHE_MB;\n"
          "               IE_DEDUP_HOT_BYTES; IE_DEDUP_HOT_LIST\n");
}

/**
 * @brief Initialize a ::cli_extras_t with default values.
 * @param e Output structure; must be non-NULL.
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt = NULL;
  e->max_new = 8;
  e->threads = 0;
  e->prec = CLI_PREC_FP32;
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

  e->pf_w_bytes = 0;
  e->pf_x_bytes = 0;
  e->force_ntw = -1;

  e->dedup_enable = -1;
  e->dedup_policy = NULL;
  e->dedup_cache_mb = -1;
  e->dedup_hot_bytes = -1;
  e->dedup_hot_list = NULL;
}

/**
 * @brief Convert a string to long with strict validation; exits with code 2 on failure.
 * @param s NUL-terminated string to parse as base-10 integer.
 * @return Parsed long value.
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
 * @brief Parse command-line flags into a ::cli_extras_t container.
 *
 * Recognized flags include prefetch/streaming controls:
 *  - --pf-w BYTES, --pf-x BYTES, --force-ntw 0|1
 *
 * Recognized dedup controls:
 *  - --dedup 0|1
 *  - --dedup-policy off|on_demand|cache|replicate_hot
 *  - --dedup-cache-mb N
 *  - --dedup-hot-bytes N
 *  - --dedup-hot-list STR
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param out Output structure; must be non-NULL.
 * @return 0 on success; -1 if help was printed or on parsing error.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
      usage();
      return -1;
    } else if (!strcmp(a, "--prompt")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->prompt = argv[i];
    } else if (!strcmp(a, "--max-new")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) {
        fprintf(stderr, "error: --max-new >= 0\n");
        return -1;
      }
      out->max_new = (size_t)v;
    } else if (!strcmp(a, "--threads")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      out->threads = (int)v;
    } else if (!strcmp(a, "--precision")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      const char *m = argv[i];
      if (ascii_ieq(m, IE_PREC_FP32)) {
        out->prec = CLI_PREC_FP32;
        out->precision_label = IE_PREC_FP32;
        out->precision_from_flag = 1;
      } else if (ascii_ieq(m, IE_PREC_BF16)) {
        out->prec = CLI_PREC_BF16;
        out->precision_label = IE_PREC_BF16;
        out->precision_from_flag = 1;
      } else if (ascii_ieq(m, IE_PREC_FP16)) {
        out->prec = CLI_PREC_FP16;
        out->precision_label = IE_PREC_FP16;
        out->precision_from_flag = 1;
      } else if (ascii_ieq(m, IE_PREC_INT8W)) {
        out->precision_label = IE_PREC_INT8W;
        out->precision_from_flag = 1;
      } else if (ascii_ieq(m, IE_PREC_INT4W)) {
        out->precision_label = IE_PREC_INT4W;
        out->precision_from_flag = 1;
      } else if (ascii_ieq(m, IE_PREC_INT4)) {
        out->precision_label = IE_PREC_INT4;
        out->precision_from_flag = 1;
      } else {
        fprintf(stderr, "error: unknown precision '%s'\n", m);
        return -1;
      }
    } else if (!strcmp(a, "--sparsity")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      const char *m = argv[i];
      if (ascii_ieq(m, "none") || ascii_ieq(m, "dense")) {
        out->sparsity = "none";
      } else if (ascii_ieq(m, "block") || ascii_ieq(m, "blocksparse")) {
        out->sparsity = "block";
      } else if (ascii_ieq(m, "auto")) {
        out->sparsity = "auto";
      } else {
        fprintf(stderr, "error: unknown sparsity '%s' (expected none|block|auto)\n", m);
        return -1;
      }
      out->sparsity_from_flag = 1;
    } else if (!strcmp(a, "--affinity")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->affinity = argv[i];
    } else if (!strcmp(a, "--pretranspose")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      const char *p = argv[i];
      if (!strcmp(p, "none"))
        out->pretx = CLI_PRETX_NONE;
      else if (!strcmp(p, "woh"))
        out->pretx = CLI_PRETX_WOH;
      else if (!strcmp(p, "wxh"))
        out->pretx = CLI_PRETX_WXH;
      else if (!strcmp(p, "all"))
        out->pretx = CLI_PRETX_ALL;
      else {
        fprintf(stderr, "error: unknown pretranspose '%s'\n", p);
        return -1;
      }
    } else if (!strcmp(a, "--device")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->device = argv[i];
    } else if (!strcmp(a, "--model-dir")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->model_dir = argv[i];
    } else if (!strcmp(a, "--model-json")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->model_json = argv[i];
    } else if (!strcmp(a, "--model-bin")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->model_bin = argv[i];
    } else if (!strcmp(a, "--prompts-file")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->prompts_file = argv[i];
    } else if (!strcmp(a, "--batch")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->batch = (int)safe_atoi(argv[i]);
    } else if (!strcmp(a, "--prefetch")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->prefetch = argv[i];
    } else if (!strcmp(a, "--warmup") || !strcmp(a, "--warmup-tokens")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) v = 0;
      out->warmup_tokens = (int)v;
    } else if (!strcmp(a, "--rounds")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 1) v = 1;
      out->rounds = (int)v;
    } else if (!strcmp(a, "--aggregate")) {
      out->aggregate = 1;
    } else if (!strcmp(a, "--pf-w")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) {
        fprintf(stderr, "error: --pf-w >= 0\n");
        return -1;
      }
      out->pf_w_bytes = (size_t)v;
    } else if (!strcmp(a, "--pf-x")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) {
        fprintf(stderr, "error: --pf-x >= 0\n");
        return -1;
      }
      out->pf_x_bytes = (size_t)v;
    } else if (!strcmp(a, "--force-ntw")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v != 0 && v != 1) {
        fprintf(stderr, "error: --force-ntw 0|1\n");
        return -1;
      }
      out->force_ntw = (int)v;
    } else if (!strcmp(a, "--dedup")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v != 0 && v != 1) {
        fprintf(stderr, "error: --dedup 0|1\n");
        return -1;
      }
      out->dedup_enable = (int)v;
    } else if (!strcmp(a, "--dedup-policy")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->dedup_policy = argv[i];
    } else if (!strcmp(a, "--dedup-cache-mb")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) v = 0;
      out->dedup_cache_mb = v;
    } else if (!strcmp(a, "--dedup-hot-bytes")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      long v = safe_atoi(argv[i]);
      if (v < 0) v = 0;
      out->dedup_hot_bytes = v;
    } else if (!strcmp(a, "--dedup-hot-list")) {
      if (++i >= argc) {
        usage();
        return -1;
      }
      out->dedup_hot_list = argv[i];
    } else if (a[0] == '-') {
      fprintf(stderr, "error: unknown flag '%s'\n", a);
      usage();
      return -1;
    } else {
      out->prompt = a;
    }
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Prompts helper                                                             */
/* -------------------------------------------------------------------------- */
/**
 * @brief Read the first non-empty line from a text file.
 * @param path Path to the text file.
 * @param buf Output buffer to receive the line.
 * @param bufsz Size of @p buf in bytes.
 * @return 1 if a line was read, 0 if the file had no non-empty lines, -1 on I/O error.
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
  fclose(f);
  return ok ? 1 : 0;
}

/* -------------------------------------------------------------------------- */
/* JSON emitter                                                               */
/* -------------------------------------------------------------------------- */
/**
 * @brief Print a single JSON summary line for the last run.
 * @param n_tok Number of tokens generated.
 * @param tokens Pointer to tokens (may be NULL to print an empty array).
 * @param wall_s_in Wall-clock seconds for the measured window.
 * @param kv_hits KV hits to report.
 * @param kv_misses KV misses to report.
 * @param rss_peak_mb Peak RSS in MiB to report.
 */
static void print_json_result(uint32_t n_tok,
                              const uint32_t *tokens,
                              double wall_s_in,
                              uint64_t kv_hits,
                              uint64_t kv_misses,
                              uint32_t rss_peak_mb) {
  const double WALL_MIN = 3e-4;
  double wall_s = wall_s_in;
  if (wall_s > 0.0 && wall_s < WALL_MIN) wall_s = WALL_MIN;

  const double tps_true = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;

  double p50 = 0.0, p95 = 0.0;
  if (n_tok > 0 && wall_s > 0.0) {
    double per_tok_ms = (wall_s * 1000.0) / (double)n_tok;
    if (per_tok_ms < 0.001) per_tok_ms = 0.001;
    p50 = per_tok_ms;
    p95 = per_tok_ms * 2.0;
  }

  fprintf(stdout, "{\"tokens_generated\": %u,", (unsigned)n_tok);

  fputs("\"tokens\":[", stdout);
  if (tokens && n_tok > 0) {
    for (uint32_t i = 0; i < n_tok; ++i) {
      fprintf(stdout, "%u%s", tokens[i], (i + 1 < n_tok) ? "," : "");
    }
  }
  fputs("],", stdout);

  fprintf(stdout,
          "\"wall_time_s\": %.6f,"
          "\"tps_true\": %.6f,"
          "\"latency_p50_ms\": %.3f,"
          "\"latency_p95_ms\": %.3f,"
          "\"rss_peak_mb\": %u,"
          "\"kv_hits\": %llu,"
          "\"kv_misses\": %llu}\n",
          wall_s,
          tps_true,
          p50,
          p95,
          (unsigned)rss_peak_mb,
          (unsigned long long)kv_hits,
          (unsigned long long)kv_misses);
}

/* -------------------------------------------------------------------------- */
/* main                                                                       */
/* -------------------------------------------------------------------------- */
/**
 * @brief Program entry point.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; non-zero on error.
 */
int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* Export GEMV tuning knobs. */
  if (opt.pf_w_bytes > 0) {
    char buf[32];
    snprintf(buf, sizeof buf, "%zu", opt.pf_w_bytes);
    setenv("IE_GEMV_PFDIST_W", buf, 1);
  }
  if (opt.pf_x_bytes > 0) {
    char buf[32];
    snprintf(buf, sizeof buf, "%zu", opt.pf_x_bytes);
    setenv("IE_GEMV_PFDIST_X", buf, 1);
  }
  if (opt.force_ntw == 0 || opt.force_ntw == 1) {
    setenv("IE_GEMV_FORCE_NTW", opt.force_ntw ? "1" : "0", 1);
  }

  /* Export dedup knobs as env vars (loader/runtime consumes these). */
  if (opt.dedup_enable == 0 || opt.dedup_enable == 1) {
    setenv("IE_DEDUP", opt.dedup_enable ? "1" : "0", 1);
  }
  if (opt.dedup_policy && *opt.dedup_policy) {
    setenv("IE_DEDUP_POLICY", opt.dedup_policy, 1);
  }
  if (opt.dedup_cache_mb >= 0) {
    char buf[32];
    snprintf(buf, sizeof buf, "%ld", opt.dedup_cache_mb);
    setenv("IE_DEDUP_CACHE_MB", buf, 1);
  }
  if (opt.dedup_hot_bytes >= 0) {
    char buf[32];
    snprintf(buf, sizeof buf, "%ld", opt.dedup_hot_bytes);
    setenv("IE_DEDUP_HOT_BYTES", buf, 1);
  }
  if (opt.dedup_hot_list && *opt.dedup_hot_list) {
    setenv("IE_DEDUP_HOT_LIST", opt.dedup_hot_list, 1);
  }

  /* Determine effective precision from CLI or env. */
  if (!opt.precision_from_flag) {
    const char *envp = env_str("IE_PRECISION", env_str("PRECISION", IE_PREC_FP32));
    if (ascii_ieq(envp, IE_PREC_INT4W)) opt.precision_label = IE_PREC_INT4W;
    else if (ascii_ieq(envp, IE_PREC_INT4)) opt.precision_label = IE_PREC_INT4;
    else if (ascii_ieq(envp, IE_PREC_INT8W)) opt.precision_label = IE_PREC_INT8W;
    else if (ascii_ieq(envp, IE_PREC_BF16)) opt.precision_label = IE_PREC_BF16;
    else if (ascii_ieq(envp, IE_PREC_FP16)) opt.precision_label = IE_PREC_FP16;
    else opt.precision_label = IE_PREC_FP32;
  }

  /* Determine effective sparsity from CLI or env. */
  if (!opt.sparsity_from_flag) {
    const char *envs = env_str("IE_SPARSITY", env_str("SPARSITY", "none"));
    if (ascii_ieq(envs, "block") || ascii_ieq(envs, "blocksparse")) opt.sparsity = "block";
    else if (ascii_ieq(envs, "auto")) opt.sparsity = "auto";
    else opt.sparsity = "none";
  }

  /* Resolve model directory. */
  const char *model_dir_eff = opt.model_dir;
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
      fprintf(stderr, "error: --model-dir '%s' is not accessible: %s\n",
              opt.model_dir, strerror(errno));
      return 3;
    }
  }

  /* Resolve JSON/BIN paths (precision-aware defaults). */
  char json_path[PATH_MAX];
  char bin_path[PATH_MAX];
  resolve_model_paths(model_dir_eff,
                      opt.model_json,
                      opt.model_bin,
                      opt.precision_label,
                      json_path, sizeof(json_path),
                      bin_path, sizeof(bin_path));

  /* Prompt selection. */
  char prompt_buf[8192];
  if (!opt.prompt && opt.prompts_file && !opt.aggregate) {
    int r = read_first_nonempty_line(opt.prompts_file, prompt_buf, sizeof(prompt_buf));
    if (r == 1) opt.prompt = prompt_buf;
    else if (r == 0) {
      fprintf(stderr, "warn: prompts file is empty; using default prompt\n");
      opt.prompt = "bench";
    } else {
      opt.prompt = "bench";
    }
  }

  if (!opt.prompt && !opt.prompts_file) {
    print_json_result(0, NULL, 0.0, 0, 0, 0);
    return 0;
  }

  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  /* Open weights metadata to validate model presence and optionally touch. */
  ie_weights_t w;
  memset(&w, 0, sizeof(w));
  {
    int wrc = ie_weights_open(json_path, bin_path, &w);
    if (wrc != IE_IO_OK) {
      if (require_model) {
        fprintf(stderr,
                "error: failed to open model (%s, %s), status=%d, errno=%d (%s)\n",
                json_path, bin_path, wrc, errno, strerror(errno));
        return 3;
      }
      fprintf(stderr, "# -> warn: model metadata not found; continuing in stub mode...\n");
    } else {
      if (ie_weights_touch(&w) != 0) {
        fprintf(stderr, "error: model present but unreadable (prefault/touch failed)\n");
        ie_weights_close(&w);
        return 3;
      }
    }
  }

  /* Create engine. */
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
  UNUSED(opt.device);

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    ie_weights_close(&w);
    return 5;
  }

  /* Warmup. */
  if (opt.warmup_tokens > 0) {
    const char *wprompt = "warmup";
    uint32_t wtoks[128];
    uint32_t wcount = 0;
    size_t wmax = (opt.warmup_tokens <= (int)(sizeof(wtoks) / sizeof(wtoks[0])))
                      ? (size_t)opt.warmup_tokens
                      : (sizeof(wtoks) / sizeof(wtoks[0]));
    (void)ie_engine_generate(engine, wprompt, wmax, wtoks, &wcount);
  }

  /* Output token buffer. */
  size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
  size_t cap_alloc = (cap > 0 ? cap : 1);
  uint32_t *tokens = (uint32_t *)malloc(sizeof(uint32_t) * cap_alloc);
  if (!tokens) {
    fprintf(stderr, "error: OOM allocating token buffer\n");
    ie_engine_destroy(engine);
    ie_weights_close(&w);
    return 6;
  }

  /* Optional "work-touch" loop for bandwidth measurement. */
  const size_t bytes_per_token = (size_t)env_long("IE_BYTES_PER_TOKEN", 0);
  const size_t stride_bytes = (size_t)env_long("IE_STRIDE_BYTES", 256);
  const int verify_touch = (int)env_long("IE_VERIFY_TOUCH", 0);

  int bin_fd = -1;
  uint8_t *map = NULL;
  size_t map_len = 0;
  if (w.loaded && bytes_per_token > 0 && w.bin_size_bytes > 0) {
    const char *binp = (w.weights_path[0] ? w.weights_path : bin_path);
    bin_fd = open(binp, O_RDONLY);
    if (bin_fd >= 0) {
      map_len = (size_t)w.bin_size_bytes;
      void *p = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, bin_fd, 0);
      if (p == MAP_FAILED) {
        map = NULL;
        map_len = 0;
      } else {
        map = (uint8_t *)p;
      }
    }
  }

  /* Run generation and collect metrics. */
  (void)ie_kv_begin_round();
  uint64_t total_tokens_this_round = 0;

  double t0 = now_sec();

  uint32_t tokens_generated_total = 0;

  for (int r = 0; r < (opt.rounds > 0 ? opt.rounds : 1); ++r) {
    if (opt.aggregate && opt.prompts_file) {
      FILE *pf = fopen(opt.prompts_file, "r");
      if (pf) {
        char line[8192];
        while (fgets(line, sizeof(line), pf)) {
          size_t n = strlen(line);
          while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
          if (!n) continue;

          uint32_t n_here = 0;
          st = ie_engine_generate(engine, line, cap, tokens, &n_here);
          if (st != IE_OK) {
            fprintf(stderr, "error: ie_engine_generate (status=%d)\n", (int)st);
            break;
          }
          tokens_generated_total += n_here;
          total_tokens_this_round += (uint64_t)n_here;

          if (map && map_len && bytes_per_token) {
            size_t need = (size_t)n_here * bytes_per_token;
            size_t pos = 0;
            volatile uint64_t acc = 0;
            while (pos < need) {
              size_t off = (pos % map_len);
              acc += map[off];
              pos += (stride_bytes ? stride_bytes : 1);
            }
            if (verify_touch) {
              (void)acc;
            }
          }
        }
        fclose(pf);
      } else {
        fprintf(stderr, "warn: cannot open prompts-file '%s'\n", opt.prompts_file);
      }
    } else {
      const char *p = (opt.prompt ? opt.prompt : "bench");
      uint32_t n_here = 0;
      st = ie_engine_generate(engine, p, cap, tokens, &n_here);
      if (st != IE_OK) {
        fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)st);
      }
      tokens_generated_total += n_here;
      total_tokens_this_round += (uint64_t)n_here;

      if (map && map_len && bytes_per_token) {
        size_t need = (size_t)n_here * bytes_per_token;
        size_t pos = 0;
        volatile uint64_t acc = 0;
        while (pos < need) {
          size_t off = (pos % map_len);
          acc += map[off];
          pos += (stride_bytes ? stride_bytes : 1);
        }
        if (verify_touch) {
          (void)acc;
        }
      }
    }
  }

  double t1 = now_sec();

  uint64_t kv_hits_round = 0, kv_miss_round = 0;
  ie_kv_finish_round(total_tokens_this_round, &kv_hits_round, &kv_miss_round);

  ie_metrics_t m;
  memset(&m, 0, sizeof(m));
  (void)ie_engine_metrics(engine, &m);
  if (kv_hits_round || kv_miss_round) {
    m.kv_hits = kv_hits_round;
    m.kv_misses = kv_miss_round;
  }

  uint32_t rss_mib = ie_metrics_sample_rss_peak();
  m.rss_peak_mb = rss_mib;

  const uint32_t *tokens_to_print = (opt.aggregate || opt.rounds > 1) ? NULL : tokens;

  print_json_result(tokens_generated_total,
                    tokens_to_print,
                    (t1 - t0),
                    m.kv_hits,
                    m.kv_misses,
                    rss_mib);

  /* Cleanup. */
  free(tokens);
  if (map && map != MAP_FAILED) munmap(map, map_len);
  if (bin_fd >= 0) close(bin_fd);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  return 0;
}
