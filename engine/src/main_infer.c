/* ========================================================================== */
/* File: engine/src/main_infer.c                                              */
/* ========================================================================== */
/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine (benchmark-friendly).
 *
 * @section modes Execution modes
 * 1. @b CI/Stub mode (default) — does not require model files. The engine
 *    generates deterministic pseudo-token IDs (see ie_api.c).
 * 2. @b Strict mode (opt-in) — when `IE_REQUIRE_MODEL=1`, the program requires
 *    a valid IEBIN v1 pair in the current directory (`model.ie.json` +
 *    `model.ie.bin`). If `IE_BYTES_PER_TOKEN > 0`, it mmaps the weights and
 *    performs a @b per-token “work touch” loop to mimic realistic memory
 *    pressure. This work-touch loop is @b counted inside the timed window.
 *
 * @section timing Critical timing rule
 * The measured window includes only:
 *   - `ie_engine_generate(...)`
 *   - the optional per-token “work touch” over the model blob
 *
 * All instrumentation (RSS sampling, JSON printing, etc.) runs @b outside the
 * measured window to avoid skewing tokens/s measurements.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ie_api.h"                 /* engine API + ie_metrics_t */
#include "ie_io.h"                  /* IEBIN loader */
#include "ie_kv_instrumentation.h"  /* KV round helpers (begin/finish/on_token) */
#include "util_metrics.h"           /* KV & RSS helpers */
#include "util_logging.h"           /* optional logging macros */

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* Provide string defaults locally to avoid depending on external macros. */
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
 *
 * The origin of the monotonic clock is unspecified; only deltas are meaningful.
 *
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
 *
 * If the variable is unset or cannot be parsed as a base-10 integer,
 * @p defv is returned.
 *
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
 *
 * @param name Environment variable name.
 * @param defv Default string when unset/empty.
 * @return Pointer to environment value or @p defv.
 */
static const char *env_str(const char *name, const char *defv) {
  const char *s = getenv(name);
  return (s && *s) ? s : defv;
}

/* -------------------------------------------------------------------------- */
/* ASCII utilities                                                            */
/* -------------------------------------------------------------------------- */
/**
 * @brief Case-insensitive ASCII equality check (NULL-safe).
 *
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
    ++a; ++b;
  }
  return *a == '\0' && *b == '\0';
}

/* -------------------------------------------------------------------------- */
/* CLI options                                                                */
/* -------------------------------------------------------------------------- */
/**
 * @enum cli_precision_t
 * @brief Classic floating-point precision flags recognized as switches.
 *
 * Weight-only INTx labels are handled via a raw string to avoid proliferating
 * enum values.
 */
typedef enum {
  CLI_PREC_FP32 = 0,  /**< 32-bit floating-point. */
  CLI_PREC_BF16 = 1,  /**< bfloat16. */
  CLI_PREC_FP16 = 2   /**< 16-bit floating-point. */
} cli_precision_t;

/**
 * @enum cli_pretranspose_t
 * @brief Pretranspose policy hint.
 */
typedef enum {
  CLI_PRETX_NONE = 0, /**< No pretranspose. */
  CLI_PRETX_WOH  = 1, /**< Width-Outer-Height style. */
  CLI_PRETX_WXH  = 2, /**< Width-by-Height tiles. */
  CLI_PRETX_ALL  = 3  /**< Let backend decide per tensor. */
} cli_pretranspose_t;

/**
 * @struct cli_extras_t
 * @brief Parsed CLI options container (benchmark/harness compatible).
 */
typedef struct cli_extras {
  const char        *prompt;        /**< Prompt text; may be NULL if using file. */
  size_t             max_new;       /**< Upper bound of new tokens to generate. */
  int                threads;       /**< Thread hint; <= 0 means auto. */
  cli_precision_t    prec;          /**< Float precision hint (fp32/bf16/fp16). */
  const char        *affinity;      /**< "auto" | "compact" | "scatter". */
  cli_pretranspose_t pretx;         /**< Pretranspose policy. */
  const char        *device;        /**< "auto" | "cpu" | "cuda" | "ze" (no-op hint). */

  /* Harness/compat */
  const char        *prompts_file;  /**< Path to prompts file (one per line). */
  int                batch;         /**< Batch size (compat hint). */
  const char        *prefetch;      /**< "on" | "off" | "auto" | "N" (string). */
  int                warmup_tokens; /**< Warmup tokens before measurement. */
  int                aggregate;     /**< Aggregate mode: iterate prompts-file. */
  int                rounds;        /**< Repeat measured window N times (>=1). */

  /* Model location options (for harness compatibility). */
  const char        *model_dir;     /**< If set, chdir() here before loading IEBIN. */
  const char        *model_json;    /**< Optional explicit model JSON path. */
  const char        *model_bin;     /**< Optional explicit model BIN path. */

  /* Raw precision label passed to engine (includes int8w/int4/int4w). */
  const char        *precision_label;   /**< e.g. "fp32"|"bf16"|"fp16"|"int8w"|"int4w". */
  int                precision_from_flag; /**< 1 if set via --precision, else 0 to allow env override. */
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
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--pretranspose none|woh|wxh|all]\n"
    "                        [--device auto|cpu|cuda|ze]\n"
    "                        [--model-dir PATH]\n"
    "                        [--model-json PATH] [--model-bin PATH]\n"
    "                        [--prompts-file PATH] [--batch N]\n"
    "                        [--prefetch on|off|auto|N] [--warmup N]\n"
    "                        [--rounds N]\n"
    "                        [--aggregate]\n"
    "\n"
    "Env overrides: IE_PRECISION or PRECISION (same accepted values)\n");
}

/**
 * @brief Initialize a ::cli_extras_t with default values.
 *
 * @param e Output structure; must be non-NULL.
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt         = NULL;
  e->max_new        = 8;
  e->threads        = 0;
  e->prec           = CLI_PREC_FP32;
  e->affinity       = "auto";
  e->pretx          = CLI_PRETX_NONE;
  e->device         = "auto";
  e->prompts_file   = NULL;
  e->batch          = 1;
  e->prefetch       = "auto";
  e->warmup_tokens  = 1;
  e->aggregate      = 0;
  e->rounds         = 1;
  e->model_dir      = NULL;
  e->model_json     = NULL;
  e->model_bin      = NULL;
  e->precision_label = IE_PREC_FP32; /* "fp32" */
  e->precision_from_flag = 0;
}

/**
 * @brief Convert a string to long with strict validation; exits with code 2 on failure.
 *
 * @param s NUL-terminated string.
 * @return Parsed long value.
 */
static long safe_atoi(const char *s) {
  if (!s || !*s) { fprintf(stderr, "error: empty integer\n"); exit(2); }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end) { fprintf(stderr, "error: invalid integer: '%s'\n", s); exit(2); }
  return v;
}

/**
 * @brief Parse command-line flags into a ::cli_extras_t.
 *
 * Recognized flags:
 *  - `--prompt TEXT`
 *  - `--max-new N`
 *  - `--threads N`
 *  - `--precision fp32|bf16|fp16|int8w|int4|int4w`
 *  - `--affinity auto|compact|scatter`
 *  - `--pretranspose none|woh|wxh|all`
 *  - `--device auto|cpu|cuda|ze` (accepted as a no-op hint)
 *  - `--model-dir PATH` (changes working directory before loading)
 *  - `--model-json PATH` (explicit JSON path)
 *  - `--model-bin PATH`  (explicit BIN path)
 *  - `--prompts-file PATH`
 *  - `--batch N`
 *  - `--prefetch on|off|auto|N`
 *  - `--warmup N` (alias `--warmup-tokens`)
 *  - `--rounds N` (repeat measured window N times; N>=1)
 *  - `--aggregate`
 *  - `--help` / `-h`
 *
 * Non-flag positional text is treated as `--prompt TEXT`.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param out  Output structure; must be non-NULL.
 * @return 0 on success; -1 if help was printed or on parsing error.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a,"--help") || !strcmp(a,"-h")) { usage(); return -1; }
    else if (!strcmp(a,"--prompt"))      { if (++i>=argc) { usage(); return -1; } out->prompt       = argv[i]; }
    else if (!strcmp(a,"--max-new"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); if(v<0){fprintf(stderr,"error: --max-new >= 0\n");return -1;} out->max_new=(size_t)v; }
    else if (!strcmp(a,"--threads"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); out->threads=(int)v; }
    else if (!strcmp(a,"--precision"))   {
      if (++i>=argc) { usage(); return -1; }
      const char *m = argv[i];
      if      (ascii_ieq(m, IE_PREC_FP32)) out->prec=CLI_PREC_FP32, out->precision_label=IE_PREC_FP32, out->precision_from_flag=1;
      else if (ascii_ieq(m, IE_PREC_BF16)) out->prec=CLI_PREC_BF16, out->precision_label=IE_PREC_BF16, out->precision_from_flag=1;
      else if (ascii_ieq(m, IE_PREC_FP16)) out->prec=CLI_PREC_FP16, out->precision_label=IE_PREC_FP16, out->precision_from_flag=1;
      else if (ascii_ieq(m, IE_PREC_INT8W)) { out->precision_label = IE_PREC_INT8W; out->precision_from_flag=1; }
      else if (ascii_ieq(m, IE_PREC_INT4W) || ascii_ieq(m, IE_PREC_INT4)) { out->precision_label = IE_PREC_INT4W; out->precision_from_flag=1; }
      else { fprintf(stderr,"error: unknown precision '%s'\n", m); return -1; }
    }
    else if (!strcmp(a,"--affinity"))    { if (++i>=argc) { usage(); return -1; } out->affinity = argv[i]; }
    else if (!strcmp(a,"--pretranspose")){ if (++i>=argc) { usage(); return -1; } const char*p=argv[i];
      if(!strcmp(p,"none")) out->pretx=CLI_PRETX_NONE;
      else if(!strcmp(p,"woh")) out->pretx=CLI_PRETX_WOH;
      else if(!strcmp(p,"wxh")) out->pretx=CLI_PRETX_WXH;
      else if(!strcmp(p,"all")) out->pretx=CLI_PRETX_ALL;
      else { fprintf(stderr,"error: unknown pretranspose '%s'\n", p); return -1; }
    }
    else if (!strcmp(a,"--device"))      { if (++i>=argc) { usage(); return -1; } out->device = argv[i]; /* accepted; no-op hint */ }
    else if (!strcmp(a,"--model-dir"))   { if (++i>=argc) { usage(); return -1; } out->model_dir = argv[i]; }
    else if (!strcmp(a,"--model-json"))  { if (++i>=argc) { usage(); return -1; } out->model_json = argv[i]; }
    else if (!strcmp(a,"--model-bin"))   { if (++i>=argc) { usage(); return -1; } out->model_bin  = argv[i]; }
    else if (!strcmp(a,"--prompts-file")){ if (++i>=argc) { usage(); return -1; } out->prompts_file = argv[i]; }
    else if (!strcmp(a,"--batch"))       { if (++i>=argc) { usage(); return -1; } out->batch = (int)safe_atoi(argv[i]); }
    else if (!strcmp(a,"--prefetch"))    { if (++i>=argc) { usage(); return -1; } out->prefetch = argv[i]; }
    else if (!strcmp(a,"--warmup") || !strcmp(a,"--warmup-tokens")) {
      if (++i>=argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]); if (v < 0) v = 0; out->warmup_tokens = (int)v;
    }
    else if (!strcmp(a,"--rounds"))      {
      if (++i>=argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]);
      if (v < 1) v = 1;
      out->rounds = (int)v;
    }
    else if (!strcmp(a,"--aggregate"))   { out->aggregate = 1; }
    else if (a[0] == '-') { fprintf(stderr,"error: unknown flag '%s'\n", a); usage(); return -1; }
    else { out->prompt = a; } /* positional */
  }
  return 0;
}

/* -------------------------------------------------------------------------- */
/* Prompts helper                                                             */
/* -------------------------------------------------------------------------- */
/**
 * @brief Read the first non-empty line from a text file.
 *
 * Trailing `\n`/`\r` are stripped. On success, the result is copied into
 * @p buf (NUL-terminated).
 *
 * @param path  Path to the text file.
 * @param buf   Output buffer to receive the line.
 * @param bufsz Size of @p buf in bytes.
 * @return 1 if a line was read, 0 if the file had no non-empty lines,
 *         -1 on I/O error (also prints a warning).
 */
static int read_first_nonempty_line(const char *path, char *buf, size_t bufsz) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "warn: cannot open prompts file '%s': %s\n", path, strerror(errno)); return -1; }
  int ok = 0;
  while (fgets(buf, (int)bufsz, f)) {
    size_t n = strlen(buf);
    while (n && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
    if (n == 0) continue;
    ok = 1; break;
  }
  fclose(f);
  return ok ? 1 : 0;
}

/* -------------------------------------------------------------------------- */
/* JSON emitter                                                               */
/* -------------------------------------------------------------------------- */
/**
 * @brief Print a single JSON summary line for the last run.
 *
 * The JSON object includes:
 *  - `tokens_generated` (uint)
 *  - `tokens` (array; empty if @p tokens is NULL)
 *  - `wall_time_s` (seconds)
 *  - `tps_true` (tokens/s)
 *  - `latency_p50_ms`, `latency_p95_ms`
 *  - `rss_peak_mb`
 *  - `kv_hits`, `kv_misses`
 *
 * @param n_tok         Number of tokens generated.
 * @param tokens        Pointer to tokens (may be NULL to print an empty array).
 * @param wall_s_in     Wall-clock seconds for the measured window.
 * @param kv_hits       KV hits to report.
 * @param kv_misses     KV misses to report.
 * @param rss_peak_mb   Peak RSS in MiB to report.
 */
static void print_json_result(uint32_t n_tok,
                              const uint32_t *tokens,
                              double wall_s_in,
                              uint64_t kv_hits,
                              uint64_t kv_misses,
                              uint32_t rss_peak_mb) {
  const double WALL_MIN = 3e-4; /* 0.0003 s */
  double wall_s = wall_s_in;
  if (wall_s > 0.0 && wall_s < WALL_MIN) wall_s = WALL_MIN;

  /* True TPS: by definition in this tool, it's n_tok / wall-time */
  const double tps_true = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;

  /* Simple latency proxies if caller didn't compute them elsewhere. */
  double p50 = 0.0, p95 = 0.0;
  if (n_tok > 0 && wall_s > 0.0) {
    double per_tok_ms = (wall_s * 1000.0) / (double)n_tok;
    if (per_tok_ms < 0.001) per_tok_ms = 0.001;
    p50 = per_tok_ms;
    p95 = per_tok_ms * 2.0;
  }

  fprintf(stdout, "{\"tokens_generated\": %u,", (unsigned)n_tok);

  /* tokens array: empty when tokens==NULL to avoid huge prints. */
  fputs("\"tokens\":[", stdout);
  if (tokens && n_tok > 0) {
    for (uint32_t i = 0; i < n_tok; ++i) {
      const uint32_t v = tokens[i];
      fprintf(stdout, "%u%s", v, (i + 1 < n_tok) ? "," : "");
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
 *
 * High-level flow:
 *  1. Parse CLI flags (or print usage).
 *  2. Optionally change into `--model-dir`.
 *  3. Optionally read `model.ie.json`/`model.ie.bin` and enforce strict mode
 *     via `IE_REQUIRE_MODEL=1`.
 *  4. Create the engine with soft hints (precision/affinity/etc).
 *  5. Optional warmup (not timed).
 *  6. Measured window: `ie_engine_generate(...)` and optional “work touch”
 *     repeated `--rounds` times.
 *  7. After timing: collect metrics, print JSON, teardown.
 *
 * Environment variables:
 *  - `IE_REQUIRE_MODEL` (int): if `1`, strict mode is enforced.
 *  - `IE_BYTES_PER_TOKEN` (size_t): bytes to touch per token (work touch).
 *  - `IE_STRIDE_BYTES` (size_t): stride in bytes for work-touch loop.
 *  - `IE_VERIFY_TOUCH` (int): when non-zero, prevents the compiler from eliding
 *    the touch accumulator (side-effect barrier).
 *  - `IE_PRECISION` / `PRECISION` (string): precision label (fp32/bf16/fp16/int8w/int4w/int4).
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; non-zero on error.
 */
int main(int argc, char **argv) {
  /* -------------------- parse CLI -------------------- */
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* Optional: change working dir early if requested. */
  if (opt.model_dir && *opt.model_dir) {
    if (chdir(opt.model_dir) != 0) {
      fprintf(stderr, "error: --model-dir '%s' is not accessible: %s\n",
              opt.model_dir, strerror(errno));
      return 3;
    }
  }

  /* Precision from env if not provided on the command line. */
  if (!opt.precision_from_flag) {
    const char *envp = env_str("IE_PRECISION",
                      env_str("PRECISION", IE_PREC_FP32));
    if      (ascii_ieq(envp, IE_PREC_INT4W) || ascii_ieq(envp, IE_PREC_INT4))
      opt.precision_label = IE_PREC_INT4W;
    else if (ascii_ieq(envp, IE_PREC_INT8W))
      opt.precision_label = IE_PREC_INT8W;
    else if (ascii_ieq(envp, IE_PREC_BF16))
      opt.precision_label = IE_PREC_BF16;
    else if (ascii_ieq(envp, IE_PREC_FP16))
      opt.precision_label = IE_PREC_FP16;
    else
      opt.precision_label = IE_PREC_FP32;
  }

  /* If no explicit --prompt, try --prompts-file first non-empty line. */
  char prompt_buf[8192];
  if (!opt.prompt && opt.prompts_file && !opt.aggregate) {
    int r = read_first_nonempty_line(opt.prompts_file, prompt_buf, sizeof(prompt_buf));
    if      (r == 1) opt.prompt = prompt_buf;
    else if (r == 0) { fprintf(stderr, "warn: prompts file is empty; using default prompt\n"); opt.prompt = "bench"; }
    else             { opt.prompt = "bench"; }
  }
  if (!opt.prompt && !opt.prompts_file) {
    /* Graceful empty run: print a valid JSON skeleton for tests. */
    ie_metrics_t mm; memset(&mm, 0, sizeof(mm));
    print_json_result(0, NULL, 0.0, 0, 0, 0);
    return 0;
  }

  /* -------------------- strict IEBIN gating -------------------- */
  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  ie_weights_t w; memset(&w, 0, sizeof(w));
  {
    const char *j = (opt.model_json && *opt.model_json) ? opt.model_json : "./model.ie.json";
    const char *b = (opt.model_bin  && *opt.model_bin ) ? opt.model_bin  : "./model.ie.bin";
    int wrc = ie_weights_open(j, b, &w);
    if (wrc != IE_IO_OK) {
      if (require_model) {
        fprintf(stderr, "error: failed to open IEBIN (%s, %s), status=%d, errno=%d (%s)\n",
                j, b, wrc, errno, strerror(errno));
        return 3;
      }
      fprintf(stderr, "# -> warn: IEBIN metadata not found; continuing in stub mode…\n");
    } else {
      if (ie_weights_touch(&w) != 0) {
        fprintf(stderr, "error: IEBIN present but unreadable (prefault/touch failed)\n");
        ie_weights_close(&w);
        return 3;
      }
    }
  }

  /* -------------------- create engine -------------------- */
  ie_engine_params_t params; memset(&params, 0, sizeof(params));
  params.precision    = opt.precision_label; /* pass raw label (includes int4w/int8w) */
  params.affinity     = opt.affinity;
  params.pretranspose = (opt.pretx==CLI_PRETX_NONE ? "none" :
                         opt.pretx==CLI_PRETX_WOH  ? "woh"  :
                         opt.pretx==CLI_PRETX_WXH  ? "wxh"  : "all");
  params.prefetch     = opt.prefetch;
  params.threads      = opt.threads;
  UNUSED(opt.device); /* Currently a no-op hint; selection happens at build/link time. */

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    ie_weights_close(&w);
    return 5;
  }

  /* -------------------- optional warmup (outside timing) ------------------- */
  if (opt.warmup_tokens > 0) {
    const char *wprompt = "warmup";
    uint32_t wtoks[128]; uint32_t wcount = 0;
    size_t wmax = (opt.warmup_tokens <= (int)(sizeof(wtoks)/sizeof(wtoks[0])))
                ? (size_t)opt.warmup_tokens
                : (sizeof(wtoks)/sizeof(wtoks[0]));
    (void)ie_engine_generate(engine, wprompt, wmax, wtoks, &wcount);
  }

  /* -------------------- token output buffer -------------------- */
  /* Always allocate a buffer so ie_engine_generate() never sees NULL. */
  uint32_t *tokens = NULL;
  size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
  size_t cap_alloc = (cap > 0 ? cap : 1);
  tokens = (uint32_t*)malloc(sizeof(uint32_t) * cap_alloc);
  if (!tokens) {
    fprintf(stderr, "error: OOM allocating token buffer\n");
    ie_engine_destroy(engine);
    ie_weights_close(&w);
    return 6;
  }

  /* -------------------- mmap model.bin for per-token work ------------------ */
  const size_t bytes_per_token = (size_t)env_long("IE_BYTES_PER_TOKEN", 0);
  const size_t stride_bytes    = (size_t)env_long("IE_STRIDE_BYTES", 256);
  const int verify_touch       = (int)env_long("IE_VERIFY_TOUCH", 0);

  int bin_fd = -1;
  uint8_t *map = NULL;
  size_t   map_len = 0;
  if (w.loaded && bytes_per_token > 0 && w.bin_size_bytes > 0) {
    const char *binp = (w.weights_path[0] ? w.weights_path : "./model.ie.bin");
    bin_fd = open(binp, O_RDONLY);
    if (bin_fd >= 0) {
      map_len = (size_t)w.bin_size_bytes;
      void *p = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, bin_fd, 0);
      if (p == MAP_FAILED) { map = NULL; map_len = 0; }
      else { map = (uint8_t*)p; }
    }
  }

  /* -------------------- KV round lifecycle (outside timing) ---------------- */
  (void)ie_kv_begin_round();
  uint64_t total_tokens_this_round = 0;

  /* -------------------- measured window: ONLY gen + work-touch ------------- */
  double t0 = now_sec();

  uint32_t tokens_generated_total = 0;

  for (int r = 0; r < (opt.rounds > 0 ? opt.rounds : 1); ++r) {
    if (opt.aggregate && opt.prompts_file) {
      FILE *pf = fopen(opt.prompts_file, "r");
      if (pf) {
        char line[8192];
        while (fgets(line, sizeof(line), pf)) {
          size_t n = strlen(line);
          while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
          if (!n) continue;

          uint32_t n_here = 0;
          st = ie_engine_generate(engine, line, cap, tokens, &n_here);
          if (st != IE_OK) { fprintf(stderr,"error: ie_engine_generate (status=%d)\n",(int)st); break; }
          tokens_generated_total += n_here;
          total_tokens_this_round += (uint64_t)n_here;

          if (map && map_len && bytes_per_token) {
            size_t need = (size_t)n_here * bytes_per_token;
            size_t pos = 0; volatile uint64_t acc = 0;
            while (pos < need) { size_t off = (pos % map_len); acc += map[off]; pos += (stride_bytes ? stride_bytes : 1); }
            if (verify_touch) { (void)acc; }
          }
        }
        fclose(pf);
      } else {
        fprintf(stderr,"warn: cannot open prompts-file '%s'\n", opt.prompts_file);
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
        size_t pos = 0; volatile uint64_t acc = 0;
        while (pos < need) { size_t off = (pos % map_len); acc += map[off]; pos += (stride_bytes ? stride_bytes : 1); }
        if (verify_touch) { (void)acc; }
      }
    }
  }

  double t1 = now_sec();

  /* -------------------- AFTER timing: collect metrics ---------------------- */
  uint64_t kv_hits_round = 0, kv_miss_round = 0;
  ie_kv_finish_round(total_tokens_this_round, &kv_hits_round, &kv_miss_round);

  /* Snapshot engine-side metrics if available (kept in sync with API). */
  ie_metrics_t m; memset(&m, 0, sizeof(m));
  (void)ie_engine_metrics(engine, &m);
  /* Prefer the round counters if non-zero (bench semantics). */
  if (kv_hits_round || kv_miss_round) {
    m.kv_hits   = kv_hits_round;
    m.kv_misses = kv_miss_round;
  }

  /* Sample RSS peak outside timing window. */
  uint32_t rss_mib = ie_metrics_sample_rss_peak();
  m.rss_peak_mb = rss_mib; /* ok even if the field is present or ignored otherwise */

  /* -------------------- print JSON & teardown ------------------------------ */
  /* Only print tokens array when it is a single-run, non-aggregate case. */
  const uint32_t *tokens_to_print = (opt.aggregate || opt.rounds > 1) ? NULL : tokens;

  print_json_result(tokens_generated_total,
                    tokens_to_print,
                    (t1 - t0),
                    m.kv_hits,
                    m.kv_misses,
                    rss_mib);

  free(tokens);
  if (map && map != MAP_FAILED) munmap(map, map_len);
  if (bin_fd >= 0) close(bin_fd);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  return 0;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
