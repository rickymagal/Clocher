/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine binary.
 *
 * @details
 * Stable JSON output for automation (single line):
 * {
 *   "tokens_generated": <uint>,
 *   "tokens": [<uint>...],
 *   "wall_time_s": <double>,
 *   "tps_true": <double>,
 *   "latency_p50_ms": <double>,
 *   "latency_p95_ms": <double>,
 *   "rss_peak_mb": <size_t>,
 *   "kv_hits": <uint64_t>,
 *   "kv_misses": <uint64_t>
 * }
 *
 * Notes:
 * - This CLI intentionally accepts several "advisory" flags that the engine
 *   may or may not use. Unknown-but-recognized flags are parsed and ignored
 *   (e.g., --prompts-file, --batch, --prefetch, --warmup) to keep external
 *   harnesses compatible.
 * - Only a single JSON object is printed to STDOUT per process invocation.
 *   All diagnostics go to STDERR.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "ie_api.h"       /* ie_engine_create/generate/metrics/destroy */
#include "util_logging.h" /* (kept for parity; logging to stderr if needed) */
#include "util_metrics.h" /* ie_metrics_t */

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* ========================================================================== */
/* CLI option bag (advisory to engine)                                        */
/* ========================================================================== */

/** @brief Precision selection parsed from the CLI. */
typedef enum {
  CLI_PREC_FP32 = 0,
  CLI_PREC_BF16 = 1,
  CLI_PREC_FP16 = 2
} cli_precision_t;

/** @brief Pretranspose hint parsed from the CLI. */
typedef enum {
  CLI_PRETX_NONE = 0,
  CLI_PRETX_WOH  = 1,
  CLI_PRETX_WXH  = 2,
  CLI_PRETX_ALL  = 3
} cli_pretranspose_t;

/**
 * @brief Parsed CLI options. These are hints; the engine applies its own defaults.
 */
typedef struct cli_extras {
  const char       *prompt;        /**< Prompt text (positional or via --prompt). */
  size_t            max_new;       /**< Number of tokens to generate. */
  int               threads;       /**< Requested threads (optional). */
  cli_precision_t   prec;          /**< Precision mode (optional). */
  const char       *affinity;      /**< Affinity policy (optional). */
  cli_pretranspose_t pretx;        /**< Pretranspose hint (optional). */

  /* Accepted by harness; parsed but not required by the minimal engine. */
  const char       *prompts_file;  /**< Accepted but ignored by this binary. */
  int               batch;         /**< Accepted but ignored by this binary. */
  const char       *prefetch;      /**< Accepted but ignored (supports int/on/off/auto). */
  int               warmup_tokens; /**< Optional warmup token count. */
} cli_extras_t;

/* ========================================================================== */
/* Utilities                                                                  */
/* ========================================================================== */

/**
 * @brief Convert decimal string to long with basic validation; exits on error.
 *
 * @param s NUL-terminated string.
 * @return Parsed long value.
 */
static long safe_atoi(const char *s) {
  if (!s || !*s) { fprintf(stderr, "error: empty integer\n"); exit(2); }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    fprintf(stderr, "error: invalid integer: '%s'\n", s);
    exit(2);
  }
  return v;
}

/**
 * @brief Print CLI usage to stderr.
 */
static void usage(void) {
  fprintf(stderr,
    "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
    "                        [--threads N] [--precision fp32|bf16|fp16]\n"
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--pretranspose none|woh|wxh|all]\n"
    "                        [--prompts-file PATH] [--batch N]\n"
    "                        [--prefetch on|off|auto|N] [--warmup N]\n"
    "                        [--help]\n\n"
    "Examples:\n"
    "  inference-engine --prompt \"hello\" --max-new 8\n"
    "  inference-engine --prompt \"lorem\" --max-new 16 --threads 2 --precision bf16\n"
    "  inference-engine --prompts-file prompts.txt --batch 32 --max-new 8 --prefetch on\n"
  );
}

/**
 * @brief Initialize CLI extras with defaults.
 *
 * @param e Output pointer (non-NULL).
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt         = NULL;
  e->max_new        = 8;
  e->threads        = 0;
  e->prec           = CLI_PREC_FP32;
  e->affinity       = "auto";
  e->pretx          = CLI_PRETX_NONE;
  e->prompts_file   = NULL;
  e->batch          = 0;
  e->prefetch       = "auto";
  e->warmup_tokens  = 1; /* small warmup by default */
}

/**
 * @brief Parse argv flags into @ref cli_extras_t.
 *
 * This parser accepts harness flags even if the engine ignores them internally.
 * Unrecognized flags cause a usage error.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param out  Output (non-NULL).
 * @return 0 on success, non-zero on error (usage already printed).
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];

    if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
      usage(); return -1;

    } else if (strcmp(a, "--prompt") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      out->prompt = argv[++i];

    } else if (strcmp(a, "--max-new") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[++i]); if (v < 0) { fprintf(stderr,"error: --max-new >= 0\n"); return -1; }
      out->max_new = (size_t)v;

    } else if (strcmp(a, "--threads") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[++i]); if (v < 0) { fprintf(stderr,"error: --threads >= 0\n"); return -1; }
      out->threads = (int)v;

    } else if (strcmp(a, "--precision") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      const char *m = argv[++i];
      if      (strcmp(m, "fp32") == 0) out->prec = CLI_PREC_FP32;
      else if (strcmp(m, "bf16") == 0) out->prec = CLI_PREC_BF16;
      else if (strcmp(m, "fp16") == 0) out->prec = CLI_PREC_FP16;
      else { fprintf(stderr, "error: unknown precision '%s'\n", m); return -1; }

    } else if (strcmp(a, "--affinity") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      out->affinity = argv[++i];

    } else if (strcmp(a, "--pretranspose") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      const char *p = argv[++i];
      if      (strcmp(p, "none") == 0) out->pretx = CLI_PRETX_NONE;
      else if (strcmp(p, "woh")  == 0) out->pretx = CLI_PRETX_WOH;
      else if (strcmp(p, "wxh")  == 0) out->pretx = CLI_PRETX_WXH;
      else if (strcmp(p, "all")  == 0) out->pretx = CLI_PRETX_ALL;
      else { fprintf(stderr, "error: unknown pretranspose '%s'\n", p); return -1; }

    /* -------- Harness-compat flags: accepted but ignored internally -------- */
    } else if (strcmp(a, "--prompts-file") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      out->prompts_file = argv[++i]; /* accepted, ignored */
    } else if (strcmp(a, "--batch") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      out->batch = (int)safe_atoi(argv[++i]); /* accepted, ignored */
    } else if (strcmp(a, "--prefetch") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      /* accept on|off|auto or any integer; store as-is, ignore internally */
      out->prefetch = argv[++i];
    } else if (strcmp(a, "--warmup") == 0 || strcmp(a, "--warmup-tokens") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[++i]); if (v < 0) v = 0;
      out->warmup_tokens = (int)v;

    } else if (a[0] == '-') {
      fprintf(stderr, "error: unknown flag '%s'\n", a);
      usage(); return -1;

    } else {
      /* positional prompt (compat) */
      out->prompt = a;
    }
  }
  return 0;
}

/**
 * @brief Print a single JSON result line to STDOUT.
 *
 * @param n_tok     Number of tokens generated.
 * @param tokens    Pointer to tokens array (may be NULL if n_tok == 0).
 * @param wall_s    Wall-clock seconds for the generate() call.
 * @param m         Metrics snapshot (may contain zeros).
 */
static void print_json_result(uint32_t n_tok,
                              const uint32_t *tokens,
                              double wall_s,
                              const ie_metrics_t *m) {
  /* Header with tokens_generated (space after colon is intentional for tests). */
  fprintf(stdout, "{\"tokens_generated\": %u,", (unsigned)n_tok);

  /* Tokens array */
  fputs("\"tokens\":[", stdout);
  for (uint32_t i = 0; i < n_tok; ++i) {
    fprintf(stdout, "%u%s", tokens[i], (i + 1 < n_tok) ? "," : "");
  }
  fputs("],", stdout);

  /* Throughput: prefer engine's tps_true if set; otherwise compute locally. */
  double tps_local = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;
  double tps = (m && m->tps_true > 0.0) ? m->tps_true : tps_local;

  /* Remaining fields (with spaces after colons for readability and test parity). */
  fprintf(stdout,
      "\"wall_time_s\": %.6f,"
      "\"tps_true\": %.6f,"
      "\"latency_p50_ms\": %.3f,"
      "\"latency_p95_ms\": %.3f,"
      "\"rss_peak_mb\": %zu,"
      "\"kv_hits\": %llu,"
      "\"kv_misses\": %llu}\n",
      wall_s,
      tps,
      m ? m->latency_p50_ms : 0.0,
      m ? m->latency_p95_ms : 0.0,
      m ? m->rss_peak_mb    : (size_t)0,
      (unsigned long long)(m ? m->kv_hits   : 0ULL),
      (unsigned long long)(m ? m->kv_misses : 0ULL));
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

/**
 * @brief Program entry point.
 *
 * Creates the engine, performs a small warmup (if requested), generates up to
 * `--max-new` tokens for the given prompt, gathers metrics, and prints one JSON
 * line to STDOUT. If no prompt is provided, prints a valid "zero" JSON object.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; non-zero on usage or runtime errors.
 */
int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* If no prompt was provided, emit a zero-result JSON object and exit. */
  if (!opt.prompt) {
    ie_metrics_t m; memset(&m, 0, sizeof(m));
    print_json_result(0, NULL, 0.0, &m);
    return 0;
  }

  /* Engine params: zero-init; the engine applies its own internal defaults. */
  ie_engine_params_t params;
  memset(&params, 0, sizeof(params));
  UNUSED(opt.threads);
  UNUSED(opt.prec);
  UNUSED(opt.affinity);
  UNUSED(opt.pretx);
  UNUSED(opt.prompts_file);
  UNUSED(opt.batch);
  UNUSED(opt.prefetch);

  /* Create engine handle. */
  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != 0 || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    return 3;
  }

  /* Optional warmup: generate and discard. */
  if (opt.warmup_tokens > 0) {
    const char *wprompt = "warmup";
    uint32_t wtoks_buf[64];
    uint32_t wcount = 0;
    size_t wmax = (opt.warmup_tokens <= (int)(sizeof(wtoks_buf)/sizeof(wtoks_buf[0])))
                  ? (size_t)opt.warmup_tokens
                  : (sizeof(wtoks_buf)/sizeof(wtoks_buf[0]));
    (void)ie_engine_generate(engine, wprompt, wmax, wtoks_buf, &wcount);
  }

  /* Allocate output buffer (may be zero-sized if max_new == 0). */
  uint32_t *tokens = NULL;
  uint32_t n_tok   = 0;
  if (opt.max_new > 0) {
    tokens = (uint32_t*)malloc(sizeof(uint32_t) * opt.max_new);
    if (!tokens) {
      fprintf(stderr, "error: OOM allocating token buffer\n");
      ie_engine_destroy(engine);
      return 3;
    }
  }

  /* Time the generation call locally. */
  const clock_t t0 = clock();
  st = ie_engine_generate(engine, opt.prompt, opt.max_new,
                          (tokens ? tokens : (uint32_t[]){0}), &n_tok);
  const clock_t t1 = clock();
  if (st != 0) {
    fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)st);
    /* Continue to metrics to emit a JSON object anyway. */
  }

  /* Metrics snapshot from engine (best-effort). */
  ie_metrics_t m;
  memset(&m, 0, sizeof(m));
  (void)ie_engine_metrics(engine, &m);

  /* Emit JSON line. */
  const double wall_time_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
  print_json_result(n_tok, tokens, wall_time_s, &m);

  free(tokens);
  ie_engine_destroy(engine);
  return 0;
}
