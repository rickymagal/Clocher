/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine binary (single or file-driven).
 *
 * Stable JSON output schema:
 * {
 *   "tokens_generated": <uint>,
 *   "tokens": [...],              // single-prompt mode; empty in aggregated mode
 *   "wall_time_s": <double>,      // elapsed wall time for the run (aggregated = sum)
 *   "tps_true": <double>,         // tokens_generated / wall_time_s if engine didn't compute
 *   "latency_p50_ms": <double>,   // from engine metrics (last run for aggregated mode)
 *   "latency_p95_ms": <double>,   // from engine metrics (last run for aggregated mode)
 *   "rss_peak_mb": <size_t>,      // from engine metrics (last run for aggregated mode)
 *   "kv_hits": <uint64_t>,        // from engine metrics (last run for aggregated mode)
 *   "kv_misses": <uint64_t>       // from engine metrics (last run for aggregated mode)
 * }
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#include "ie_api.h"        /* engine create/generate/metrics/destroy, types */
#include "util_logging.h"
#include "util_metrics.h"  /* ie_metrics_t */

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* ========================================================================== */
/* Small utilities                                                            */
/* ========================================================================== */

/**
 * @brief Convert a NUL-terminated C string to long with error checking.
 *
 * On parse failure, prints an error to stderr and exits with code 2.
 *
 * @param s NUL-terminated input string.
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
 * @brief Print CLI usage information to stderr.
 */
static void usage(void) {
  fprintf(stderr,
    "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
    "                        [--threads N] [--precision fp32|bf16|fp16]\n"
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--pretranspose none|woh|wxh|all]\n"
    "                        [--prompts-file PATH] [--batch N]\n"
    "                        [--prefetch on|off|auto] [--warmup N]\n"
    "                        [--jsonl] [--help]\n\n"
    "Examples:\n"
    "  inference-engine --prompt \"hello\" --max-new 8\n"
    "  inference-engine --prompts-file prompts.txt --batch 32 --max-new 8 --prefetch on --warmup 4\n"
  );
}

/**
 * @brief Trim trailing newline characters from a string.
 *
 * @param s Mutable C string; may be NULL. Modifies in place.
 */
static void rstrip_newline(char *s) {
  if (!s) return;
  size_t n = strlen(s);
  while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r')) {
    s[--n] = '\0';
  }
}

/**
 * @brief Heap-duplicate a C string (portable strdup).
 *
 * @param src NUL-terminated input string (non-NULL).
 * @return Newly allocated duplicate, or NULL on OOM.
 */
static char *xstrdup(const char *src) {
  size_t n = strlen(src);
  char *p = (char*)malloc(n + 1);
  if (!p) return NULL;
  memcpy(p, src, n + 1);
  return p;
}

/**
 * @brief Read one line from a FILE* into a growing heap buffer (portable).
 *
 * This getline-like helper does not rely on POSIX. It returns the length of
 * the line read (excluding the terminating NUL). On EOF, returns (size_t)-1.
 *
 * @param f      Input FILE* (non-NULL).
 * @param buf    In/out pointer to a heap buffer (may be *buf==NULL initially).
 * @param cap    In/out capacity of *buf in bytes (0 if *buf==NULL).
 * @return Number of bytes read (excluding NUL) or (size_t)-1 on EOF or error.
 */
static size_t read_line_portable(FILE *f, char **buf, size_t *cap) {
  if (!f || !buf || !cap) return (size_t)-1;
  if (*buf == NULL || *cap == 0) {
    *cap = 256;
    *buf = (char*)malloc(*cap);
    if (!*buf) return (size_t)-1;
  }

  size_t len = 0;
  for (;;) {
    int ch = fgetc(f);
    if (ch == EOF) {
      if (len == 0) return (size_t)-1;
      break;
    }
    if (len + 1 >= *cap) {
      size_t newcap = (*cap < 4096) ? (*cap * 2) : (*cap + 4096);
      char *tmp = (char*)realloc(*buf, newcap);
      if (!tmp) return (size_t)-1;
      *buf = tmp;
      *cap = newcap;
    }
    (*buf)[len++] = (char)ch;
    if (ch == '\n') break;
  }
  (*buf)[len] = '\0';
  return len;
}

/* ========================================================================== */
/* CLI bag (advisory to engine defaults)                                      */
/* ========================================================================== */

/** @brief Precision options parsed from CLI. */
typedef enum { CLI_PREC_FP32=0, CLI_PREC_BF16=1, CLI_PREC_FP16=2 } cli_precision_t;
/** @brief Pretranspose options parsed from CLI. */
typedef enum { CLI_PRETX_NONE=0, CLI_PRETX_WOH=1, CLI_PRETX_WXH=2, CLI_PRETX_ALL=3 } cli_pretranspose_t;

/**
 * @brief Aggregated CLI options (advisory; engine applies internal defaults).
 */
typedef struct cli_extras {
  const char       *prompt;       /**< Single prompt text (optional if --prompts-file). */
  const char       *prompts_file; /**< File with one prompt per line (optional). */
  size_t            max_new;      /**< Max new tokens to generate per prompt. */
  int               threads;      /**< Requested threads (optional). */
  cli_precision_t   prec;         /**< Precision mode (optional). */
  const char       *affinity;     /**< Affinity policy (optional). */
  cli_pretranspose_t pretx;       /**< Pretranspose hint (optional). */
  size_t            batch;        /**< Advisory microbatch size (optional). */
  const char       *prefetch;     /**< Advisory prefetch: "on"|"off"|"auto". */
  size_t            warmup;       /**< Warmup tokens to generate before timing. */
  int               jsonl;        /**< If 1, emit one JSON per prompt (NDJSON). Default 0 = aggregated. */
} cli_extras_t;

/**
 * @brief Initialize CLI extras with defaults.
 *
 * @param e Output pointer (non-NULL).
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt       = NULL;
  e->prompts_file = NULL;
  e->max_new      = 8;
  e->threads      = 0;
  e->prec         = CLI_PREC_FP32;
  e->affinity     = "auto";
  e->pretx        = CLI_PRETX_NONE;
  e->batch        = 0;       /* 0 → engine decides or no explicit batching */
  e->prefetch     = "auto";  /* advisory */
  e->warmup       = 1;       /* sensible default for bench */
  e->jsonl        = 0;       /* aggregated by default for prompts-file */
}

/**
 * @brief Parse argv flags into @ref cli_extras_t.
 *
 * Recognized flags: --prompt, --prompts-file, --max-new, --threads, --precision,
 * --affinity, --pretranspose, --batch, --prefetch, --warmup, --jsonl, --help.
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

    } else if (strcmp(a, "--prompts-file") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      out->prompts_file = argv[++i];

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

    } else if (strcmp(a, "--batch") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[++i]); if (v < 0) { fprintf(stderr,"error: --batch >= 0\n"); return -1; }
      out->batch = (size_t)v;

    } else if (strcmp(a, "--prefetch") == 0) {
      /* Accept presence-only (→ "on") or explicit value on|off|auto. */
      if (i + 1 < argc && argv[i+1][0] != '-') {
        const char *v = argv[++i];
        if (strcmp(v, "on") == 0 || strcmp(v, "off") == 0 || strcmp(v, "auto") == 0) {
          out->prefetch = v;
        } else {
          out->prefetch = "auto";
        }
      } else {
        out->prefetch = "on";
      }

    } else if (strcmp(a, "--warmup") == 0) {
      if (i + 1 >= argc) { usage(); return -1; }
      long v = safe_atoi(argv[++i]); if (v < 0) { fprintf(stderr,"error: --warmup >= 0\n"); return -1; }
      out->warmup = (size_t)v;

    } else if (strcmp(a, "--jsonl") == 0) {
      out->jsonl = 1;

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

/* ========================================================================== */
/* Prompt file I/O (portable)                                                 */
/* ========================================================================== */

/**
 * @brief Read prompts (one per line) from a UTF-8 text file.
 *
 * Empty lines are skipped. Each returned string is heap-allocated.
 * This function is portable (no POSIX getline/strdup).
 *
 * @param path   File path.
 * @param out    Output pointer to array of C strings (allocated on success).
 * @param out_n  Output number of prompts.
 * @return 0 on success, non-zero on error.
 */
static int read_prompts_file(const char *path, char ***out, size_t *out_n) {
  if (!path || !out || !out_n) return -1;
  *out = NULL; *out_n = 0;

  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "error: failed to open prompts file: %s\n", path);
    return -2;
  }

  size_t cap = 64;
  size_t n = 0;
  char **arr = (char**)calloc(cap, sizeof(char*));
  if (!arr) { fclose(f); return -3; }

  char *line = NULL;
  size_t bufcap = 0;
  for (;;) {
    size_t got = read_line_portable(f, &line, &bufcap);
    if (got == (size_t)-1) break;
    rstrip_newline(line);
    if (line[0] == '\0') continue; /* skip empty */

    if (n == cap) {
      cap *= 2;
      char **tmp = (char**)realloc(arr, cap * sizeof(char*));
      if (!tmp) { fclose(f); free(line); for (size_t i=0;i<n;++i) free(arr[i]); free(arr); return -4; }
      arr = tmp;
    }
    arr[n] = xstrdup(line);
    if (!arr[n]) { fclose(f); free(line); for (size_t i=0;i<n;++i) free(arr[i]); free(arr); return -5; }
    n++;
  }
  free(line);
  fclose(f);

  *out = arr;
  *out_n = n;
  return 0;
}

/**
 * @brief Free a list of prompts previously returned by @ref read_prompts_file.
 *
 * @param list Pointer to array of heap-allocated C strings (may be NULL).
 * @param n    Number of entries in @p list.
 */
static void free_prompts_list(char **list, size_t n) {
  if (!list) return;
  for (size_t i = 0; i < n; ++i) free(list[i]);
  free(list);
}

/* ========================================================================== */
/* Engine helpers                                                             */
/* ========================================================================== */

/**
 * @brief Perform a warmup call (generate @p warmup_tokens tokens) if requested.
 *
 * @param engine        Engine handle (non-NULL).
 * @param warmup_tokens Number of tokens to generate in warmup (0 to skip).
 */
static void run_warmup(ie_engine_t *engine, size_t warmup_tokens) {
  if (!engine || warmup_tokens == 0) return;
  const char *wprompt = "warmup";
  const size_t cap = (warmup_tokens > 1024) ? 1024 : warmup_tokens;
  uint32_t *buf = (uint32_t*) (cap ? malloc(sizeof(uint32_t) * cap) : NULL);
  uint32_t got = 0;
  (void)ie_engine_generate(engine, wprompt, warmup_tokens,
                           (buf ? buf : (uint32_t[]){0}), &got);
  if (buf) free(buf);
}

/**
 * @brief Per-run stats container (local measurements + engine metrics).
 */
typedef struct run_stats {
  uint32_t    tokens_generated; /**< Number of tokens emitted by the run. */
  double      wall_time_s;      /**< Local measured wall time. */
  ie_metrics_t m;               /**< Snapshot of engine metrics after the run. */
} run_stats_t;

/**
 * @brief Execute a generation and capture stats without printing.
 *
 * @param engine   Engine handle (non-NULL).
 * @param prompt   Prompt text (non-NULL).
 * @param max_new  Maximum new tokens to generate.
 * @param out      Output stats (non-NULL).
 */
static void do_generate(ie_engine_t *engine,
                        const char *prompt,
                        size_t max_new,
                        run_stats_t *out) {
  memset(out, 0, sizeof(*out));

  uint32_t *tokens = NULL;
  if (max_new > 0) {
    tokens = (uint32_t*)malloc(sizeof(uint32_t) * max_new);
    if (!tokens) {
      fprintf(stderr, "error: OOM allocating token buffer\n");
      return;
    }
  }

  const clock_t t0 = clock();
  uint32_t n_tok = 0;
  (void)ie_engine_generate(engine, prompt, max_new,
                           (tokens ? tokens : (uint32_t[]){0}), &n_tok);
  const clock_t t1 = clock();

  out->tokens_generated = n_tok;
  out->wall_time_s = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

  memset(&out->m, 0, sizeof(out->m));
  (void)ie_engine_metrics(engine, &out->m);

  free(tokens);
}

/**
 * @brief Print a single JSON object from @ref run_stats_t and optional tokens.
 *
 * @param s           Stats to print.
 * @param tokens      Token buffer (may be NULL if not needed).
 * @param n_tok       Number of tokens in @p tokens.
 */
static void print_json_from_stats(const run_stats_t *s,
                                  const uint32_t *tokens,
                                  uint32_t n_tok) {
  const double tps_fallback = (s->wall_time_s > 0.0)
                            ? ((double)s->tokens_generated / s->wall_time_s)
                            : 0.0;

  fprintf(stdout, "{\"tokens_generated\": %u,", (unsigned)s->tokens_generated);

  fprintf(stdout, "\"tokens\":[");
  for (uint32_t i = 0; i < n_tok; ++i) {
    fprintf(stdout, "%u%s", tokens[i], (i + 1 < n_tok) ? "," : "");
  }
  fprintf(stdout, "],");

  fprintf(stdout,
      "\"wall_time_s\": %.6f,"
      "\"tps_true\": %.6f,"
      "\"latency_p50_ms\": %.3f,"
      "\"latency_p95_ms\": %.3f,"
      "\"rss_peak_mb\": %zu,"
      "\"kv_hits\": %llu,"
      "\"kv_misses\": %llu}\n",
      s->wall_time_s,
      (s->m.tps_true > 0.0 ? s->m.tps_true : tps_fallback),
      s->m.latency_p50_ms,
      s->m.latency_p95_ms,
      s->m.rss_peak_mb,
      (unsigned long long)s->m.kv_hits,
      (unsigned long long)s->m.kv_misses);
}

/**
 * @brief Run generation for a single prompt and print one JSON line.
 *
 * @param engine   Engine handle (non-NULL).
 * @param prompt   Prompt text (non-NULL).
 * @param max_new  Maximum new tokens to generate.
 */
static void run_single(ie_engine_t *engine, const char *prompt, size_t max_new) {
  run_stats_t s;
  do_generate(engine, prompt, max_new, &s);

  /* In single mode we don't return tokens, just emit counts + metrics. */
  print_json_from_stats(&s, NULL, 0);
}

/**
 * @brief Run generation for all prompts in a file.
 *
 * If @p jsonl is 1, prints one JSON per prompt (NDJSON).
 * If @p jsonl is 0 (default), prints a single aggregated JSON object:
 *   - tokens_generated: sum across prompts
 *   - wall_time_s: sum of per-prompt wall times
 *   - tps_true: tokens_generated / wall_time_s (unless engine supplies non-zero)
 *   - other fields are from the last run's engine metrics snapshot
 *
 * @param params    Engine params (may be zero-initialized; engine sets defaults).
 * @param path      Path to prompts file (one prompt per line).
 * @param max_new   Maximum new tokens to generate per prompt.
 * @param batch     Advisory microbatch size (currently unused here).
 * @param prefetch  Advisory prefetch policy string ("on"|"off"|"auto").
 * @param warmup    Number of warmup tokens before timing.
 * @param jsonl     If non-zero, emit one JSON per prompt; else a single aggregate.
 * @return 0 on success; non-zero on failure to create engine or read file.
 */
static int run_from_file(const ie_engine_params_t *params,
                         const char *path,
                         size_t max_new,
                         size_t batch,
                         const char *prefetch,
                         size_t warmup,
                         int jsonl) {
  char **prompts = NULL;
  size_t n = 0;
  if (read_prompts_file(path, &prompts, &n) != 0) {
    return 5;
  }
  UNUSED(batch);
  UNUSED(prefetch);

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(params, &engine);
  if (st != 0 || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    free_prompts_list(prompts, n);
    return 3;
  }

  run_warmup(engine, warmup);

  if (jsonl) {
    for (size_t i = 0; i < n; ++i) {
      run_single(engine, prompts[i], max_new);
    }
  } else {
    /* Aggregate into a single JSON object. */
    uint64_t tokens_total = 0;
    double   wall_total   = 0.0;
    ie_metrics_t last_m;
    memset(&last_m, 0, sizeof(last_m));

    for (size_t i = 0; i < n; ++i) {
      run_stats_t s;
      do_generate(engine, prompts[i], max_new, &s);
      tokens_total += s.tokens_generated;
      wall_total   += s.wall_time_s;
      last_m        = s.m; /* keep the last snapshot */
    }

    /* Compose aggregate JSON (tokens array intentionally empty). */
    const double tps_fallback = (wall_total > 0.0)
                              ? ((double)tokens_total / wall_total)
                              : 0.0;

    fprintf(stdout,
      "{\"tokens_generated\": %llu,"
       "\"tokens\": [],"
       "\"wall_time_s\": %.6f,"
       "\"tps_true\": %.6f,"
       "\"latency_p50_ms\": %.3f,"
       "\"latency_p95_ms\": %.3f,"
       "\"rss_peak_mb\": %zu,"
       "\"kv_hits\": %llu,"
       "\"kv_misses\": %llu}\n",
       (unsigned long long)tokens_total,
       wall_total,
       (last_m.tps_true > 0.0 ? last_m.tps_true : tps_fallback),
       last_m.latency_p50_ms,
       last_m.latency_p95_ms,
       last_m.rss_peak_mb,
       (unsigned long long)last_m.kv_hits,
       (unsigned long long)last_m.kv_misses);
  }

  ie_engine_destroy(engine);
  free_prompts_list(prompts, n);
  return 0;
}

/* ========================================================================== */
/* Main                                                                       */
/* ========================================================================== */

/**
 * @brief Program entry point.
 *
 * Modes:
 *  - Single prompt:   --prompt "text" (or positional) → one JSON line.
 *  - File of prompts: --prompts-file path → one JSON line (aggregated), or
 *                     add --jsonl for one JSON per prompt (NDJSON).
 *
 * If neither prompt nor prompts-file is provided, prints a zero-token JSON
 * line (used by tests).
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; non-zero on error.
 */
int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  if (!opt.prompt && !opt.prompts_file) {
    fprintf(stdout,
      "{\"tokens_generated\": %d,"
       "\"tokens\": [],"
       "\"wall_time_s\": %.6f,"
       "\"tps_true\": %.6f,"
       "\"latency_p50_ms\": %.3f,"
       "\"latency_p95_ms\": %.3f,"
       "\"rss_peak_mb\": %zu,"
       "\"kv_hits\": %llu,"
       "\"kv_misses\": %llu}\n",
      0, 0.0, 0.0, 0.0, 0.0,
      (size_t)0,
      (unsigned long long)0,
      (unsigned long long)0);
    return 0;
  }

  /* Engine params: zero-init; engine applies its own defaults. */
  ie_engine_params_t params;
  memset(&params, 0, sizeof(params));
  /* Future: map opt.threads/precision/affinity/pretranspose/prefetch/batch → params. */
  UNUSED(opt.prec);
  UNUSED(opt.affinity);
  UNUSED(opt.pretx);
  UNUSED(opt.prefetch);
  UNUSED(opt.batch);
  UNUSED(opt.threads);

  if (opt.prompts_file) {
    return run_from_file(&params, opt.prompts_file, opt.max_new,
                         opt.batch, opt.prefetch, opt.warmup, opt.jsonl);
  }

  /* Single prompt path */
  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != 0 || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    return 3;
  }

  run_warmup(engine, opt.warmup);
  run_single(engine, opt.prompt, opt.max_new);

  ie_engine_destroy(engine);
  return 0;
}
