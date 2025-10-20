/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine binary.
 *
 * This program invokes the C core (ie_api) to generate tokens for either a
 * single prompt or a list of prompts read from a file. It prints a single JSON
 * object with wall-clock timing taken from CLOCK_MONOTONIC — there is NO
 * artificial clamping or flooring of times here.
 *
 * Output JSON (single object):
 * {
 *   "tokens_generated": <uint64>,        // total new tokens generated
 *   "wall_time_s": <double>,             // total wall time in seconds
 *   "tps_true": <double>,                // tokens_generated / wall_time_s
 *   "latency_p50_ms": <double>,          // from ie_engine_metrics (0.0 if unavailable)
 *   "latency_p95_ms": <double>,          // from ie_engine_metrics (0.0 if unavailable)
 *   "rss_peak_mb": <size_t>,             // from ie_engine_metrics (0 if unavailable)
 *   "kv_hits": <uint64>,                 // from ie_engine_metrics (0 if unavailable)
 *   "kv_misses": <uint64>                // from ie_engine_metrics (0 if unavailable)
 * }
 *
 * Flags:
 *   --prompt TEXT             : Single prompt (mutually exclusive with --prompts-file)
 *   --prompts-file PATH       : Read prompts from file; by default only the first non-empty line
 *   --aggregate               : Iterate ALL non-empty lines in --prompts-file and accumulate totals
 *   --max-new N               : Max new tokens per run (default 8)
 *   --threads N               : Thread hint (parsed; core may ignore)
 *   --precision fp32|bf16|fp16
 *   --affinity auto|compact|scatter
 *   --pretranspose none|woh|wxh|all
 *   --batch N, --prefetch ... : Harness-compat (parsed; not enforced here)
 *   --warmup N                : Optional warm-up new tokens with fixed prompt "warmup"
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ie_api.h"
#include "util_logging.h"
#include "util_metrics.h"

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* ========================================================================== */
/* Timing                                                                     */
/* ========================================================================== */

/**
 * @brief Monotonic wall-clock in seconds (CLOCK_MONOTONIC).
 * @return Seconds as double.
 */
static double ie_now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ========================================================================== */
/* CLI options                                                                */
/* ========================================================================== */

/** @brief Precision hint. */
typedef enum { CLI_PREC_FP32=0, CLI_PREC_BF16=1, CLI_PREC_FP16=2 } cli_precision_t;

/** @brief Pretranspose hint. */
typedef enum { CLI_PRETX_NONE=0, CLI_PRETX_WOH=1, CLI_PRETX_WXH=2, CLI_PRETX_ALL=3 } cli_pretranspose_t;

/**
 * @brief Parsed CLI options and harness-compat flags.
 */
typedef struct cli_extras {
  const char        *prompt;        /**< Direct prompt text. */
  size_t             max_new;       /**< Max new tokens per call. */
  int                threads;       /**< Thread hint for core. */
  cli_precision_t    prec;          /**< Precision hint. */
  const char        *affinity;      /**< Affinity policy. */
  cli_pretranspose_t pretx;         /**< Pretranspose policy. */

  /* Harness-compat (parsed; may be ignored by core) */
  const char        *prompts_file;  /**< File with prompts. */
  int                batch;         /**< Batch size (compat). */
  const char        *prefetch;      /**< Prefetch policy (compat). */
  int                warmup_tokens; /**< Warm-up tokens with "warmup". */

  int                aggregate;     /**< If non-zero, iterate all lines in prompts_file. */
} cli_extras_t;

/* ========================================================================== */
/* CLI parsing                                                                */
/* ========================================================================== */

/**
 * @brief Strict base-10 integer parser; exits(2) on error.
 */
static long ie_parse_long(const char *s) {
  if (!s || !*s) { fprintf(stderr, "error: empty integer\n"); exit(2); }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end) { fprintf(stderr, "error: invalid integer: '%s'\n", s); exit(2); }
  return v;
}

/**
 * @brief Print CLI usage help.
 */
static void ie_usage(void) {
  fprintf(stderr,
    "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
    "                        [--threads N] [--precision fp32|bf16|fp16]\n"
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--pretranspose none|woh|wxh|all]\n"
    "                        [--prompts-file PATH] [--aggregate]\n"
    "                        [--batch N] [--prefetch on|off|auto|N]\n"
    "                        [--warmup N] [--help]\n");
}

/**
 * @brief Initialize default CLI options.
 */
static void cli_defaults(cli_extras_t *e) {
  e->prompt         = NULL;
  e->max_new        = 8;
  e->threads        = 0;
  e->prec           = CLI_PREC_FP32;
  e->affinity       = "auto";
  e->pretx          = CLI_PRETX_NONE;

  e->prompts_file   = NULL;
  e->batch          = 0;
  e->prefetch       = "auto";
  e->warmup_tokens  = 0;

  e->aggregate      = 0;
}

/**
 * @brief Parse argv flags into @ref cli_extras_t.
 * @return 0 on success; -1 if help was requested.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_defaults(out);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a,"--help") || !strcmp(a,"-h"))          { ie_usage(); return -1; }
    else if (!strcmp(a,"--prompt"))                      { if (++i>=argc){ie_usage();return -1;} out->prompt = argv[i]; }
    else if (!strcmp(a,"--max-new"))                     { if (++i>=argc){ie_usage();return -1;} long v=ie_parse_long(argv[i]); if(v<0){fprintf(stderr,"error: --max-new >= 0\n");return -1;} out->max_new=(size_t)v; }
    else if (!strcmp(a,"--threads"))                     { if (++i>=argc){ie_usage();return -1;} long v=ie_parse_long(argv[i]); if(v<0){fprintf(stderr,"error: --threads >= 0\n");return -1;} out->threads=(int)v; }
    else if (!strcmp(a,"--precision"))                   { if (++i>=argc){ie_usage();return -1;} const char*m=argv[i]; if(!strcmp(m,"fp32")) out->prec=CLI_PREC_FP32; else if(!strcmp(m,"bf16")) out->prec=CLI_PREC_BF16; else if(!strcmp(m,"fp16")) out->prec=CLI_PREC_FP16; else { fprintf(stderr,"error: unknown precision '%s'\n", m); return -1; } }
    else if (!strcmp(a,"--affinity"))                    { if (++i>=argc){ie_usage();return -1;} out->affinity = argv[i]; }
    else if (!strcmp(a,"--pretranspose"))                { if (++i>=argc){ie_usage();return -1;} const char*p=argv[i]; if(!strcmp(p,"none")) out->pretx=CLI_PRETX_NONE; else if(!strcmp(p,"woh")) out->pretx=CLI_PRETX_WOH; else if(!strcmp(p,"wxh")) out->pretx=CLI_PRETX_WXH; else if(!strcmp(p,"all")) out->pretx=CLI_PRETX_ALL; else { fprintf(stderr,"error: unknown pretranspose '%s'\n", p); return -1; } }
    else if (!strcmp(a,"--prompts-file"))                { if (++i>=argc){ie_usage();return -1;} out->prompts_file = argv[i]; }
    else if (!strcmp(a,"--aggregate"))                   { out->aggregate = 1; }
    else if (!strcmp(a,"--batch"))                       { if (++i>=argc){ie_usage();return -1;} out->batch = (int)ie_parse_long(argv[i]); }
    else if (!strcmp(a,"--prefetch"))                    { if (++i>=argc){ie_usage();return -1;} out->prefetch = argv[i]; }
    else if (!strcmp(a,"--warmup") || !strcmp(a,"--warmup-tokens")) {
      if (++i>=argc){ie_usage();return -1;}
      long v = ie_parse_long(argv[i]); if (v < 0) v = 0; out->warmup_tokens = (int)v;
    }
    else if (a[0] == '-') { fprintf(stderr,"error: unknown flag '%s'\n", a); ie_usage(); return -1; }
    else { out->prompt = a; } /* positional */
  }
  return 0;
}

/* ========================================================================== */
/* File helpers                                                               */
/* ========================================================================== */

/**
 * @brief Read the first non-empty line from a text file into @p buf.
 * @param path File path.
 * @param buf Output buffer.
 * @param bufsz Capacity of @p buf including the final NUL.
 * @return 1 on success, 0 if none found, -1 on IO error (warn printed).
 */
static int read_first_nonempty_line(const char *path, char *buf, size_t bufsz) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "warn: cannot open prompts file '%s': %s\n", path, strerror(errno)); return -1; }
  int ok = 0;
  while (fgets(buf, (int)bufsz, f)) {
    size_t n = strlen(buf);
    while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';
    if (n == 0) continue;
    ok = 1; break;
  }
  fclose(f);
  return ok ? 1 : 0;
}

/**
 * @brief Callback type for iterating lines of a file.
 */
typedef int (*line_cb_t)(const char *line, void *user);

/**
 * @brief Iterate all non-empty lines from a file and call @p cb for each line.
 * @return 0 on success, non-zero on error or if callback returns non-zero.
 */
static int for_each_nonempty_line(const char *path, line_cb_t cb, void *user) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open prompts file '%s': %s\n", path, strerror(errno));
    return -1;
  }
  char *line = NULL; size_t n = 0; ssize_t got;
  int rc = 0;
  while ((got = getline(&line, &n, f)) != -1) {
    while (got > 0 && (line[got-1] == '\n' || line[got-1] == '\r')) line[--got] = '\0';
    if (got == 0) continue;
    rc = cb(line, user);
    if (rc != 0) break;
  }
  free(line);
  fclose(f);
  return rc;
}

/* ========================================================================== */
/* JSON emitter                                                               */
/* ========================================================================== */

/**
 * @brief Emit a single JSON line to stdout with totals.
 * @param tokens_total Total tokens generated across runs.
 * @param wall_s_total Total wall time across runs (seconds).
 * @param m_in Engine metrics pointer (may be NULL).
 *
 * No clamping: if @p wall_s_total <= 0, tps_true is 0.0.
 */
static void emit_json(uint64_t tokens_total,
                      double wall_s_total,
                      const ie_metrics_t *m_in) {
  double tps = (wall_s_total > 0.0) ? ((double)tokens_total / wall_s_total) : 0.0;

  double p50 = (m_in ? m_in->latency_p50_ms : 0.0);
  double p95 = (m_in ? m_in->latency_p95_ms : 0.0);

  fprintf(stdout,
    "{"
      "\"tokens_generated\": %" PRIu64 ","
      "\"wall_time_s\": %.6f,"
      "\"tps_true\": %.6f,"
      "\"latency_p50_ms\": %.3f,"
      "\"latency_p95_ms\": %.3f,"
      "\"rss_peak_mb\": %zu,"
      "\"kv_hits\": %" PRIu64 ","
      "\"kv_misses\": %" PRIu64
    "}\n",
    (uint64_t)tokens_total,
    wall_s_total,
    tps,
    p50,
    p95,
    (size_t)(m_in ? m_in->rss_peak_mb : 0),
    (uint64_t)(m_in ? m_in->kv_hits   : 0ULL),
    (uint64_t)(m_in ? m_in->kv_misses : 0ULL)
  );
}

/* ========================================================================== */
/* Aggregate callback (no nested functions)                                   */
/* ========================================================================== */

/** @brief Context for aggregate iteration. */
typedef struct agg_ctx {
  ie_engine_t *eng;       /**< Engine handle. */
  size_t       cap;       /**< Max tokens per call. */
  uint32_t    *buf;       /**< Optional token buffer (cap elements) or NULL. */
  uint64_t    *tok_total; /**< Accumulator for tokens. */
  double      *wall_total;/**< Accumulator for wall time in seconds. */
} agg_ctx_t;

/**
 * @brief Callback used by for_each_nonempty_line during aggregate mode.
 * @param line Prompt line.
 * @param user Pointer to @ref agg_ctx_t.
 * @return 0 to continue; non-zero to stop with error.
 */
static int aggregate_cb(const char *line, void *user) {
  agg_ctx_t *c = (agg_ctx_t*)user;
  uint32_t got = 0;
  double t0 = ie_now_sec();
  ie_status_t rc = ie_engine_generate(c->eng, line, c->cap,
                        (c->buf ? c->buf : (uint32_t[]){0}), &got);
  double t1 = ie_now_sec();
  if (rc != 0) {
    fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)rc);
    return (int)rc;
  }
  *(c->tok_total)  += (uint64_t)got;
  *(c->wall_total) += (t1 - t0);
  return 0;
}

/* ========================================================================== */
/* main                                                                       */
/* ========================================================================== */

/**
 * @brief Program entry point.
 * @return Process exit code (0 on success).
 */
int main(int argc, char **argv) {
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* Resolve prompt. If --aggregate is not set and --prompts-file exists,
     we take the first non-empty line from that file (compat behavior). */
  char prompt_buf[8192];
  if (!opt.prompt && opt.prompts_file && !opt.aggregate) {
    int r = read_first_nonempty_line(opt.prompts_file, prompt_buf, sizeof(prompt_buf));
    if (r == 1)      { opt.prompt = prompt_buf; }
    else if (r == 0) { fprintf(stderr, "warn: prompts file is empty; using default prompt\n"); opt.prompt = "bench"; }
    else             { opt.prompt = "bench"; }
  }

  /* Create engine (hints parsed above; core may ignore in this build). */
  ie_engine_params_t params; memset(&params, 0, sizeof(params));
  UNUSED(opt.threads); UNUSED(opt.prec); UNUSED(opt.affinity);
  UNUSED(opt.pretx);   UNUSED(opt.batch); UNUSED(opt.prefetch);

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != 0 || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    return 3;
  }

  /* Optional warmup. */
  if (opt.warmup_tokens > 0) {
    const char *wprompt = "warmup";
    size_t cap = (opt.warmup_tokens > 0 ? (size_t)opt.warmup_tokens : 1);
    if (cap > 4096) cap = 4096;
    uint32_t *wtoks = (uint32_t*)malloc(sizeof(uint32_t) * cap);
    if (wtoks) {
      uint32_t wcount = 0;
      (void)ie_engine_generate(engine, wprompt, cap, wtoks, &wcount);
      free(wtoks);
    }
  }

  uint64_t     total_tokens = 0;
  double       total_wall   = 0.0;
  ie_metrics_t m_last; memset(&m_last, 0, sizeof(m_last));

  if (opt.aggregate) {
    /* Aggregate all non-empty lines in --prompts-file. */
    if (!opt.prompts_file) {
      fprintf(stderr, "error: --aggregate requires --prompts-file\n");
      ie_engine_destroy(engine);
      return 2;
    }
    size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
    uint32_t *tokbuf = NULL;
    if (cap > 0) {
      tokbuf = (uint32_t*)malloc(sizeof(uint32_t) * cap);
      if (!tokbuf) { fprintf(stderr,"error: OOM\n"); ie_engine_destroy(engine); return 3; }
    }
    agg_ctx_t ctx;
    ctx.eng = engine; ctx.cap = cap; ctx.buf = tokbuf;
    ctx.tok_total = &total_tokens; ctx.wall_total = &total_wall;

    int rc = for_each_nonempty_line(opt.prompts_file, aggregate_cb, &ctx);
    if (tokbuf) free(tokbuf);
    if (rc != 0) { ie_engine_destroy(engine); return 3; }
    (void)ie_engine_metrics(engine, &m_last);
  } else {
    /* Single-prompt path. */
    if (!opt.prompt) opt.prompt = "bench";
    size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
    uint32_t *tokbuf = NULL;
    if (cap > 0) {
      tokbuf = (uint32_t*)malloc(sizeof(uint32_t) * cap);
      if (!tokbuf) { fprintf(stderr,"error: OOM\n"); ie_engine_destroy(engine); return 3; }
    }
    uint32_t got = 0;
    double t0 = ie_now_sec();
    st = ie_engine_generate(engine, opt.prompt, cap,
                            (tokbuf ? tokbuf : (uint32_t[]){0}), &got);
    double t1 = ie_now_sec();
    if (st != 0) fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)st);
    total_tokens = (uint64_t)got;
    total_wall   = (t1 - t0);
    if (tokbuf) free(tokbuf);
    (void)ie_engine_metrics(engine, &m_last);
  }

  /* Print one compact JSON (no token array to avoid huge outputs). */
  emit_json(total_tokens, total_wall, &m_last);

  ie_engine_destroy(engine);
  return 0;
}
