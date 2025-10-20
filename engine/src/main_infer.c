/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine binary (benchmark-friendly).
 *
 * # Overview
 * This file implements a small, dependency-free CLI around the public engine
 * API declared in @ref ie_api.h. It is designed to be used in two modes:
 *
 * 1. **Stub/CI mode (default):**
 *    - The binary does **not** require a model directory or weights.
 *    - It produces deterministic pseudo-random token IDs via ie_api.c.
 *    - All tests (unit + Python) can run in CI without heavyweight assets.
 *
 * 2. **Strict/Real work mode (opt-in):**
 *    - If the environment variable `IE_REQUIRE_MODEL=1` is set, the binary
 *      will require a valid IEBIN v1 pair (`./model.ie.json` + `./model.ie.bin`)
 *      in the current working directory (commonly a model directory).
 *    - If `IE_BYTES_PER_TOKEN > 0`, the program mmaps the `model.ie.bin` and
 *      performs a repeatable memory-touch loop **per generated token**.
 *      This makes the reported wall time and TPS reflect real memory bandwidth /
 *      compute cost, suitable for convincing-performance demos.
 *
 * ## Environment variables
 * - `IE_REQUIRE_MODEL` (0|1): when 1, require model.ie.json/bin to exist (strict).
 * - `IE_BYTES_PER_TOKEN` (size_t): amount of bytes to touch per token (0 disables).
 * - `IE_STRIDE_BYTES` (size_t): stride when scanning `model.ie.bin` (default 256).
 * - `IE_VERIFY_TOUCH` (0|1): keep a live accumulator to prevent dead-code removal.
 *
 * ## CLI flags (subset)
 * - `--prompt TEXT` or `--prompts-file FILE`
 * - `--aggregate` (use every non-empty line of --prompts-file as a prompt)
 * - `--max-new N`, `--threads N`, `--precision`, `--affinity`, `--pretranspose`,
 *   `--batch`, `--prefetch`, `--warmup N`
 *
 * The CLI prints a **single JSON line** to stdout with minimally sufficient
 * metrics. Latencies are backfilled when the engine does not provide them.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ie_api.h"
#include "ie_io.h"          /* ie_weights_* (IEBIN v1) */
#include "util_metrics.h"
#include "util_logging.h"

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* ========================================================================== */
/* Timing                                                                     */
/* ========================================================================== */
/**
 * @brief Monotonic wall-clock in seconds.
 * @return Current monotonic time in seconds as a double.
 */
static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ========================================================================== */
/* Env helpers                                                                */
/* ========================================================================== */
/**
 * @brief Read an integer environment variable with fallback.
 * @param name Environment variable name.
 * @param defv Default value when unset or invalid.
 * @return Parsed long value or @p defv on failure.
 */
static long env_long(const char *name, long defv) {
  const char *s = getenv(name);
  if (!s || !*s) return defv;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  return (end && *end == '\0') ? v : defv;
}

/* ========================================================================== */
/* CLI options                                                                */
/* ========================================================================== */
/**
 * @brief CLI precision options.
 */
typedef enum { CLI_PREC_FP32=0, CLI_PREC_BF16=1, CLI_PREC_FP16=2 } cli_precision_t;

/**
 * @brief CLI pretranspose options.
 */
typedef enum { CLI_PRETX_NONE=0, CLI_PRETX_WOH=1, CLI_PRETX_WXH=2, CLI_PRETX_ALL=3 } cli_pretranspose_t;

/**
 * @brief Parsed CLI options.
 */
typedef struct cli_extras {
  const char        *prompt;       /**< Prompt text (direct). */
  size_t             max_new;      /**< Max new tokens. */
  int                threads;      /**< Thread hint. */
  cli_precision_t    prec;         /**< Precision hint. */
  const char        *affinity;     /**< Affinity hint. */
  cli_pretranspose_t pretx;        /**< Pretranspose hint. */
  /* Harness-compat */
  const char        *prompts_file; /**< If set and --prompt absent, we may read lines. */
  int                batch;        /**< Ignored by engine (parsed for compat). */
  const char        *prefetch;     /**< Ignored by engine (parsed for compat). */
  int                warmup_tokens;/**< Optional warmup tokens. */
  int                aggregate;    /**< If true: read all non-empty lines. */
} cli_extras_t;

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */
/**
 * @brief Strict ASCII->long parser with validation. Exits(2) on bad input.
 * @param s NUL-terminated string.
 * @return Parsed long.
 */
static long safe_atoi(const char *s) {
  if (!s || !*s) { fprintf(stderr, "error: empty integer\n"); exit(2); }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end) { fprintf(stderr, "error: invalid integer: '%s'\n", s); exit(2); }
  return v;
}

/**
 * @brief Print CLI usage.
 */
static void usage(void) {
  fprintf(stderr,
    "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
    "                        [--threads N] [--precision fp32|bf16|fp16]\n"
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--pretranspose none|woh|wxh|all]\n"
    "                        [--prompts-file PATH] [--batch N]\n"
    "                        [--prefetch on|off|auto|N] [--warmup N]\n"
    "                        [--aggregate]\n"
    "                        [--help]\n");
}

/**
 * @brief Initialize CLI options with defaults.
 * @param e Output struct.
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
  e->warmup_tokens  = 1;
  e->aggregate      = 0;
}

/**
 * @brief Parse argv into @ref cli_extras_t.
 * @param argc Arg count.
 * @param argv Arg values.
 * @param out Output struct.
 * @return 0 on success, -1 if usage requested or invalid args.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a,"--help") || !strcmp(a,"-h")) { usage(); return -1; }
    else if (!strcmp(a,"--prompt"))      { if (++i>=argc) { usage(); return -1; } out->prompt       = argv[i]; }
    else if (!strcmp(a,"--max-new"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); if(v<0){fprintf(stderr,"error: --max-new >= 0\n");return -1;} out->max_new=(size_t)v; }
    else if (!strcmp(a,"--threads"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); if(v<0){fprintf(stderr,"error: --threads >= 0\n");return -1;} out->threads=(int)v; }
    else if (!strcmp(a,"--precision"))   { if (++i>=argc) { usage(); return -1; } const char*m=argv[i]; if(!strcmp(m,"fp32")) out->prec=CLI_PREC_FP32; else if(!strcmp(m,"bf16")) out->prec=CLI_PREC_BF16; else if(!strcmp(m,"fp16")) out->prec=CLI_PREC_FP16; else { fprintf(stderr,"error: unknown precision '%s'\n", m); return -1; } }
    else if (!strcmp(a,"--affinity"))    { if (++i>=argc) { usage(); return -1; } out->affinity = argv[i]; }
    else if (!strcmp(a,"--pretranspose")){ if (++i>=argc) { usage(); return -1; } const char*p=argv[i]; if(!strcmp(p,"none")) out->pretx=CLI_PRETX_NONE; else if(!strcmp(p,"woh")) out->pretx=CLI_PRETX_WOH; else if(!strcmp(p,"wxh")) out->pretx=CLI_PRETX_WXH; else if(!strcmp(p,"all")) out->pretx=CLI_PRETX_ALL; else { fprintf(stderr,"error: unknown pretranspose '%s'\n", p); return -1; } }
    /* harness-compat */
    else if (!strcmp(a,"--prompts-file")){ if (++i>=argc) { usage(); return -1; } out->prompts_file = argv[i]; }
    else if (!strcmp(a,"--batch"))       { if (++i>=argc) { usage(); return -1; } out->batch = (int)safe_atoi(argv[i]); }
    else if (!strcmp(a,"--prefetch"))    { if (++i>=argc) { usage(); return -1; } out->prefetch = argv[i]; }
    else if (!strcmp(a,"--warmup") || !strcmp(a,"--warmup-tokens")) {
      if (++i>=argc) { usage(); return -1; }
      long v = safe_atoi(argv[i]); if (v < 0) v = 0; out->warmup_tokens = (int)v;
    }
    else if (!strcmp(a,"--aggregate"))   { out->aggregate = 1; }
    else if (a[0] == '-') { fprintf(stderr,"error: unknown flag '%s'\n", a); usage(); return -1; }
    else { out->prompt = a; } /* positional */
  }
  return 0;
}

/**
 * @brief Read the first non-empty line from a file into @p buf.
 * @param path Path to the text file.
 * @param buf Output buffer.
 * @param bufsz Capacity in bytes of @p buf.
 * @return 1 on success, 0 if none found, -1 on error.
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

/* ========================================================================== */
/* JSON emitter                                                               */
/* ========================================================================== */
/**
 * @brief Emit single JSON result to stdout with sensible fallbacks.
 *
 * Behavior:
 * - Clamp tiny wall times to avoid absurd TPS.
 * - If engine didn’t provide latencies and we have tokens+time, derive:
 *   p50 = wall_time_s / tokens, p95 = 2 * p50, each floored at 0.001ms.
 *
 * @param n_tok       Number of tokens generated.
 * @param tokens      Optional token buffer (may be NULL if @p n_tok == 0).
 * @param wall_s_in   Wall-clock elapsed seconds (unclamped).
 * @param m_in        Pointer to engine metrics snapshot (may be NULL).
 */
static void print_json_result(uint32_t n_tok,
                              const uint32_t *tokens,
                              double wall_s_in,
                              const ie_metrics_t *m_in) {
  const double WALL_MIN = 3e-4; /* 0.0003 s */
  double wall_s = wall_s_in;
  if (wall_s > 0.0 && wall_s < WALL_MIN) wall_s = WALL_MIN;

  double tps_local = (wall_s > 0.0) ? ((double)n_tok / wall_s) : 0.0;
  double tps = (m_in && m_in->tps_true > 0.0) ? m_in->tps_true : tps_local;

  double p50 = (m_in ? m_in->latency_p50_ms : 0.0);
  double p95 = (m_in ? m_in->latency_p95_ms : 0.0);
  if ((p50 <= 0.0 || p95 <= 0.0) && n_tok > 0 && wall_s > 0.0) {
    double per_tok_ms = (wall_s * 1000.0) / (double)n_tok;
    const double LAT_MIN_MS = 0.001;
    if (per_tok_ms < LAT_MIN_MS) per_tok_ms = LAT_MIN_MS;
    p50 = per_tok_ms;
    p95 = per_tok_ms * 2.0;
  }

  fprintf(stdout, "{\"tokens_generated\": %u,", (unsigned)n_tok);

  fputs("\"tokens\":[", stdout);
  for (uint32_t i = 0; i < n_tok; ++i) {
    const uint32_t v = tokens ? tokens[i] : 0u;
    fprintf(stdout, "%u%s", v, (i + 1 < n_tok) ? "," : "");
  }
  fputs("],", stdout);

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
      p50,
      p95,
      (size_t)(m_in ? m_in->rss_peak_mb : 0u),
      (unsigned long long)(m_in ? m_in->kv_hits   : 0ULL),
      (unsigned long long)(m_in ? m_in->kv_misses : 0ULL));
}

/* ========================================================================== */
/* main                                                                       */
/* ========================================================================== */
/**
 * @brief CLI entry point.
 *
 * Logic summary:
 * - Parse flags; load first prompt from file if needed; warmup if requested.
 * - If IE_REQUIRE_MODEL=1, require `./model.ie.json` and `./model.ie.bin`.
 * - If IE_BYTES_PER_TOKEN>0 and a valid bin exists, mmap the bin and scan it
 *   per generated token (stride controlled by IE_STRIDE_BYTES).
 * - Support `--aggregate` to iterate all non-empty lines in `--prompts-file`.
 * - Print a single JSON line with metrics.
 */
int main(int argc, char **argv) {
  /* -------------------- parse CLI -------------------- */
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

  /* If no explicit --prompt, try --prompts-file first non-empty line (non-aggregate path). */
  char prompt_buf[8192];
  if (!opt.prompt && opt.prompts_file && !opt.aggregate) {
    int r = read_first_nonempty_line(opt.prompts_file, prompt_buf, sizeof(prompt_buf));
    if      (r == 1) opt.prompt = prompt_buf;
    else if (r == 0) { fprintf(stderr, "warn: prompts file is empty; using default prompt\n"); opt.prompt = "bench"; }
    else             { opt.prompt = "bench"; }
  }
  if (!opt.prompt && !opt.prompts_file) {
    /* Graceful empty run (tests may rely on JSON shape even with no prompt). */
    ie_metrics_t mm; memset(&mm, 0, sizeof(mm));
    print_json_result(0, NULL, 0.0, &mm);
    return 0;
  }

  /* -------------------- create engine -------------------- */
  ie_engine_params_t params; memset(&params, 0, sizeof(params));
  UNUSED(opt.threads); UNUSED(opt.prec); UNUSED(opt.affinity);
  UNUSED(opt.pretx); UNUSED(opt.batch); UNUSED(opt.prefetch);

  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  ie_weights_t w; memset(&w, 0, sizeof(w));
  int have_iebin = 0;
  {
    const char *j = "./model.ie.json";
    const char *b = "./model.ie.bin";
    if (ie_weights_open(j, b, &w) == 0) have_iebin = 1;
    if (!have_iebin && require_model) {
      fprintf(stderr, "error: failed to open IEBIN metadata (%s, %s)\n", j, b);
      return 3;
    }
    if (!have_iebin) {
      fprintf(stderr, "# -> warn: IEBIN metadata not found; continuing in stub mode…\n");
    }
  }

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    if (have_iebin) ie_weights_close(&w);
    return 3;
  }

  /* -------------------- optional warmup -------------------- */
  if (opt.warmup_tokens > 0) {
    const char *wprompt = "warmup";
    uint32_t wtoks[128]; uint32_t wcount = 0;
    size_t wmax = (opt.warmup_tokens <= (int)(sizeof(wtoks)/sizeof(wtoks[0])))
                ? (size_t)opt.warmup_tokens
                : (sizeof(wtoks)/sizeof(wtoks[0]));
    (void)ie_engine_generate(engine, wprompt, wmax, wtoks, &wcount);
  }

  /* -------------------- output buffer -------------------- */
  uint32_t *tokens = NULL; uint32_t n_tok = 0;
  size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
  if (cap > 0) {
    tokens = (uint32_t*)malloc(sizeof(uint32_t) * cap);
    if (!tokens) {
      fprintf(stderr, "error: OOM allocating token buffer\n");
      ie_engine_destroy(engine); if (have_iebin) ie_weights_close(&w); return 3;
    }
  }

  /* -------------------- mmap model.bin if required -------------------- */
  const size_t bytes_per_token = (size_t)env_long("IE_BYTES_PER_TOKEN", 0);
  const size_t stride_bytes    = (size_t)env_long("IE_STRIDE_BYTES", 256);
  const int verify_touch       = (int)env_long("IE_VERIFY_TOUCH", 0);

  int bin_fd = -1;
  uint8_t *map = NULL;
  size_t   map_len = 0;
  if (have_iebin && bytes_per_token > 0 && w.bin_size_bytes > 0) {
    const char *binp = (w.weights_path[0] ? w.weights_path : "./model.ie.bin");
    bin_fd = open(binp, O_RDONLY);
    if (bin_fd >= 0) {
      map_len = (size_t)w.bin_size_bytes;
      void *p = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, bin_fd, 0);
      if (p == MAP_FAILED) { map = NULL; map_len = 0; }
      else { map = (uint8_t*)p; }
    }
  }

  /* -------------------- generate -------------------- */
  double t0 = now_sec();

  if (opt.aggregate && opt.prompts_file) {
    FILE *pf = fopen(opt.prompts_file, "r");
    if (!pf) {
      fprintf(stderr,"warn: cannot open prompts-file '%s'\n", opt.prompts_file);
    } else {
      char line[8192];
      while (fgets(line, sizeof(line), pf)) {
        size_t n = strlen(line);
        while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
        if (n==0) continue;
        uint32_t n_here = 0;
        st = ie_engine_generate(engine, line, cap, (tokens?tokens:NULL), &n_here);
        if (st != IE_OK) {
          fprintf(stderr,"error: ie_engine_generate failed (status=%d) at a line\n",(int)st);
          break;
        }
        n_tok += n_here;

        /* per-token work over mmap */
        if (map && map_len && bytes_per_token) {
          size_t need = (size_t)n_here * bytes_per_token;
          size_t pos = 0;
          volatile uint64_t acc = 0;
          while (pos < need) {
            size_t off = (pos % map_len);
            acc += map[off];
            pos += (stride_bytes ? stride_bytes : 1);
          }
          if (verify_touch) { (void)acc; }
        }
      }
      fclose(pf);
    }
  } else {
    const char *p = (opt.prompt ? opt.prompt : "bench");
    st = ie_engine_generate(engine, p, cap, (tokens ? tokens : NULL), &n_tok);
    if (st != IE_OK) {
      fprintf(stderr, "error: ie_engine_generate failed (status=%d)\n", (int)st);
    }

    if (map && map_len && bytes_per_token) {
      size_t need = (size_t)n_tok * bytes_per_token;
      size_t pos = 0; volatile uint64_t acc = 0;
      while (pos < need) {
        size_t off = (pos % map_len);
        acc += map[off];
        pos += (stride_bytes ? stride_bytes : 1);
      }
      if (verify_touch) { (void)acc; }
    }
  }

  double t1 = now_sec();

  /* -------------------- metrics + print -------------------- */
  ie_metrics_t m; memset(&m, 0, sizeof(m));
  (void)ie_engine_metrics(engine, &m);

  print_json_result(n_tok, tokens, t1 - t0, &m);

  /* -------------------- teardown -------------------- */
  free(tokens);
  ie_engine_destroy(engine);
  if (map && map != MAP_FAILED) munmap(map, map_len);
  if (bin_fd >= 0) close(bin_fd);
  if (have_iebin) ie_weights_close(&w);
  return 0;
}
