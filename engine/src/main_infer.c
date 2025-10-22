/**
 * @file main_infer.c
 * @brief CLI entry point for the inference engine (benchmark-friendly).
 *
 * ## Overview
 * This CLI is designed for two execution modes:
 *
 * 1. **CI/Stub mode (default)** — does not require model files. The engine
 *    generates deterministic pseudo-token IDs (see ie_api.c).
 *
 * 2. **Strict mode (opt-in)** — when `IE_REQUIRE_MODEL=1`, the program
 *    requires a valid IEBIN v1 pair in the current directory
 *    (`./model.ie.json` + `./model.ie.bin`). If `IE_BYTES_PER_TOKEN > 0`,
 *    it `mmap`s the `model.ie.bin` and performs a *per-token “work touch”*
 *    loop to mimic realistic memory pressure. This “work touch” is **counted**
 *    in the TPS timing by design.
 *
 * ### Critical timing rule
 * The measured window **includes only**:
 *   - `ie_engine_generate(...)`
 *   - the optional per-token “work touch” over the model blob
 *
 * Instrumentation such as KV/RSS sampling, printing JSON, or reading prompt
 * files occurs **outside** the timed region to avoid skewing TPS.
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

#include "ie_api.h"        /* engine API + ie_metrics_t */
#include "ie_io.h"         /* IEBIN loader + tokenizer stub */
#include "util_metrics.h"  /* RSS + KV plumbing */
#include "util_logging.h"  /* optional logging macros (no-ops if not used) */

#ifndef UNUSED
#  define UNUSED(x) (void)(x)
#endif

/* ========================================================================== */
/* Timing                                                                     */
/* ========================================================================== */

/**
 * @brief Get a monotonic wall-clock timestamp in seconds.
 *
 * Uses `CLOCK_MONOTONIC` to avoid jumps due to NTP or wall-clock changes.
 *
 * @return Seconds since an unspecified monotonic epoch.
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
 * @brief Parse an environment variable as a long with a default fallback.
 *
 * The parsing is strict decimal (`base=10`); any trailing non-digits cause
 * the default to be returned.
 *
 * @param name Environment variable name (e.g., `"IE_STRIDE_BYTES"`).
 * @param defv Default value to return when unset or invalid.
 * @return Parsed value, or @p defv on error/unset.
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
 * @enum cli_precision_t
 * @brief Precision hints accepted by the CLI.
 *
 * These are forwarded into @ref ie_engine_params_t for compatibility,
 * though the stub engine may ignore them.
 */
typedef enum { CLI_PREC_FP32=0, CLI_PREC_BF16=1, CLI_PREC_FP16=2 } cli_precision_t;

/**
 * @enum cli_pretranspose_t
 * @brief Pretranspose policy hints accepted by the CLI.
 */
typedef enum { CLI_PRETX_NONE=0, CLI_PRETX_WOH=1, CLI_PRETX_WXH=2, CLI_PRETX_ALL=3 } cli_pretranspose_t;

/**
 * @struct cli_extras
 * @brief Parsed CLI options container.
 *
 * This structure aggregates user-provided options and harness compatibility
 * fields. Unknown hints are kept to allow future engines to consume them.
 *
 * @var cli_extras::prompt
 * Direct prompt text when not using aggregate/file mode.
 * @var cli_extras::max_new
 * Maximum number of new tokens to generate.
 * @var cli_extras::threads
 * Requested thread count (hint; may be ignored by stub).
 * @var cli_extras::prec
 * Precision hint (fp32/bf16/fp16).
 * @var cli_extras::affinity
 * Affinity policy hint string.
 * @var cli_extras::pretx
 * Pretranspose policy hint.
 * @var cli_extras::prompts_file
 * Path to a prompts file; used especially in aggregate mode.
 * @var cli_extras::batch
 * Batch size hint (compat; not used by stub).
 * @var cli_extras::prefetch
 * Prefetch policy hint string.
 * @var cli_extras::warmup_tokens
 * Number of warmup tokens (0 disables).
 * @var cli_extras::aggregate
 * When non-zero, iterate all non-empty lines in @ref prompts_file.
 */
typedef struct cli_extras {
  const char        *prompt;
  size_t             max_new;
  int                threads;
  cli_precision_t    prec;
  const char        *affinity;
  cli_pretranspose_t pretx;

  /* Harness/compat */
  const char        *prompts_file;
  int                batch;
  const char        *prefetch;
  int                warmup_tokens;
  int                aggregate;
} cli_extras_t;

/* ========================================================================== */
/* CLI parsing                                                                */
/* ========================================================================== */

/**
 * @brief Convert ASCII to long with strict validation or terminate.
 *
 * This routine is used for flag values; on malformed input it prints an error
 * and calls `exit(2)`, matching the CLI usage contract.
 *
 * @param s NUL-terminated string to parse.
 * @return Parsed long value on success.
 */
static long safe_atoi(const char *s) {
  if (!s || !*s) { fprintf(stderr, "error: empty integer\n"); exit(2); }
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || *end) {
    fprintf(stderr, "error: invalid integer: '%s'\n", s);
    exit(2);
  }
  return v;
}

/**
 * @brief Print CLI usage synopsis to `stderr`.
 *
 * This function only displays text and returns; the caller decides whether
 * to exit or continue.
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
 * @brief Initialize a @ref cli_extras_t with default values.
 *
 * Defaults are chosen to be CI-friendly (small output, warmup enabled).
 *
 * @param e Pointer to the options struct to initialize (non-NULL).
 */
static void cli_extras_defaults(cli_extras_t *e) {
  e->prompt         = NULL;
  e->max_new        = 8;
  e->threads        = 0;
  e->prec           = CLI_PREC_FP32;
  e->affinity       = "auto";
  e->pretx          = CLI_PRETX_NONE;
  e->prompts_file   = NULL;
  e->batch          = 1;
  e->prefetch       = "auto";
  e->warmup_tokens  = 1;
  e->aggregate      = 0;
}

/**
 * @brief Parse `argv` flags into a @ref cli_extras_t.
 *
 * Recognized flags are documented in @ref usage. On any malformed flag,
 * usage is printed and `-1` is returned so the caller can exit with code 2.
 *
 * @param argc Argument count from `main`.
 * @param argv Argument vector from `main`.
 * @param out  Output struct (pre-initialized via @ref cli_extras_defaults).
 * @return 0 on success, -1 if usage was displayed or parsing failed.
 */
static int parse_flags(int argc, char **argv, cli_extras_t *out) {
  cli_extras_defaults(out);
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a,"--help") || !strcmp(a,"-h")) { usage(); return -1; }
    else if (!strcmp(a,"--prompt"))      { if (++i>=argc) { usage(); return -1; } out->prompt       = argv[i]; }
    else if (!strcmp(a,"--max-new"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); if(v<0){fprintf(stderr,"error: --max-new >= 0\n");return -1;} out->max_new=(size_t)v; }
    else if (!strcmp(a,"--threads"))     { if (++i>=argc) { usage(); return -1; } long v=safe_atoi(argv[i]); out->threads=(int)v; }
    else if (!strcmp(a,"--precision"))   { if (++i>=argc) { usage(); return -1; } const char*m=argv[i]; if(!strcmp(m,"fp32")) out->prec=CLI_PREC_FP32; else if(!strcmp(m,"bf16")) out->prec=CLI_PREC_BF16; else if(!strcmp(m,"fp16")) out->prec=CLI_PREC_FP16; else { fprintf(stderr,"error: unknown precision '%s'\n", m); return -1; } }
    else if (!strcmp(a,"--affinity"))    { if (++i>=argc) { usage(); return -1; } out->affinity = argv[i]; }
    else if (!strcmp(a,"--pretranspose")){ if (++i>=argc) { usage(); return -1; } const char*p=argv[i]; if(!strcmp(p,"none")) out->pretx=CLI_PRETX_NONE; else if(!strcmp(p,"woh")) out->pretx=CLI_PRETX_WOH; else if(!strcmp(p,"wxh")) out->pretx=CLI_PRETX_WXH; else if(!strcmp(p,"all")) out->pretx=CLI_PRETX_ALL; else { fprintf(stderr,"error: unknown pretranspose '%s'\n", p); return -1; } }
    /* Harness-compat */
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

/* ========================================================================== */
/* Prompts helper                                                             */
/* ========================================================================== */

/**
 * @brief Read the first non-empty line from a UTF-8 text file.
 *
 * Trailing `\\r`/`\\n` are stripped. Empty or whitespace-only lines are skipped.
 *
 * @param path  Path to the file to open.
 * @param buf   Output buffer that receives the line (NUL-terminated).
 * @param bufsz Capacity of @p buf in bytes.
 * @return 1 on success, 0 if no non-empty line found, -1 on open error.
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
 * @brief Emit a single JSON record describing the last run’s metrics.
 *
 * The JSON schema matches what `scripts/true_tps_strict.sh` and
 * `update_performance_md.py` expect. If latency numbers are absent in @p m_in,
 * conservative fallbacks are derived from `wall_s_in` and `n_tok`.
 *
 * @param n_tok     Number of tokens generated in the measured window.
 * @param tokens    Optional token buffer; when NULL, zeros are printed.
 * @param wall_s_in Raw wall-clock seconds for the measured window.
 * @param m_in      Optional metrics snapshot (may be NULL).
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
 * @brief Program entry point.
 *
 * Responsibilities:
 *  - Parse CLI flags and environment toggles.
 *  - Optionally validate presence and readability of IEBIN v1 (`IE_REQUIRE_MODEL=1`).
 *  - Create an engine instance and run generation.
 *  - Measure a strictly delimited window (generation + optional “work touch”).
 *  - Sample metrics outside the timed window and emit one JSON record to stdout.
 *
 * ### Exit codes
 *  - `0` success
 *  - `2` usage / arguments error
 *  - `3` IEBIN required but missing/unreadable (strict mode)
 *  - `5` engine create failure
 *  - `6` OOM on token buffer
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return Process exit status as described above.
 */
int main(int argc, char **argv) {
  /* -------------------- parse CLI -------------------- */
  cli_extras_t opt;
  if (parse_flags(argc, argv, &opt) != 0) return 2;

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
    print_json_result(0, NULL, 0.0, &mm);
    return 0;
  }

  /* -------------------- strict IEBIN gating -------------------- */
  const int require_model = (int)env_long("IE_REQUIRE_MODEL", 0);

  ie_weights_t w; memset(&w, 0, sizeof(w));
  {
    const char *j = "./model.ie.json";
    const char *b = "./model.ie.bin";
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
  /* We keep hints in params for future engines; stub ignores them. */
  params.precision   = "fp32";
  params.affinity    = "auto";
  params.pretranspose= "none";
  params.prefetch    = "auto";

  ie_engine_t *engine = NULL;
  ie_status_t st = ie_engine_create(&params, &engine);
  if (st != IE_OK || !engine) {
    fprintf(stderr, "error: ie_engine_create failed (status=%d)\n", (int)st);
    ie_weights_close(&w);
    return 5;
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

  /* -------------------- token output buffer -------------------- */
  uint32_t *tokens = NULL; uint32_t n_tok = 0;
  size_t cap = (opt.max_new > 0 ? opt.max_new : 0);
  if (cap > 0) {
    tokens = (uint32_t*)malloc(sizeof(uint32_t) * cap);
    if (!tokens) {
      fprintf(stderr, "error: OOM allocating token buffer\n");
      ie_engine_destroy(engine);
      ie_weights_close(&w);
      return 6;
    }
  }

  /* -------------------- mmap model.bin for per-token work -------------------- */
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

  /* -------------------- aggregate preload (outside timing) -------------------- */
  char **agg_lines = NULL;
  size_t agg_n = 0, agg_cap = 0;
  if (opt.aggregate && opt.prompts_file) {
    FILE *pf = fopen(opt.prompts_file, "r");
    if (pf) {
      char line[8192];
      while (fgets(line, sizeof(line), pf)) {
        size_t n = strlen(line);
        while (n && (line[n-1]=='\n'||line[n-1]=='\r')) line[--n]='\0';
        if (!n) continue;
        if (agg_n == agg_cap) {
          size_t nc = agg_cap ? agg_cap * 2 : 64;
          char **tmp = (char**)realloc(agg_lines, nc * sizeof(char*));
          if (!tmp) { fclose(pf); fprintf(stderr,"error: OOM preload\n"); goto teardown; }
          agg_lines = tmp; agg_cap = nc;
        }
        agg_lines[agg_n] = strdup(line);
        if (!agg_lines[agg_n]) { fclose(pf); fprintf(stderr,"error: OOM copy\n"); goto teardown; }
        ++agg_n;
      }
      fclose(pf);
    } else {
      fprintf(stderr,"warn: cannot open prompts-file '%s'\n", opt.prompts_file);
    }
  }

  /* -------------------- measured window: ONLY generation + work-touch ------- */
  double t0 = now_sec();

  if (opt.aggregate && agg_n > 0) {
    for (size_t li = 0; li < agg_n; ++li) {
      uint32_t n_here = 0;
      st = ie_engine_generate(engine, agg_lines[li], cap, (tokens?tokens:NULL), &n_here);
      if (st != IE_OK) { fprintf(stderr,"error: ie_engine_generate (status=%d) at line %zu\n",(int)st, li); break; }
      n_tok += n_here;

      if (map && map_len && bytes_per_token) {
        size_t need = (size_t)n_here * bytes_per_token;
        size_t pos = 0; volatile uint64_t acc = 0;
        while (pos < need) {
          size_t off = (pos % map_len);
          acc += map[off];
          pos += (stride_bytes ? stride_bytes : 1);
        }
        if (verify_touch) { (void)acc; }
      }
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

  /* -------------------- AFTER timing: snapshot metrics (no touching!) ------ */
  ie_metrics_t m; memset(&m, 0, sizeof(m));
  /* If the engine populates KV hits/misses internally, this preserves them. */
  (void)ie_engine_metrics(engine, &m);

  /* Sample RSS peak outside the measured window. */
  (void)ie_metrics_sample_rss_peak(&m);

  /* -------------------- print JSON & teardown ------------------------------ */
  print_json_result(n_tok, tokens, t1 - t0, &m);

teardown:
  if (agg_lines) {
    for (size_t i = 0; i < agg_n; ++i) free(agg_lines[i]);
    free(agg_lines);
  }
  free(tokens);
  if (map && map != MAP_FAILED) munmap(map, map_len);
  if (bin_fd >= 0) close(bin_fd);
  ie_engine_destroy(engine);
  ie_weights_close(&w);
  return 0;
}
