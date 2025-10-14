/**
 * @file main_infer.c
 * @brief CLI for the inference engine with JSON metrics output.
 *
 * Features:
 *  - Flags: --prompt, --max-new, --threads, --precision, --affinity, --grainsize, --numa, --help
 *  - Also accepts a single positional PROMPT if --prompt is not provided.
 *  - Rejects unknown flags with non-zero exit and usage to stderr.
 *  - Honors --max-new 0 (prints tokens_generated: 0).
 *  - Emits tokens array in the JSON (used by determinism tests).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "ie_api.h"

/**
 * @brief Return current time in seconds using C11 timespec_get.
 *
 * @return Wall-clock seconds as double.
 */
static double now_s(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief Print CLI usage to a stream.
 *
 * @param f Output stream (stdout or stderr).
 */
static void print_usage(FILE *f) {
  fprintf(f,
    "Usage: inference-engine [--prompt TEXT] [--max-new N]\n"
    "                        [--threads N] [--precision fp32|bf16|fp16]\n"
    "                        [--affinity auto|compact|scatter]\n"
    "                        [--grainsize K] [--numa compact|interleave|node:X]\n"
    "                        [--help]\n"
    "       inference-engine TEXT   # positional prompt\n"
    "\n"
    "Examples:\n"
    "  inference-engine --prompt \"hello\" --max-new 8\n"
    "  inference-engine \"test prompt\" --max-new 16 --threads 2 --precision bf16\n"
    "  IE_TP_USE_AFFINITY=1 inference-engine --prompt hi --threads 8 --affinity compact\n"
  );
}

/**
 * @brief Parse CLI flags into engine parameters and prompt/max_new.
 *
 * @param argc        Argument count.
 * @param argv        Argument vector.
 * @param p           Output engine parameters (filled with defaults first).
 * @param prompt_out  Receives prompt pointer (owned by argv; do not free).
 * @param max_new_out Receives max-new value (default 16).
 * @return 0 on success; non-zero on invalid usage.
 */
static int parse_flags(int argc, char **argv,
                       ie_engine_params_t *p,
                       const char **prompt_out,
                       uint32_t *max_new_out) {
  const char *prompt = NULL;
  uint32_t max_new = 16;

  memset(p, 0, sizeof(*p));
  p->threads   = 0;       /* engine will default to 1 for stability */
  p->precision = "fp32";
  p->affinity  = "auto";
  p->grainsize = 0;
  p->numa_mode = NULL;

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
      print_usage(stdout);
      exit(0);
    } else if (strcmp(arg, "--prompt") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --prompt requires a value\n"); return -1; }
      prompt = argv[++i];
    } else if (strcmp(arg, "--max-new") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --max-new requires a value\n"); return -1; }
      long v = strtol(argv[++i], NULL, 10);
      if (v < 0) { fprintf(stderr, "error: --max-new must be >= 0\n"); return -1; }
      max_new = (uint32_t)v;
    } else if (strcmp(arg, "--threads") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --threads requires a value\n"); return -1; }
      long v = strtol(argv[++i], NULL, 10);
      if (v < 0) { fprintf(stderr, "error: --threads must be >= 0\n"); return -1; }
      p->threads = (uint32_t)v;
    } else if (strcmp(arg, "--precision") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --precision requires a value\n"); return -1; }
      const char *val = argv[++i];
      if (strcmp(val, "fp32") && strcmp(val, "bf16") && strcmp(val, "fp16")) {
        fprintf(stderr, "error: --precision must be fp32|bf16|fp16\n");
        return -1;
      }
      p->precision = val;
    } else if (strcmp(arg, "--affinity") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --affinity requires a value\n"); return -1; }
      const char *val = argv[++i];
      if (strcmp(val, "auto") && strcmp(val, "compact") && strcmp(val, "scatter")) {
        fprintf(stderr, "error: --affinity must be auto|compact|scatter\n");
        return -1;
      }
      p->affinity = val;
    } else if (strcmp(arg, "--grainsize") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --grainsize requires a value\n"); return -1; }
      long v = strtol(argv[++i], NULL, 10);
      if (v < 0) { fprintf(stderr, "error: --grainsize must be >= 0\n"); return -1; }
      p->grainsize = (uint32_t)v;
    } else if (strcmp(arg, "--numa") == 0) {
      if (i + 1 >= argc) { fprintf(stderr, "error: --numa requires a value\n"); return -1; }
      p->numa_mode = argv[++i]; /* Documentation only in this baseline. */
    } else if (arg[0] == '-') {
      fprintf(stderr, "error: unknown flag: %s\n\n", arg);
      print_usage(stderr);
      return -1;
    } else {
      if (prompt == NULL) prompt = arg; /* positional prompt */
      else { fprintf(stderr, "error: unexpected positional arg: %s\n\n", arg); print_usage(stderr); return -1; }
    }
  }

  if (prompt == NULL) prompt = "";

  *prompt_out  = prompt;
  *max_new_out = max_new;
  return 0;
}

/**
 * @brief Main entry point: parse flags, run generation, print JSON metrics.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success; non-zero error code on failure.
 */
int main(int argc, char **argv) {
  ie_engine_params_t params;
  const char *prompt = NULL;
  uint32_t max_new = 16;

  if (parse_flags(argc, argv, &params, &prompt, &max_new) != 0) {
    return 2; /* invalid CLI */
  }

  ie_engine_t *eng = NULL;
  ie_status_t st = ie_engine_create(&params, &eng);
  if (st != IE_OK) {
    fprintf(stderr, "error: engine_create failed: %d\n", (int)st);
    return 3;
  }

  uint32_t *tokens = NULL;
  if (max_new > 0) {
    tokens = (uint32_t*)malloc(sizeof(uint32_t) * max_new);
    if (!tokens) {
      fprintf(stderr, "error: OOM allocating tokens\n");
      ie_engine_destroy(eng);
      return 4;
    }
  }

  const double t0 = now_s();
  uint32_t out_count = 0;
  st = ie_engine_generate(eng, prompt, max_new, tokens, &out_count);
  if (st != IE_OK) {
    fprintf(stderr, "error: generate failed: %d\n", (int)st);
    free(tokens);
    ie_engine_destroy(eng);
    return 5;
  }
  const double t1 = now_s();

  ie_metrics_t m;
  st = ie_engine_metrics(eng, &m);
  if (st != IE_OK) {
    fprintf(stderr, "error: metrics failed: %d\n", (int)st);
    free(tokens);
    ie_engine_destroy(eng);
    return 6;
  }

  /* JSON with spaces after ':' to match tests. */
  fputs("{", stdout);
  fprintf(stdout, "\"tokens_generated\": %u", (unsigned)out_count);

  /* tokens array */
  fputs(",\"tokens\": [", stdout);
  for (uint32_t i = 0; i < out_count; ++i) {
    fprintf(stdout, "%s%u", (i ? "," : ""), tokens[i]);
  }
  fputs("]", stdout);

  /* metrics */
  fprintf(stdout, ",\"wall_time_s\": %.6f", (t1 - t0));
  fprintf(stdout, ",\"tps_true\": %.6f", m.tps_true);
  fprintf(stdout, ",\"latency_p50_ms\": %.3f", m.latency_p50_ms);
  fprintf(stdout, ",\"latency_p95_ms\": %.3f", m.latency_p95_ms);
  fprintf(stdout, ",\"rss_peak_mb\": %zu", (size_t)m.rss_peak_mb);
  fprintf(stdout, ",\"kv_hits\": %llu", (unsigned long long)m.kv_hits);
  fprintf(stdout, ",\"kv_misses\": %llu", (unsigned long long)m.kv_misses);
  fputs("}\n", stdout);

  free(tokens);
  ie_engine_destroy(eng);
  return 0;
}
