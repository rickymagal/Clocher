/* engine/src/main_chat.c */
/**
 * @file main_chat.c
 * @brief Interactive CLI for tokenizer validation and (later) text generation.
 *
 * This binary is meant to be "human-facing":
 *  - encode: text -> ids
 *  - decode: ids -> text
 *  - roundtrip: encode + decode
 *
 * Once you wire real generation into ie_api.c / ie_engine_generate(), you can
 * add a "prompt -> generate -> decode" mode here without touching the benchmark
 * harness behavior in main_infer.c.
 *
 * Build integration:
 *  - Add this file to your Makefile as a separate target (e.g. `chat`).
 *  - Link with the same engine objects you already build, plus tokenizer_gptoss.c.
 */

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util_logging.h"
#include "ie_tokenizer_gptoss.h"

static void usage(const char *argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s --tokenizer <tokenizer.json> --encode  \"text here\"\n"
          "  %s --tokenizer <tokenizer.json> --decode  \"1,2,3,4\"\n"
          "  %s --tokenizer <tokenizer.json> --roundtrip \"text here\"\n",
          argv0, argv0, argv0);
}

static int parse_ids_csv(const char *s, uint32_t **out_ids, uint32_t *out_n) {
  if (!s || !out_ids || !out_n) return -1;
  *out_ids = NULL;
  *out_n = 0;

  size_t cap = 64;
  size_t n = 0;
  uint32_t *ids = (uint32_t *)calloc(cap, sizeof(uint32_t));
  if (!ids) return -1;

  const char *p = s;
  while (*p) {
    while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;
    if (!*p) break;

    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) { free(ids); return -1; }
    if (v > 0xFFFFFFFFul) { free(ids); return -1; }

    if (n == cap) {
      cap *= 2;
      uint32_t *ni = (uint32_t *)realloc(ids, cap * sizeof(uint32_t));
      if (!ni) { free(ids); return -1; }
      ids = ni;
    }
    ids[n++] = (uint32_t)v;
    p = end;
  }

  *out_ids = ids;
  *out_n = (uint32_t)n;
  return 0;
}

static int cmd_decode(ie_tok_gptoss_t *tok, const char *csv) {
  uint32_t *ids = NULL;
  uint32_t n = 0;
  if (parse_ids_csv(csv, &ids, &n) != 0) {
    ie_log_error("Invalid id list.");
    return 1;
  }

  size_t need = 0;
  int rc = ie_tok_gptoss_decode(tok, ids, n, NULL, &need);
  if (rc != IE_TOK_GPTOSS_OK || need == 0) {
    free(ids);
    ie_log_error("Decode size query failed (%d).", rc);
    return 1;
  }

  char *out = (char *)malloc(need);
  if (!out) {
    free(ids);
    ie_log_error("Out of memory.");
    return 1;
  }

  size_t cap = need;
  rc = ie_tok_gptoss_decode(tok, ids, n, out, &cap);
  free(ids);
  if (rc != IE_TOK_GPTOSS_OK) {
    free(out);
    ie_log_error("Decode failed (%d).", rc);
    return 1;
  }

  printf("%s\n", out);
  free(out);
  return 0;
}

static int cmd_encode(ie_tok_gptoss_t *tok, const char *text) {
  uint32_t need = 0;
  int rc = ie_tok_gptoss_encode(tok, text, NULL, &need);
  if (rc != IE_TOK_GPTOSS_OK) {
    ie_log_error("Encode size query failed (%d).", rc);
    return 1;
  }

  uint32_t *ids = (uint32_t *)calloc((size_t)need ? (size_t)need : 1u, sizeof(uint32_t));
  if (!ids) {
    ie_log_error("Out of memory.");
    return 1;
  }

  uint32_t cap = need;
  rc = ie_tok_gptoss_encode(tok, text, ids, &cap);
  if (rc != IE_TOK_GPTOSS_OK) {
    free(ids);
    ie_log_error("Encode failed (%d).", rc);
    return 1;
  }

  for (uint32_t i = 0; i < cap; ++i) {
    if (i) putchar(',');
    printf("%u", ids[i]);
  }
  putchar('\n');
  free(ids);
  return 0;
}

static int cmd_roundtrip(ie_tok_gptoss_t *tok, const char *text) {
  uint32_t need = 0;
  int rc = ie_tok_gptoss_encode(tok, text, NULL, &need);
  if (rc != IE_TOK_GPTOSS_OK) {
    ie_log_error("Encode size query failed (%d).", rc);
    return 1;
  }

  uint32_t *ids = (uint32_t *)calloc((size_t)need ? (size_t)need : 1u, sizeof(uint32_t));
  if (!ids) return 1;

  uint32_t cap = need;
  rc = ie_tok_gptoss_encode(tok, text, ids, &cap);
  if (rc != IE_TOK_GPTOSS_OK) {
    free(ids);
    ie_log_error("Encode failed (%d).", rc);
    return 1;
  }

  size_t out_need = 0;
  rc = ie_tok_gptoss_decode(tok, ids, cap, NULL, &out_need);
  if (rc != IE_TOK_GPTOSS_OK || out_need == 0) {
    free(ids);
    ie_log_error("Decode size query failed (%d).", rc);
    return 1;
  }

  char *out = (char *)malloc(out_need);
  if (!out) { free(ids); return 1; }

  size_t out_cap = out_need;
  rc = ie_tok_gptoss_decode(tok, ids, cap, out, &out_cap);
  free(ids);
  if (rc != IE_TOK_GPTOSS_OK) {
    free(out);
    ie_log_error("Decode failed (%d).", rc);
    return 1;
  }

  printf("%s\n", out);
  free(out);
  return 0;
}

int main(int argc, char **argv) {
  const char *tok_path = NULL;
  const char *mode = NULL;
  const char *arg = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tok_path = argv[++i];
    } else if (strcmp(argv[i], "--encode") == 0 && i + 1 < argc) {
      mode = "encode";
      arg = argv[++i];
    } else if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
      mode = "decode";
      arg = argv[++i];
    } else if (strcmp(argv[i], "--roundtrip") == 0 && i + 1 < argc) {
      mode = "roundtrip";
      arg = argv[++i];
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  if (!tok_path || !mode || !arg) {
    usage(argv[0]);
    return 2;
  }

  ie_tok_gptoss_t *tok = NULL;
  int rc = ie_tok_gptoss_open(tok_path, &tok);
  if (rc != IE_TOK_GPTOSS_OK) {
    ie_log_error("Failed to open tokenizer (%d): %s", rc, tok_path);
    return 1;
  }

  int ret = 0;
  if (strcmp(mode, "decode") == 0) ret = cmd_decode(tok, arg);
  else if (strcmp(mode, "encode") == 0) ret = cmd_encode(tok, arg);
  else if (strcmp(mode, "roundtrip") == 0) ret = cmd_roundtrip(tok, arg);
  else { usage(argv[0]); ret = 2; }

  ie_tok_gptoss_close(tok);
  return ret;
}
