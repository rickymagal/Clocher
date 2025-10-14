/**
 * @file weights.c
 * @brief IEBIN v1 lightweight loader (dependency-free, portable).
 *
 * Baseline goals:
 *  - Verify that model.ie.json exists and contains "version" and "dtype" keys
 *    (very relaxed JSON scanning; no 3rd-party parser).
 *  - Record model.ie.bin size if present (0 allowed for placeholder).
 *  - Fill ie_weights_t for engine initialization.
 *
 * NOTE: This baseline does not mmap or parse tensor maps yet (Step 3 will extend).
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "ie_io.h"

/* ---- tiny helpers ------------------------------------------------------- */

static int file_exists(const char *p) {
  struct stat st;
  return (p && stat(p, &st) == 0 && S_ISREG(st.st_mode));
}

static uint64_t file_size(const char *p) {
  struct stat st;
  if (!p || stat(p, &st) != 0) return 0;
  return (uint64_t)st.st_size;
}

static int read_all_text(const char *p, char **out_buf, size_t *out_len) {
  *out_buf = NULL; *out_len = 0;
  FILE *f = fopen(p, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
  long n = ftell(f);
  if (n < 0) { fclose(f); return -1; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
  char *buf = (char*)malloc((size_t)n + 1);
  if (!buf) { fclose(f); return -1; }
  size_t rd = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (rd != (size_t)n) { free(buf); return -1; }
  buf[n] = '\0';
  *out_buf = buf; *out_len = (size_t)n;
  return 0;
}

static int scan_json_key_int(const char *json, const char *key, int *out_val) {
  /* naive: find "<key>" then a colon and parse an integer */
  const char *k = strstr(json, key);
  if (!k) return -1;
  const char *c = strchr(k, ':');
  if (!c) return -1;
  int v = 0; int got = 0;
  while (*++c) {
    if (*c >= '0' && *c <= '9') { v = v*10 + (*c - '0'); got = 1; }
    else if (got) break;
  }
  if (!got) return -1;
  *out_val = v; return 0;
}

static int scan_json_key_string(const char *json, const char *key, char *out, size_t cap) {
  /* naive: find "<key>" then first quoted string after colon */
  const char *k = strstr(json, key);
  if (!k) return -1;
  const char *c = strchr(k, ':');
  if (!c) return -1;
  const char *q1 = strchr(c, '\"');
  if (!q1) return -1;
  const char *q2 = strchr(q1+1, '\"');
  if (!q2) return -1;
  size_t n = (size_t)(q2 - (q1+1));
  if (n+1 > cap) return -1;
  memcpy(out, q1+1, n); out[n] = '\0';
  return 0;
}

/* ---- public API --------------------------------------------------------- */

int ie_weights_open(const char *json_path,
                    const char *bin_path,
                    ie_weights_t *out) {
  if (!json_path || !out) return -1;
  if (!file_exists(json_path)) return -1;

  memset(out, 0, sizeof(*out));
  strncpy(out->json_path, json_path, sizeof(out->json_path)-1);

  if (bin_path && file_exists(bin_path)) {
    strncpy(out->weights_path, bin_path, sizeof(out->weights_path)-1);
    out->bin_size_bytes = file_size(bin_path);
  } else {
    /* allow empty bin in baseline */
    strncpy(out->weights_path, bin_path ? bin_path : "", sizeof(out->weights_path)-1);
    out->bin_size_bytes = 0;
  }

  char *buf = NULL; size_t n = 0;
  if (read_all_text(json_path, &buf, &n) != 0) {
    return -1;
  }

  /* defaults */
  out->version = 1;
  strncpy(out->dtype, "fp32", sizeof(out->dtype)-1);

  /* try to parse */
  (void)scan_json_key_int(buf, "\"version\"", &out->version);
  (void)scan_json_key_string(buf, "\"dtype\"", out->dtype, sizeof(out->dtype));

  free(buf);
  return 0;
}

void ie_weights_close(ie_weights_t *w) {
  (void)w; /* nothing to release in baseline */
}
