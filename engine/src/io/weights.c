/**
 * @file weights.c
 * @brief IEBIN v1 lightweight loader (dependency-free, portable).
 *
 * Baseline goals:
 *  - Verify that model.ie.json exists and contains "version" and "dtype" keys
 *    (relaxed JSON scanning; no 3rd-party parser).
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

/* ---------- helpers ---------- */
/**
 * @brief Test whether a path refers to an existing regular file.
 * @param p Path (UTF-8).
 * @return Non-zero if exists and is regular; 0 otherwise.
 */
static int file_exists(const char *p) {
  struct stat st;
  return (p && stat(p, &st) == 0 && S_ISREG(st.st_mode));
}

/**
 * @brief Return file size in bytes, or 0 on error.
 * @param p Path.
 * @return Size in bytes or 0.
 */
static uint64_t file_size(const char *p) {
  struct stat st;
  if (!p || stat(p, &st) != 0) return 0;
  return (uint64_t)st.st_size;
}

/**
 * @brief Read entire file into a NUL-terminated buffer.
 * @param p        Path to file.
 * @param out_buf  *out receives malloc'ed buffer (caller frees).
 * @param out_len  *out receives length (excluding NUL).
 * @return 0 on success, -1 on failure.
 */
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

/**
 * @brief Naively scan an integer value following a JSON key.
 * @param json    JSON text.
 * @param key     Key string (e.g., "\"version\"").
 * @param out_val Out integer.
 * @return 0 on success; -1 if key/value not found.
 */
static int scan_json_key_int(const char *json, const char *key, int *out_val) {
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

/**
 * @brief Naively scan a string value following a JSON key.
 * @param json JSON text.
 * @param key  Key string (e.g., "\"dtype\"").
 * @param out  Output buffer for value (no quotes), NUL-terminated.
 * @param cap  Capacity of @p out.
 * @return 0 on success; -1 on failure.
 */
static int scan_json_key_string(const char *json, const char *key, char *out, size_t cap) {
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

/* ---------- public API ---------- */
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
    strncpy(out->weights_path, bin_path ? bin_path : "", sizeof(out->weights_path)-1);
    out->bin_size_bytes = 0;
  }

  char *buf = NULL; size_t n = 0;
  if (read_all_text(json_path, &buf, &n) != 0) {
    return -1;
  }

  out->version = 1;
  strncpy(out->dtype, "fp32", sizeof(out->dtype)-1);

  (void)scan_json_key_int(buf, "\"version\"", &out->version);
  (void)scan_json_key_string(buf, "\"dtype\"", out->dtype, sizeof(out->dtype));

  free(buf);
  return 0;
}

void ie_weights_close(ie_weights_t *w) {
  (void)w; /* no resources to free in baseline */
}
