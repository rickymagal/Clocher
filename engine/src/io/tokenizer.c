/**
 * @file tokenizer.c
 * @brief Minimal tokenizer loader + whitespace tokenization fallback.
 *
 * Baseline behavior:
 *  - Reads vocab.json if present; extracts "vocab_size" (optional).
 *  - Encoding: split on ASCII whitespace; map each token to a stable pseudo-ID
 *    via FNV-1a hash folded to 16 bits and offset to [1000, 1000+65535].
 *  - Decoding: produce "T<ID>" sequences separated by spaces.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "ie_io.h"

/* ---------- helpers ---------- */
/**
 * @brief Check if a file exists (regular open test).
 * @param p Path to file.
 * @return Non-zero if file can be opened; 0 otherwise.
 */
static int file_exists(const char *p) {
  FILE *f = fopen(p, "rb");
  if (!f) return 0;
  fclose(f);
  return 1;
}

/**
 * @brief Read entire file into a NUL-terminated buffer.
 * @param p        Path to file.
 * @param out_buf  *out receives malloc'ed buffer (caller frees).
 * @param out_len  *out receives length (excluding NUL).
 * @return 0 on success; -1 on failure.
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
 * @param key     Key string (e.g., "\"vocab_size\"").
 * @param out_val Output integer.
 * @return 0 on success; -1 otherwise.
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
 * @brief FNV-1a 32-bit, folded to 16 bits, offset by 1000 to avoid special IDs.
 * @param s Pointer to token bytes.
 * @param n Length in bytes.
 * @return Deterministic token id in [1000, 1000+65535].
 */
static uint32_t hash_token(const char *s, size_t n) {
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (unsigned char)s[i];
    h *= 16777619u;
  }
  uint32_t folded = (h >> 16) ^ (h & 0xFFFFu);
  return 1000u + (folded & 0xFFFFu);
}

/* ---------- public API ---------- */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out) {
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  if (vocab_path) {
    strncpy(out->vocab_path, vocab_path, sizeof(out->vocab_path)-1);
  }
  out->vocab_size = 0;

  if (vocab_path && file_exists(vocab_path)) {
    char *buf = NULL; size_t n = 0;
    if (read_all_text(vocab_path, &buf, &n) == 0) {
      (void)scan_json_key_int(buf, "\"vocab_size\"", &out->vocab_size);
      free(buf);
    }
  }
  if (out->vocab_size <= 0) out->vocab_size = 256; /* safe fallback */
  return 0;
}

void ie_vocab_free(ie_vocab_t *v) {
  (void)v; /* nothing to release in baseline */
}

int ie_tok_encode(const ie_vocab_t *v,
                  const char *utf8,
                  uint32_t *out_ids,
                  uint32_t *out_count) {
  (void)v;
  if (!utf8 || !out_count) return -1;

  /* first pass: count tokens */
  unsigned count = 0;
  const char *p = utf8;
  while (*p) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (!*p) break;
    const char *start = p;
    while (*p && !isspace((unsigned char)*p)) p++;
    (void)start;
    count++;
  }

  if (!out_ids) { *out_count = count; return 0; }

  /* second pass: emit ids */
  p = utf8;
  unsigned idx = 0;
  while (*p && idx < count) {
    while (*p && isspace((unsigned char)*p)) p++;
    if (!*p) break;
    const char *start = p;
    while (*p && !isspace((unsigned char)*p)) p++;
    const size_t len = (size_t)(p - start);
    out_ids[idx++] = hash_token(start, len);
  }

  *out_count = idx;
  return 0;
}

int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out_utf8,
                  size_t out_capacity) {
  (void)v;
  if (!ids || !out_utf8 || out_capacity == 0) return -1;
  size_t used = 0;
  for (uint32_t i = 0; i < count; ++i) {
    char buf[32];
    int n = snprintf(buf, sizeof(buf), "T%u", ids[i]);
    if (n <= 0) return -1;
    if (used + (size_t)n + (i ? 1u : 0u) + 1u > out_capacity) return -1;
    if (i) out_utf8[used++] = ' ';
    memcpy(out_utf8 + used, buf, (size_t)n);
    used += (size_t)n;
  }
  out_utf8[used] = '\0';
  return 0;
}
