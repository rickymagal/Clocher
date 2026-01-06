/* ============================================================================
 * File: engine/src/io/tokenizer.c
 * ============================================================================
 */
/**
 * @file tokenizer.c
 * @brief Minimal, dependency-free tokenizer used by tests and the CLI harness.
 *
 * @details
 * This module provides the small tokenizer/vocabulary API declared in @ref ie_io.h:
 *  - @ref ie_vocab_load  : Load a vocabulary descriptor (best-effort).
 *  - @ref ie_vocab_free  : Release vocabulary resources (no-op here).
 *  - @ref ie_tok_encode  : Encode a UTF-8 string into token IDs.
 *  - @ref ie_tok_decode  : Convert token IDs back into a deterministic textual form.
 *
 * Important:
 * This is intentionally NOT a real BPE/WordPiece tokenizer. It exists to keep
 * unit tests deterministic and to allow a lightweight harness without external
 * tokenizer artifacts. Production inference must use the model's real tokenizer.
 *
 * Tokenization model:
 *  - Split on ASCII whitespace (isspace()).
 *  - Each token is mapped to a stable 31-bit positive integer using FNV-1a.
 *
 * Size-query behavior:
 *  - Encoding supports a size-only pass by calling @ref ie_tok_encode with
 *    @p ids == NULL. The required number of tokens is returned via @p out_count.
 *
 * Logging:
 *  - INFO logs for vocabulary load decisions.
 *  - ERROR logs for invalid args and capacity issues.
 *  - Optional per-call verbose logs controlled by IE_TOKENIZER_VERBOSE=1.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <ctype.h>      /* isspace */
#include <errno.h>
#include <inttypes.h>   /* PRIu32 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_io.h"
#include "util_logging.h"

/* ========================================================================== */
/* Internal helpers (file-local)                                              */
/* ========================================================================== */

/**
 * @brief Read a boolean environment variable as 0/1.
 *
 * @details
 * Accepted spellings (case-sensitive variants included):
 *  - false: "0", "false", "FALSE", "no", "NO", "off", "OFF"
 *  - true:  "1", "true", "TRUE", "yes", "YES", "on", "ON"
 *
 * Any other non-empty value is treated as true.
 *
 * @param name Environment variable name (must not be NULL).
 * @param default_value Value used when variable is unset or empty.
 * @return 0 or 1.
 */
static int tok_env_flag_(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 ||
      strcmp(v, "no") == 0 || strcmp(v, "NO") == 0 || strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0) {
    return 0;
  }
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 ||
      strcmp(v, "yes") == 0 || strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 || strcmp(v, "ON") == 0) {
    return 1;
  }
  return 1;
}

/**
 * @brief Return 1 if @p c is ASCII whitespace, 0 otherwise.
 *
 * @param c Character byte (unsigned).
 * @return 1 if whitespace, 0 otherwise.
 */
static int is_space_byte_(unsigned char c) {
  return isspace((int)c) ? 1 : 0;
}

/**
 * @brief Compute 32-bit FNV-1a hash for a memory region.
 *
 * @details
 * The returned value occupies the full 32-bit range. Callers typically clamp
 * to a positive ID space for portability across signed/unsigned paths.
 *
 * @param ptr Pointer to the input bytes.
 * @param len Number of bytes.
 * @return 32-bit FNV-1a hash.
 */
static uint32_t fnv1a_32_(const void *ptr, size_t len) {
  const uint8_t *p = (const uint8_t *)ptr;
  uint32_t h = 2166136261u; /* offset basis */
  for (size_t i = 0; i < len; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;         /* FNV prime */
  }
  return h;
}

/**
 * @brief Clamp an unsigned 32-bit value into a positive 31-bit ID space.
 *
 * @details
 * Ensures the returned value is in the range [1, 0x7FFFFFFF].
 *
 * @param x Arbitrary 32-bit value.
 * @return Value in the range [1, 0x7FFFFFFF].
 */
static uint32_t clamp_to_pos31_(uint32_t x) {
  uint32_t y = x & 0x7FFFFFFFu;
  return (y == 0u) ? 1u : y;
}

/**
 * @brief Log a short, safe preview of a string for diagnostics.
 *
 * @details
 * Logs at most 96 bytes, replacing newlines/tabs with spaces.
 *
 * @param label Label for the log line (may be NULL).
 * @param s String to preview (may be NULL).
 */
static void tok_log_preview_(const char *label, const char *s) {
  if (!label) label = "text";
  if (!s) {
    ie_log_info("%s: (null)", label);
    return;
  }

  char tmp[128];
  size_t n = 0;
  for (; s[n] && n + 1 < sizeof(tmp) && n < 96; ++n) {
    char ch = s[n];
    if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
    tmp[n] = ch;
  }
  tmp[n] = '\0';

  ie_log_info("%s: bytes=%zu preview=\"%s%s\"", label, strlen(s), tmp, (s[n] ? "..." : ""));
}

/* ========================================================================== */
/* Public API: Vocabulary                                                     */
/* ========================================================================== */

/**
 * @brief Load a vocabulary file or default to a stub.
 *
 * @details
 * This implementation does not parse a full tokenizer model. It only attempts
 * a best-effort scan for a JSON-like field `"vocabSize": <int>` in the first
 * few KB of the file.
 *
 * If the file is missing/unreadable/unparsable, a deterministic stub value
 * is returned to keep tests stable.
 *
 * @param vocab_path Path to a vocabulary file (may be NULL or empty).
 * @param out Output vocabulary (must not be NULL).
 * @retval 0  Success (including stub fallback).
 * @retval -1 Invalid arguments.
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out) {
  if (!out) {
    ie_log_error("ie_vocab_load: bad args (out=NULL)");
    return -1;
  }

  /* Default stub value if anything goes wrong. */
  out->vocab_size = 50000;

  if (!vocab_path || !*vocab_path) {
    ie_log_info("ie_vocab_load: no path provided; using stub vocab_size=%d", out->vocab_size);
    return 0;
  }

  FILE *f = fopen(vocab_path, "rb");
  if (!f) {
    ie_log_info("ie_vocab_load: cannot open \"%s\" (errno=%d: %s); using stub vocab_size=%d",
                vocab_path,
                errno,
                strerror(errno),
                out->vocab_size);
    return 0;
  }

  /* Read a small prefix to try and find "vocabSize". */
  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  buf[n] = '\0';

  /* Very relaxed scan for a number after "vocabSize". */
  const char *key = "vocabSize";
  char *hit = strstr(buf, key);
  if (hit) {
    char *p = hit + (int)strlen(key);
    while (*p && (*p == ' ' || *p == '\t' || *p == '"' || *p == '\'' || *p == ':')) ++p;

    long val = 0;
    if (sscanf(p, "%ld", &val) == 1 && val > 0 && val < 100000000L) {
      out->vocab_size = (int)val;
      ie_log_info("ie_vocab_load: parsed vocab_size=%d from \"%s\"", out->vocab_size, vocab_path);
      return 0;
    }

    ie_log_info("ie_vocab_load: found key \"%s\" but failed to parse number; using stub vocab_size=%d",
                key,
                out->vocab_size);
    return 0;
  }

  ie_log_info("ie_vocab_load: key \"%s\" not found; using stub vocab_size=%d", key, out->vocab_size);
  return 0;
}

/**
 * @brief Free vocabulary resources.
 *
 * @details
 * This tokenizer implementation does not allocate dynamic resources for the vocab.
 * This function is kept for API symmetry and future expansion.
 *
 * @param v Vocabulary handle (may be NULL).
 */
void ie_vocab_free(ie_vocab_t *v) {
  (void)v;
}

/* ========================================================================== */
/* Public API: Encode/Decode                                                  */
/* ========================================================================== */

/**
 * @brief Encode UTF-8 text with whitespace tokenization and hashed IDs.
 *
 * @details
 * Tokenization rule: split on one or more ASCII whitespace characters.
 * Empty tokens are never produced.
 *
 * Size-only mode:
 *  - Pass @p ids == NULL to receive the required token count in @p out_count.
 *
 * @param v         Loaded vocabulary descriptor (unused by this implementation).
 * @param text      NUL-terminated UTF-8 string (must not be NULL).
 * @param ids       Output buffer (or NULL for size query).
 * @param out_count In: capacity of @p ids when @p ids != NULL.
 *                  Out: number of tokens written (or needed when @p ids == NULL).
 * @retval 0  Success.
 * @retval -1 Invalid arguments.
 * @retval -2 Insufficient capacity when @p ids != NULL.
 */
int ie_tok_encode(const ie_vocab_t *v,
                  const char *text,
                  uint32_t *ids,
                  uint32_t *out_count) {
  (void)v;
  if (!text || !out_count) {
    ie_log_error("ie_tok_encode: bad args (text=%p out_count=%p)", (const void *)text, (void *)out_count);
    return -1;
  }

  const int verbose = tok_env_flag_("IE_TOKENIZER_VERBOSE", 0);
  if (verbose) tok_log_preview_("ie_tok_encode input", text);

  const unsigned char *s = (const unsigned char *)text;
  const size_t L = strlen(text);

  uint32_t needed = 0;

  /* First pass: count tokens. */
  size_t i = 0;
  while (i < L) {
    while (i < L && is_space_byte_(s[i])) ++i;
    if (i >= L) break;

    size_t start = i;
    while (i < L && !is_space_byte_(s[i])) ++i;

    if (i > start) ++needed;
  }

  if (!ids) {
    *out_count = needed;
    if (verbose) {
      ie_log_info("ie_tok_encode: size-query result needed=%" PRIu32, needed);
    }
    return 0;
  }

  /* Second pass: produce IDs; respect capacity in *out_count. */
  const uint32_t cap = *out_count;
  uint32_t wrote = 0;

  i = 0;
  while (i < L && wrote < cap) {
    while (i < L && is_space_byte_(s[i])) ++i;
    if (i >= L) break;

    size_t start = i;
    while (i < L && !is_space_byte_(s[i])) ++i;

    const size_t tok_len = i - start;
    if (tok_len > 0) {
      const uint32_t h = fnv1a_32_(s + start, tok_len);
      ids[wrote++] = clamp_to_pos31_(h);
    }
  }

  *out_count = wrote;

  if (wrote != needed) {
    ie_log_error("ie_tok_encode: insufficient capacity (needed=%" PRIu32 " cap=%" PRIu32 " wrote=%" PRIu32 ")",
                 needed,
                 cap,
                 wrote);
    return -2;
  }

  if (verbose) {
    ie_log_info("ie_tok_encode: ok (tokens=%" PRIu32 ")", wrote);
  }
  return 0;
}

/**
 * @brief Decode token IDs to a deterministic placeholder string.
 *
 * @details
 * Output format:
 *   - "T<ID0> T<ID1> ... T<IDn>" (space-separated)
 *
 * This is intended for unit tests and harness debugging, not for real model text.
 *
 * @param v      Loaded vocabulary descriptor (unused by this implementation).
 * @param ids    Array of token IDs (may be NULL when @p count == 0).
 * @param count  Number of IDs.
 * @param out    Output buffer (must not be NULL).
 * @param out_sz Capacity of @p out in bytes (must be > 0).
 * @retval 0  Success.
 * @retval -1 Invalid arguments.
 * @retval -2 Output buffer too small or formatting truncated.
 */
int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out,
                  size_t out_sz) {
  (void)v;

  if (!out || out_sz == 0) {
    ie_log_error("ie_tok_decode: bad args (out=%p out_sz=%zu)", (void *)out, out_sz);
    return -1;
  }

  const int verbose = tok_env_flag_("IE_TOKENIZER_VERBOSE", 0);

  /* Handle empty sequence. */
  if (!ids || count == 0) {
    out[0] = '\0';
    if (verbose) ie_log_info("ie_tok_decode: empty input");
    return 0;
  }

  /* Conservative sizing: worst case "T4294967295 " per token + NUL. */
  const size_t worst_per = sizeof("T4294967295 ") - 1; /* 12 bytes */
  const size_t worst = (size_t)count * worst_per + 1;

  if (out_sz < worst) {
    ie_log_error("ie_tok_decode: output buffer too small (count=%" PRIu32 " out_sz=%zu need_at_least=%zu)",
                 count,
                 out_sz,
                 worst);
    return -2;
  }

  char *p = out;
  size_t rem = out_sz;

  for (uint32_t i = 0; i < count; ++i) {
    const int n = (i == 0)
      ? snprintf(p, rem, "T%u", (unsigned)ids[i])
      : snprintf(p, rem, " T%u", (unsigned)ids[i]);

    if (n < 0 || (size_t)n >= rem) {
      ie_log_error("ie_tok_decode: snprintf failed/truncated (i=%" PRIu32 " rem=%zu n=%d)", i, rem, n);
      return -2;
    }
    p += (size_t)n;
    rem -= (size_t)n;
  }

  if (rem == 0) {
    ie_log_error("ie_tok_decode: no space for NUL terminator (count=%" PRIu32 " out_sz=%zu)", count, out_sz);
    return -2;
  }
  *p = '\0';

  if (verbose) tok_log_preview_("ie_tok_decode output", out);
  return 0;
}
