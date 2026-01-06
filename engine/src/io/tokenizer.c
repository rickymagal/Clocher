/* ============================================================================
 * File: engine/src/io/tokenizer.c
 * ============================================================================
 */
/**
 * @file tokenizer.c
 * @brief Minimal, dependency-free tokenizer used by tests and the CLI harness.
 *
 * @details
 * This module provides a small public API declared in @ref ie_io.h:
 *  - @ref ie_vocab_load : Load a vocabulary file or fall back to a stub.
 *  - @ref ie_vocab_free : Release resources (no-op for this implementation).
 *  - @ref ie_tok_encode : Encode a UTF-8 string into token IDs.
 *  - @ref ie_tok_decode : Convert token IDs back into printable placeholders.
 *
 * Tokenization model:
 *  - Whitespace tokenization (ASCII whitespace via isspace()).
 *  - Stable IDs via 32-bit FNV-1a hash clamped to positive 31-bit.
 *
 * Logging:
 *  - INFO logs for vocabulary load decisions.
 *  - ERROR logs for invalid args and capacity issues.
 *  - Optional verbose logs (per-call) controlled by environment variables:
 *      - IE_TOKENIZER_VERBOSE=1 : Enable additional encode/decode details.
 *
 * This is intentionally not a real BPE/WordPiece tokenizer; its purpose is to
 * keep tests deterministic and provide a lightweight harness without extra artifacts.
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
 * Accepts common spellings:
 *  - false: "0", "false", "no", "off" (case-sensitive variants included)
 *  - true:  "1", "true", "yes", "on"
 * Unknown non-empty values are treated as true.
 *
 * @param name Environment variable name.
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
static int is_space_byte(unsigned char c) {
  return isspace((int)c) ? 1 : 0;
}

/**
 * @brief Compute 32-bit FNV-1a hash for a memory region.
 *
 * @details
 * Produces a stable ID for each token. The returned value is in the full
 * 32-bit range; callers may clamp to 31-bit positive domain as needed.
 *
 * @param ptr Pointer to bytes.
 * @param len Number of bytes.
 * @return 32-bit hash value.
 */
static uint32_t fnv1a_32(const void *ptr, size_t len) {
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
 * @param x Arbitrary 32-bit value.
 * @return Value in the range [1, 0x7FFFFFFF].
 */
static uint32_t clamp_to_pos31(uint32_t x) {
  uint32_t y = x & 0x7FFFFFFFu;
  return (y == 0u) ? 1u : y;
}

/**
 * @brief Log a short, safe preview of a string.
 *
 * @details
 * Logs at most 96 bytes, replacing newlines/tabs with spaces.
 *
 * @param label Label for the log line.
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
 * The implementation accepts any readable file but only attempts to detect
 * a simple JSON-like `"vocabSize": <int>` field. If the file is missing or
 * unparsable, a small stub vocab is returned to keep execution deterministic.
 *
 * Logging:
 *  - INFO when using stub because path is missing/unreadable/unparsable.
 *  - INFO when a vocab size is successfully parsed.
 *
 * @param vocab_path Path to a vocabulary file (may be NULL).
 * @param out Output vocabulary (written on success).
 * @retval 0  on success (including stub fallback).
 * @retval -1 on invalid arguments (e.g., @p out is NULL).
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
 * The function is kept for API symmetry and future expansion.
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
 * Tokenization rule: split on **one or more** whitespace characters (ASCII).
 * Consecutive whitespace collapses to a single separator, producing no empty
 * tokens. For example, `"hello  world"` yields 2 tokens.
 *
 * **Size-only mode:** pass `ids == NULL` to receive the required length in
 * `*out_count` without writing IDs.
 *
 * Logging:
 *  - ERROR on invalid args.
 *  - ERROR if capacity is insufficient (returns -2).
 *  - Optional verbose logs if IE_TOKENIZER_VERBOSE=1.
 *
 * @param v         Loaded vocabulary descriptor (only `vocab_size` is reserved for future use).
 * @param text      NUL-terminated UTF-8 string. Must not be NULL.
 * @param ids       Output buffer of length `*out_count` (or NULL for size query).
 * @param out_count In: capacity of `ids` when `ids != NULL`.
 *                  Out: number of tokens written (or needed).
 * @retval 0    on success.
 * @retval -1   on invalid arguments.
 * @retval -2   if provided capacity is insufficient to hold all tokens.
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
  uint32_t needed = 0;

  /* First pass: count tokens. */
  size_t i = 0;
  const size_t L = strlen(text);
  while (i < L) {
    while (i < L && is_space_byte(s[i])) ++i;
    if (i >= L) break;

    size_t start = i;
    while (i < L && !is_space_byte(s[i])) ++i;
    size_t tok_len = i - start;
    if (tok_len > 0) ++needed;
  }

  if (!ids) {
    *out_count = needed;
    if (verbose) {
      ie_log_info("ie_tok_encode: size-query result needed=%" PRIu32, needed);
    }
    return 0;
  }

  /* Second pass: produce IDs; respect capacity in *out_count. */
  uint32_t cap = *out_count;
  uint32_t wrote = 0;

  i = 0;
  while (i < L && wrote < cap) {
    while (i < L && is_space_byte(s[i])) ++i;
    if (i >= L) break;

    size_t start = i;
    while (i < L && !is_space_byte(s[i])) ++i;
    size_t tok_len = i - start;

    if (tok_len > 0) {
      uint32_t h = fnv1a_32(s + start, tok_len);
      ids[wrote++] = clamp_to_pos31(h);
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
 * @brief Decode token IDs to a simple, deterministic placeholder string.
 *
 * @details
 * Format: `"T<ID0> T<ID1> ... T<IDn>"` (space-separated). This is sufficient
 * for unit-test invariants that check spacing and a predictable prefix.
 *
 * Logging:
 *  - ERROR on invalid args.
 *  - ERROR if out buffer is too small (returns -2).
 *  - Optional verbose logs if IE_TOKENIZER_VERBOSE=1.
 *
 * @param v       Loaded vocabulary (unused but reserved).
 * @param ids     Array of token IDs.
 * @param count   Number of IDs.
 * @param out     Output buffer for the textual form.
 * @param out_sz  Capacity of @p out in bytes.
 * @retval 0    on success.
 * @retval -1   on invalid arguments (e.g., NULL @p out or zero @p out_sz).
 * @retval -2   if @p out is too small to hold the formatted string.
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
  size_t worst = (size_t)count * worst_per + 1;

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
    int n = (i == 0)
      ? snprintf(p, rem, "T%u", ids[i])
      : snprintf(p, rem, " T%u", ids[i]);

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
