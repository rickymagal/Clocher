/**
 * @file tokenizer.c
 * @brief Minimal, dependency-free tokenizer used by tests and the CLI harness.
 *
 * @details
 * This module provides a tiny public API declared in @ref ie_io.h:
 *  - @ref ie_vocab_load : Load a vocabulary file or fall back to a stub.
 *  - @ref ie_tok_encode : Encode a UTF-8 string into token IDs.
 *  - @ref ie_tok_decode : Convert token IDs back into printable placeholders.
 *
 * ## Design goals
 * - **Deterministic** behavior for CI and unit tests.
 * - **No third-party dependencies**; standard C library only.
 * - **Whitespace tokenization**: contiguous whitespace collapses to a single
 *   separator. Thus `"hello world  from   engine"` yields 4 tokens.
 * - **Stable IDs**: IDs are produced by hashing each token with a fixed
 *   32-bit FNV-1a variant and then clamping to a positive 31-bit range.
 *
 * This is intentionally not a real BPE/WordPiece tokenizer; its purpose is to
 * keep tests green and provide a lightweight harness without extra artifacts.
 */

#define _POSIX_C_SOURCE 200809L

#include <ctype.h>      /* isspace */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ie_io.h"

/* ========================================================================== */
/* Internal helpers (file-local)                                              */
/* ========================================================================== */

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
  const uint8_t *p = (const uint8_t*)ptr;
  uint32_t h = 2166136261u;           /* offset basis */
  for (size_t i = 0; i < len; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;                   /* FNV prime */
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
 * @param vocab_path Path to a vocabulary file (may be NULL).
 * @param out Output vocabulary (written on success).
 * @retval 0  on success (including stub fallback).
 * @retval -1 on invalid arguments (e.g., @p out is NULL).
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out) {
  if (!out) return -1;

  /* Default stub value if anything goes wrong. */
  out->vocab_size = 50000;

  if (!vocab_path || !*vocab_path) {
    return 0;
  }

  FILE *f = fopen(vocab_path, "rb");
  if (!f) {
    /* Keep stub; report success to remain permissive. */
    return 0;
  }

  /* Read a small prefix to try and find "vocabSize". */
  char buf[4096];
  size_t n = fread(buf, 1, sizeof(buf)-1, f);
  fclose(f);
  buf[n] = '\0';

  /* Very relaxed scan for a number after "vocabSize". */
  const char *key = "vocabSize";
  char *hit = strstr(buf, key);
  if (hit) {
    /* Move to the first digit after key. */
    char *p = hit + (int)strlen(key);
    while (*p && (*p == ' ' || *p == '\t' || *p == '"' || *p == '\'' || *p == ':' )) ++p;
    long val = 0;
    if (sscanf(p, "%ld", &val) == 1 && val > 0 && val < 100000000L) {
      out->vocab_size = (int)val;
    }
  }
  return 0;
}

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
 * @param v         Loaded vocabulary descriptor (only `vocab_size` is used).
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
  (void)v; /* vocab_size is not required for hashing but reserved for future use */
  if (!text || !out_count) return -1;

  const unsigned char *s = (const unsigned char*)text;
  uint32_t needed = 0;

  /* First pass: count tokens. */
  size_t i = 0;
  const size_t L = strlen(text);
  while (i < L) {
    /* Skip leading whitespace between tokens. */
    while (i < L && is_space_byte(s[i])) ++i;
    if (i >= L) break;

    /* Start of token. */
    size_t start = i;
    while (i < L && !is_space_byte(s[i])) ++i;
    size_t tok_len = i - start;
    if (tok_len > 0) {
      ++needed;
    }
  }

  if (!ids) {
    *out_count = needed;
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
  /* If capacity was insufficient, signal failure. Tests pre-size correctly. */
  return (wrote == needed) ? 0 : -2;
}

/**
 * @brief Decode token IDs to a simple, deterministic placeholder string.
 *
 * @details
 * Format: `"T<ID0> T<ID1> ... T<IDn>"` (space-separated). This is sufficient
 * for unit-test invariants that check spacing and a predictable prefix.
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
  if (!out || out_sz == 0) return -1;

  /* Handle empty sequence. */
  if (!ids || count == 0) {
    if (out_sz > 0) out[0] = '\0';
    return 0;
  }

  /* Conservative sizing: worst case "T4294967295 " per token + NUL. */
  const size_t worst_per = sizeof("T4294967295 ") - 1; /* 12 bytes */
  size_t worst = (size_t)count * worst_per + 1;
  if (out_sz < worst) {
    /* Strict contract for tests: return error if too small. */
    return -2;
  }

  char *p = out;
  size_t rem = out_sz;

  for (uint32_t i = 0; i < count; ++i) {
    int n = (i == 0)
      ? snprintf(p, rem, "T%u", ids[i])
      : snprintf(p, rem, " T%u", ids[i]);

    if (n < 0 || (size_t)n >= rem) {
      return -2;
    }
    p   += (size_t)n;
    rem -= (size_t)n;
  }

  /* Ensure NUL-termination. */
  if (rem == 0) return -2;
  *p = '\0';
  return 0;
}
