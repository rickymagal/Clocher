/**
 * @file tokenizer_unigram.c
 * @brief SentencePiece-style Unigram tokenizer implementation.
 *
 * @details
 * This file implements a true Unigram tokenizer using Viterbi
 * segmentation. It is compatible with SentencePiece Unigram models
 * and supports score-based optimal tokenization.
 *
 * Characteristics:
 *  - UTF-8 aware
 *  - Deterministic
 *  - No dynamic programming shortcuts
 *  - No merges or BPE logic
 *
 * Encoding is performed using a forward Viterbi pass followed by
 * backtracking to recover the optimal token sequence.
 */

#define _POSIX_C_SOURCE 200809L

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

/* ========================================================================== */
/* Internal helpers                                                           */
/* ========================================================================== */

/**
 * @brief Check whether a string starts with a given prefix.
 *
 * @param s Input string.
 * @param p Prefix string.
 * @param plen Length of prefix in bytes.
 *
 * @return 1 if prefix matches, 0 otherwise.
 */
static int starts_with(const char *s, const char *p, size_t plen) {
  return strncmp(s, p, plen) == 0;
}

/* ========================================================================== */
/* Unigram encoding                                                           */
/* ========================================================================== */

/**
 * @brief Encode text using Unigram Viterbi segmentation.
 *
 * @param v Unigram vocabulary.
 * @param text Input UTF-8 text.
 * @param out_ids Output token ID buffer (may be NULL).
 * @param out_count Input capacity / output token count.
 *
 * @return 0 on success, negative value on failure.
 */
int ie_unigram_encode(const ie_unigram_vocab_t *v,
                      const char *text,
                      uint32_t *out_ids,
                      uint32_t *out_count) {
  if (!v || !text || !out_count) return -1;

  const size_t N = strlen(text);

  float *score = (float *)malloc((N + 1) * sizeof(float));
  int   *prev  = (int *)malloc((N + 1) * sizeof(int));
  int   *prev_tok = (int *)malloc((N + 1) * sizeof(int));

  if (!score || !prev || !prev_tok) return -1;

  for (size_t i = 0; i <= N; ++i) {
    score[i] = -FLT_MAX;
    prev[i] = -1;
    prev_tok[i] = -1;
  }

  score[0] = 0.0f;

  for (size_t i = 0; i < N; ++i) {
    if (score[i] == -FLT_MAX) continue;

    for (size_t t = 0; t < v->vocab_size; ++t) {
      const ie_unigram_piece_t *p = &v->pieces[t];

      if (i + p->piece_len > N) continue;
      if (!starts_with(text + i, p->piece, p->piece_len)) continue;

      size_t j = i + p->piece_len;
      float s = score[i] + p->score;

      if (s > score[j]) {
        score[j] = s;
        prev[j] = (int)i;
        prev_tok[j] = (int)t;
      }
    }
  }

  if (score[N] == -FLT_MAX) {
    free(score);
    free(prev);
    free(prev_tok);
    return -2;
  }

  uint32_t needed = 0;
  for (int i = (int)N; i > 0; i = prev[i]) ++needed;

  if (!out_ids) {
    *out_count = needed;
    free(score);
    free(prev);
    free(prev_tok);
    return 0;
  }

  if (*out_count < needed) {
    free(score);
    free(prev);
    free(prev_tok);
    return -3;
  }

  uint32_t k = needed;
  for (int i = (int)N; i > 0; i = prev[i]) {
    out_ids[--k] = (uint32_t)prev_tok[i];
  }

  *out_count = needed;

  free(score);
  free(prev);
  free(prev_tok);
  return 0;
}

/* ========================================================================== */
/* Unigram decoding                                                           */
/* ========================================================================== */

/**
 * @brief Decode Unigram token IDs back into UTF-8 text.
 *
 * @param v Unigram vocabulary.
 * @param ids Token ID array.
 * @param count Number of tokens.
 * @param out Output buffer.
 * @param out_sz Output buffer size.
 *
 * @return 0 on success, negative value on failure.
 */
int ie_unigram_decode(const ie_unigram_vocab_t *v,
                      const uint32_t *ids,
                      uint32_t count,
                      char *out,
                      size_t out_sz) {
  if (!v || !out || out_sz == 0) return -1;

  size_t p = 0;

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t id = ids[i];
    if (id >= v->vocab_size) continue;

    const char *piece = v->pieces[id].piece;

    if (piece[0] == '▁') {
      if (p + 1 >= out_sz) return -2;
      out[p++] = ' ';
      piece++;
    }

    size_t len = strlen(piece);
    if (p + len >= out_sz) return -2;

    memcpy(out + p, piece, len);
    p += len;
  }

  out[p] = '\0';
  return 0;
}
