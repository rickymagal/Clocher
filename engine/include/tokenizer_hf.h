#ifndef TOKENIZER_HF_H_
#define TOKENIZER_HF_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * @file tokenizer_hf.h
 * @brief HuggingFace GPT-style tokenizer (decode-focused).
 *
 * This module loads a HuggingFace `tokenizer.json` file and supports
 * decoding token IDs into real UTF-8 text using the GPT-2 byte decoder.
 *
 * Encoding is optional and intentionally omitted here; this is meant
 * to unblock REAL TEXT GENERATION from model token outputs.
 */

/* ------------------------------------------------------------------------- */
/* Tokenizer handle                                                          */
/* ------------------------------------------------------------------------- */

typedef struct tokenizer_hf_s {
  char   **id_to_token;   /* token_id -> UTF-8 token string */
  uint32_t vocab_size;
  int loaded;
} tokenizer_hf_t;

/* ------------------------------------------------------------------------- */
/* Lifecycle                                                                  */
/* ------------------------------------------------------------------------- */

/**
 * @brief Load tokenizer.json from disk.
 *
 * @param path Path to HuggingFace tokenizer.json
 * @param out  Tokenizer object to initialize
 * @return 0 on success, non-zero on failure
 */
int tokenizer_hf_load(const char *path, tokenizer_hf_t *out);

/**
 * @brief Free tokenizer resources.
 *
 * @param tok Tokenizer object
 */
void tokenizer_hf_free(tokenizer_hf_t *tok);

/* ------------------------------------------------------------------------- */
/* Decode                                                                     */
/* ------------------------------------------------------------------------- */

/**
 * @brief Decode token IDs into UTF-8 text.
 *
 * Uses GPT-2 byte-level decoding.
 *
 * @param tok     Loaded tokenizer
 * @param ids     Token ID array
 * @param n_ids   Number of token IDs
 * @param out     Output buffer
 * @param out_sz  Output buffer capacity
 * @return 0 on success, non-zero on failure
 */
int tokenizer_hf_decode(const tokenizer_hf_t *tok,
                         const int *ids,
                         size_t n_ids,
                         char *out,
                         size_t out_sz);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_HF_H_ */
