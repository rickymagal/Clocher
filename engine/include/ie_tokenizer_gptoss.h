/* engine/include/ie_tokenizer_gptoss.h */
/**
 * @file ie_tokenizer_gptoss.h
 * @brief GPT-OSS (GPT-2 style) Byte-Level BPE tokenizer (tokenizer.json loader).
 *
 * This module is intended to:
 *  - Load HuggingFace `tokenizer.json` (ByteLevelBPETokenizer / GPT-2 BPE format).
 *  - Provide encode/decode between UTF-8 text and vocab token IDs.
 *
 * Design notes:
 *  - No third-party JSON library is used. We parse only what we need using a
 *    small, strict scanner tailored to tokenizer.json layout.
 *  - Decode is the most important part for "tokens -> coherent text".
 *  - Encode is implemented (basic pretokenization + BPE), but if you need exact
 *    parity with HuggingFace tokenization (regex rules, special token handling),
 *    you may refine the pretokenizer later.
 *
 * Thread-safety:
 *  - A tokenizer handle is read-only after open. Encode/decode are re-entrant
 *    as long as the caller provides independent output buffers.
 */

#ifndef IE_TOKENIZER_GPTOSS_H_
#define IE_TOKENIZER_GPTOSS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/** Opaque tokenizer handle. */
typedef struct ie_tok_gptoss_s ie_tok_gptoss_t;

/** Status codes for tokenizer operations. */
typedef enum ie_tok_gptoss_status_e {
  IE_TOK_GPTOSS_OK = 0,
  IE_TOK_GPTOSS_ERR_ARGS = -1,
  IE_TOK_GPTOSS_ERR_IO = -2,
  IE_TOK_GPTOSS_ERR_JSON = -3,
  IE_TOK_GPTOSS_ERR_NOMEM = -4,
  IE_TOK_GPTOSS_ERR_RANGE = -5,
  IE_TOK_GPTOSS_ERR_INTERNAL = -6
} ie_tok_gptoss_status_t;

/**
 * @brief Open a GPT-OSS tokenizer from a HuggingFace `tokenizer.json`.
 *
 * @param tokenizer_json_path Path to tokenizer.json.
 * @param out_tok             Output handle (set on success).
 * @return IE_TOK_GPTOSS_OK on success, negative status on error.
 */
int ie_tok_gptoss_open(const char *tokenizer_json_path, ie_tok_gptoss_t **out_tok);

/**
 * @brief Close and free a tokenizer handle.
 *
 * @param tok Tokenizer handle (may be NULL).
 */
void ie_tok_gptoss_close(ie_tok_gptoss_t *tok);

/**
 * @brief Get vocabulary size.
 *
 * @param tok Tokenizer handle.
 * @return vocab size on success, 0 on error.
 */
uint32_t ie_tok_gptoss_vocab_size(const ie_tok_gptoss_t *tok);

/**
 * @brief Encode UTF-8 text into vocab token IDs.
 *
 * Usage:
 *  - Size query: ids==NULL, set *inout_count to 0; function writes required size.
 *  - Encode: provide ids buffer and capacity in *inout_count.
 *
 * @param tok          Tokenizer handle.
 * @param text         NUL-terminated UTF-8 text.
 * @param ids          Output token id buffer (may be NULL for size query).
 * @param inout_count  In: capacity; Out: tokens written/required.
 * @return IE_TOK_GPTOSS_OK on success, negative status on error.
 */
int ie_tok_gptoss_encode(const ie_tok_gptoss_t *tok,
                         const char *text,
                         uint32_t *ids,
                         uint32_t *inout_count);

/**
 * @brief Decode vocab token IDs into UTF-8 text.
 *
 * Usage:
 *  - Size query: out==NULL, set *inout_bytes=0; function writes required bytes
 *    including the trailing NUL.
 *  - Decode: provide out buffer and capacity in *inout_bytes.
 *
 * @param tok           Tokenizer handle.
 * @param ids           Token IDs.
 * @param count         Number of IDs.
 * @param out           Output buffer (may be NULL for size query).
 * @param inout_bytes   In: capacity; Out: bytes written/required (including NUL).
 * @return IE_TOK_GPTOSS_OK on success, negative status on error.
 */
int ie_tok_gptoss_decode(const ie_tok_gptoss_t *tok,
                         const uint32_t *ids,
                         uint32_t count,
                         char *out,
                         size_t *inout_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_TOKENIZER_GPTOSS_H_ */
