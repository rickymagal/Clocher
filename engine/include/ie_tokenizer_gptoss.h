/* ==========================================================================
 * File: engine/include/ie_tokenizer_gptoss.h
 * ==========================================================================
 */
/**
 * @file ie_tokenizer_gptoss.h
 * @brief GPT-OSS tokenizer loader + encode/decode for:
 *        - HuggingFace `tokenizer.json` (GPT-2 Byte-Level BPE style)
 *        - OpenAI-style `.tiktoken` rank files (base64 token bytes + rank)
 *        - Packed `IETOK1` format (recommended)
 *
 * Design notes:
 *  - No third-party JSON library is used.
 *  - `.tiktoken` tokens are stored as raw bytes (binary-safe).
 *  - Decode must be correct for "token ids -> coherent text".
 *  - Encode is correctness-first. For `.tiktoken`, we apply byte-level BPE per
 *    whitespace/non-whitespace segments (conservative split) to avoid merging
 *    across obvious boundaries.
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

typedef struct ie_tok_gptoss_s ie_tok_gptoss_t;

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
 * @brief Open a tokenizer from:
 *  - tokenizer.json
 *  - tokenizer.tiktoken
 *  - packed IETOK1 file
 *
 * @param[in]  tokenizer_path Path to tokenizer file.
 * @param[out] out_tok        Tokenizer handle.
 * @return IE_TOK_GPTOSS_OK on success, negative error otherwise.
 */
int ie_tok_gptoss_open(const char *tokenizer_path, ie_tok_gptoss_t **out_tok);

void ie_tok_gptoss_close(ie_tok_gptoss_t *tok);

uint32_t ie_tok_gptoss_vocab_size(const ie_tok_gptoss_t *tok);

int ie_tok_gptoss_encode(const ie_tok_gptoss_t *tok,
                         const char *text,
                         uint32_t *ids,
                         uint32_t *inout_count);

int ie_tok_gptoss_decode(const ie_tok_gptoss_t *tok,
                         const uint32_t *ids,
                         uint32_t count,
                         char *out,
                         size_t *inout_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_TOKENIZER_GPTOSS_H_ */
