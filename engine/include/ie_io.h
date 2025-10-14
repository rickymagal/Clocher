/**
 * @file ie_io.h
 * @brief Model weights loader (IEBIN v1) and tokenizer (minimal, dependency-free).
 *
 * IEBIN v1 layout:
 *   models/<name>/
 *     model.ie.json   // metadata (version, dtype, optional tensor entries)
 *     model.ie.bin    // flat little-endian blob (optional for baseline)
 *     vocab.json      // tokenizer spec (BPE/WordPiece-like; baseline: stub)
 *
 * This header exposes a minimal API so the engine can:
 *   - open/close weights (validate basic schema, record sizes/paths)
 *   - load/free vocabulary
 *   - encode/decode tokens (baseline: whitespace fallback)
 *
 * All functions return 0 on success; non-zero on error.
 */

#ifndef IE_IO_H
#define IE_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Weights handle (opaque to users of the API). */
typedef struct {
  char    weights_path[512];
  char    json_path[512];
  uint64_t bin_size_bytes;     /**< Size of model.ie.bin (0 allowed in baseline). */
  int      version;            /**< Parsed from model.ie.json (default 1). */
  char     dtype[16];          /**< "fp32"|"bf16"|"int8w" (default "fp32"). */
} ie_weights_t;

/** @brief Open IEBIN v1 weights. Only does lightweight validation in baseline. */
int ie_weights_open(const char *json_path,
                    const char *bin_path,
                    ie_weights_t *out);

/** @brief Close weights handle (no-op in baseline). */
void ie_weights_close(ie_weights_t *w);

/** @brief Vocabulary handle (opaque). */
typedef struct {
  char   vocab_path[512];
  int    vocab_size;           /**< From vocab.json if present; else fallback. */
} ie_vocab_t;

/** @brief Load vocabulary (baseline: reads file if exists; tolerates stub). */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out);

/** @brief Free vocabulary (no-op in baseline). */
void ie_vocab_free(ie_vocab_t *v);

/**
 * @brief Encode UTF-8 text into token IDs.
 *
 * Baseline behavior:
 *   - Split on ASCII whitespace.
 *   - Each token hashed deterministically to a pseudo-ID in [1000, 1000+65535].
 *   - If out_ids is NULL, function returns required capacity in *out_count.
 */
int ie_tok_encode(const ie_vocab_t *v,
                  const char *utf8,
                  uint32_t *out_ids,
                  uint32_t *out_count);

/**
 * @brief Decode token IDs back to a placeholder string (baseline).
 * Produces a simple space-separated "T<ID>" sequence.
 */
int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out_utf8,
                  size_t out_capacity);

#ifdef __cplusplus
}
#endif
#endif /* IE_IO_H */
