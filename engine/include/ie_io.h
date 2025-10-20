/**
 * @file ie_io.h
 * @brief Model weights loader (IEBIN v1) and tokenizer (minimal).
 *
 * @defgroup IE_IO Model I/O and Tokenization
 * @brief Lightweight utilities for loading model metadata and encoding text.
 * @{
 */

#ifndef IE_IO_H
#define IE_IO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief In-memory handle for IEBIN v1 model metadata.
 *
 * This handle stores only light metadata required by the baseline engine:
 * - resolved paths to the JSON index and weights binary,
 * - the size of the weights binary (if present),
 * - a simple "version" integer and textual "dtype".
 *
 * The baseline loader does not mmap or materialize tensors. Those features
 * can be added in later steps without changing this public structure.
 */
typedef struct {
  char     weights_path[512];  /**< Resolved path to model.ie.bin (may be empty). */
  char     json_path[512];     /**< Resolved path to model.ie.json. */
  uint64_t bin_size_bytes;     /**< Size of model.ie.bin in bytes (0 allowed). */
  int      version;            /**< Metadata version (default 1). */
  char     dtype[16];          /**< "fp32"|"bf16"|"int8w" (default "fp32"). */
} ie_weights_t;

/**
 * @brief Open IEBIN v1 model metadata.
 *
 * This function performs **lightweight** validation and metadata extraction:
 * - Verifies that @p json_path exists and is a regular file.
 * - Records @p bin_path (if provided) and its file size (0 if absent).
 * - Parses @c "version" (integer) and @c "dtype" (string) via a relaxed scan.
 *
 * No heavy parsing or memory mapping is performed by this baseline routine.
 *
 * @param[in]  json_path  Path to @c model.ie.json (required, must exist).
 * @param[in]  bin_path   Path to @c model.ie.bin (optional; may be NULL or missing).
 * @param[out] out        Populated weights handle (must be non-NULL).
 * @return 0 on success; non-zero on failure (e.g., JSON missing/unreadable).
 */
int ie_weights_open(const char *json_path,
                    const char *bin_path,
                    ie_weights_t *out);

/**
 * @brief Close a weights handle.
 *
 * The baseline implementation is a no-op (kept for symmetry and to allow
 * future extensions that may allocate resources, e.g., mmaps).
 *
 * @param[in,out] w Weights handle (may be NULL).
 */
void ie_weights_close(ie_weights_t *w);

/**
 * @brief Tokenizer vocabulary descriptor (baseline).
 *
 * The baseline tokenizer keeps only the resolved path and a parsed/assumed
 * vocabulary size. Real tokenization logic can evolve independently.
 */
typedef struct {
  char vocab_path[512]; /**< Resolved path to vocab.json (may be empty). */
  int  vocab_size;      /**< Parsed from JSON if present; defaults to 256. */
} ie_vocab_t;

/**
 * @brief Load a vocabulary file.
 *
 * Attempts to read @p vocab_path and parse a @c "vocab_size" field. If the
 * file is absent or the field is missing, a safe default value is used.
 *
 * The exact tokenization scheme is intentionally left simple in the baseline.
 *
 * @param[in]  vocab_path  Path to @c vocab.json (may be NULL or missing).
 * @param[out] out         Populated vocabulary handle (must be non-NULL).
 * @return 0 on success; non-zero on failure.
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out);

/**
 * @brief Free resources held by a vocabulary handle.
 *
 * The baseline implementation is a no-op, provided for API symmetry and to
 * permit future implementations that allocate resources.
 *
 * @param[in,out] v Vocabulary handle (may be NULL).
 */
void ie_vocab_free(ie_vocab_t *v);

/**
 * @brief Encode UTF-8 text into token IDs (baseline placeholder).
 *
 * Baseline behavior:
 *  - Split on ASCII whitespace.
 *  - Hash each token deterministically (FNV-1a folded to 16 bits) and offset to
 *    the ID range @c [1000, 1000+65535].
 *
 * If @p out_ids is NULL, the function sets @p *out_count to the required
 * capacity and returns 0 without writing IDs.
 *
 * @param[in]  v          Loaded vocabulary handle (only @ref ie_vocab_t::vocab_size is used).
 * @param[in]  utf8       UTF-8 input text (may be NULL to indicate empty).
 * @param[out] out_ids    Buffer for token IDs or NULL for size query.
 * @param[out] out_count  On success, number of IDs written (or required capacity if @p out_ids is NULL).
 * @return 0 on success; non-zero on failure (invalid args).
 */
int ie_tok_encode(const ie_vocab_t *v,
                  const char *utf8,
                  uint32_t *out_ids,
                  uint32_t *out_count);

/**
 * @brief Decode token IDs into a placeholder string (baseline).
 *
 * Baseline format emits a space-separated sequence @c "T<ID>" for each token.
 *
 * @param[in]  v             Vocabulary handle (unused in baseline).
 * @param[in]  ids           Array of token IDs.
 * @param[in]  count         Number of IDs.
 * @param[out] out_utf8      Output buffer for UTF-8 text.
 * @param[in]  out_capacity  Capacity of @p out_utf8 in bytes.
 * @return 0 on success; non-zero if the buffer is too small or on invalid args.
 */
int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out_utf8,
                  size_t out_capacity);

#ifdef __cplusplus
}
#endif
/** @} */ /* end of IE_IO */

#endif /* IE_IO_H */
