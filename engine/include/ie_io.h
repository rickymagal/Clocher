/**
 * @file ie_io.h
 * @brief Minimal public I/O interfaces: IEBIN v1 loader and tokenizer stubs.
 *
 * @details
 * This header exposes two small subsystems used by the CLI and tests:
 *
 * 1) **IEBIN v1 weights loader** — opens `model.ie.json`, resolves
 *    `model.ie.bin`, captures basic metadata, and provides a light touch/read
 *    check. See @ref ie_weights_open and @ref ie_weights_touch.
 *
 * 2) **Tokenizer (stub)** — tiny, dependency-free tokenizer API used by tests
 *    and by the engine’s harness. The reference implementation in
 *    @ref tokenizer.c supports a whitespace splitter with hashed-token
 *    placeholders. Its purpose is to keep CI green and avoid large third-party
 *    dependencies. See @ref ie_vocab_load, @ref ie_tok_encode and
 *    @ref ie_tok_decode.
 *
 * The goal is a stable, warning-free C interface that works with `-Werror`.
 */

#ifndef IE_IO_H_
#define IE_IO_H_

/* POSIX feature request: needed for some libc prototypes (e.g., pread) in impls. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint32_t */

/* ========================================================================== */
/* Status codes                                                               */
/* ========================================================================== */

/**
 * @enum ie_io_status_e
 * @brief Status codes for I/O helpers.
 *
 * @var IE_IO_OK
 * Success.
 * @var IE_IO_ERR_ARGS
 * Invalid arguments.
 * @var IE_IO_ERR_JSON
 * Failed to read/parse JSON metadata.
 * @var IE_IO_ERR_BIN_UNSPEC
 * JSON missing `bin` key and no override provided.
 * @var IE_IO_ERR_STAT
 * Failed to stat the weights file.
 * @var IE_IO_ERR_OPEN
 * Failed to open the weights file.
 * @var IE_IO_ERR_READ
 * Failed to read the weights file.
 */
typedef enum ie_io_status_e {
  IE_IO_OK = 0,
  IE_IO_ERR_ARGS = -1,
  IE_IO_ERR_JSON = -2,
  IE_IO_ERR_BIN_UNSPEC = -3,
  IE_IO_ERR_STAT = -4,
  IE_IO_ERR_OPEN = -5,
  IE_IO_ERR_READ = -6
} ie_io_status_t;

/* ========================================================================== */
/* Weights (IEBIN v1)                                                         */
/* ========================================================================== */

/**
 * @struct ie_weights_s
 * @brief In-memory descriptor for IEBIN v1 metadata and resolved paths.
 *
 * @details
 * Instances are trivially copyable; no owned heap pointers are stored here.
 * Use @ref ie_weights_open to fill, and @ref ie_weights_close for symmetry.
 */
typedef struct ie_weights_s {
  /* ---- JSON header ---- */
  int      version;                 /**< Parsed header `version` (>=1). */
  char     dtype[16];               /**< Parsed header `dtype` (e.g., "float32"). */

  /* ---- Resolved locations ---- */
  char     json_path[512];          /**< Path to the JSON metadata actually opened. */
  char     weights_path[512];       /**< Resolved path to the weights binary. */

  /* ---- Binary info ---- */
  size_t   bin_size_bytes;          /**< Size of `weights_path` in bytes. */

  /* ---- Bookkeeping ---- */
  int      loaded;                  /**< Non-zero if open succeeded. */
} ie_weights_t;

/**
 * @brief Open and parse IEBIN v1 metadata and resolve the weights path.
 *
 * @param json_path Path to `model.ie.json` (must be readable).
 * @param bin_path  Optional override path for the binary (`model.ie.bin`);
 *                  if NULL, the JSON `bin` field is used.
 * @param out       Output descriptor; the function zero-initializes it on entry.
 * @retval IE_IO_OK            on success.
 * @retval IE_IO_ERR_ARGS      if arguments are invalid.
 * @retval IE_IO_ERR_JSON      if JSON cannot be read/parsed.
 * @retval IE_IO_ERR_BIN_UNSPEC if a weights path cannot be determined.
 * @retval IE_IO_ERR_STAT      if the binary cannot be `stat()`-ed.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out);

/**
 * @brief Touch the weights binary to verify readability (and optionally warm caches).
 *
 * @details
 * Performs a small positional read at offset 0 and at the end of file (if large
 * enough). Intended as a fast sanity check during startup or tests. The
 * implementation may be a no-op if the platform lacks `pread`.
 *
 * @param w Opened weights descriptor (must come from @ref ie_weights_open).
 * @retval IE_IO_OK        on success (binary considered readable).
 * @retval IE_IO_ERR_ARGS  on invalid arguments or unopened descriptor.
 * @retval IE_IO_ERR_OPEN  if the binary cannot be opened.
 * @retval IE_IO_ERR_READ  if the binary cannot be read.
 */
int ie_weights_touch(const ie_weights_t *w);

/**
 * @brief Close weights resources (currently a no-op; reserved for future use).
 *
 * @param w Weights descriptor previously opened with @ref ie_weights_open.
 */
void ie_weights_close(ie_weights_t *w);

/* ========================================================================== */
/* Tokenizer (stub API used by tests and harness)                              */
/* ========================================================================== */

/**
 * @struct ie_vocab_s
 * @brief Lightweight vocabulary descriptor.
 *
 * @details
 * The reference implementation accepts either a JSON-like file or a simple text
 * list; when absent, it falls back to a stub vocabulary with a positive
 * `vocab_size`. Internal fields are intentionally omitted to keep the public
 * surface small.
 */
typedef struct ie_vocab_s {
  int vocab_size;       /**< Number of entries; stub sets this to a positive value. */
  /* Internal fields are intentionally omitted to keep the public surface small. */
} ie_vocab_t;

/**
 * @brief Load a vocabulary from @p vocab_path (or default to a stub).
 *
 * @details
 * Implementations should never fail hard: for reproducible CI, returning a small
 * stub vocab is valid. The function attempts to parse a `vocabSize` integer if a
 * JSON-like file is provided; otherwise a default size is used.
 *
 * @param vocab_path Path to a vocab file (e.g., `vocab.json`); may be NULL.
 * @param out        Output vocabulary (written on success).
 * @retval 0 on success (including stub fallback).
 * @retval negative on unrecoverable failure (e.g., @p out is NULL).
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out);

/**
 * @brief Encode UTF-8 text into token IDs (whitespace split, hashed IDs).
 *
 * @details
 * **Size-only mode:** pass `ids == NULL` to compute the required length in
 * `*out_count` without writing token data.
 *
 * @param v         Loaded vocabulary.
 * @param text      NUL-terminated UTF-8 text.
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
                  uint32_t *out_count);

/**
 * @brief Decode token IDs into a printable placeholder string.
 *
 * @details
 * Implementations may produce a stable, testing-friendly textual form rather
 * than true detokenization. The reference implementation returns
 * `"T<ID0> T<ID1> ..."` into @p out.
 *
 * @param v       Loaded vocabulary.
 * @param ids     Array of token IDs.
 * @param count   Number of IDs (elements in @p ids).
 * @param out     Output character buffer (NUL-terminated on success).
 * @param out_sz  Capacity of @p out in bytes.
 * @retval 0    on success.
 * @retval -1   on invalid arguments or zero-sized buffer.
 * @retval -2   if @p out is too small to hold the formatted sequence.
 */
int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out,
                  size_t out_sz);


/**
 * @brief Release resources held by a vocabulary.
 *
 * @details
 * The current tokenizer implementation does not heap-allocate inside
 * @ref ie_vocab_t, so this function is a no-op. It exists for symmetry and
 * forward compatibility if the tokenizer later gains owned resources.
 *
 * @param v Vocabulary pointer (may be NULL; no-op).
 */
void ie_vocab_free(ie_vocab_t *v);
  
#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_IO_H_ */
