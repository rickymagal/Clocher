/**
 * @file ie_io.h
 * @brief Public I/O interfaces: IEBIN v1 loader and lightweight tokenizer.
 *
 * @details
 * This header exposes two small subsystems used by the CLI, tests, and engine:
 *
 * - **IEBIN v1 weights loader** — opens `model.ie.json`, resolves the
 *   corresponding `model.ie.bin`, captures basic metadata, and provides a
 *   light touch/read check. See @ref ie_weights_open and @ref ie_weights_touch.
 *
 * - **Tokenizer (stub)** — tiny, dependency-free tokenizer API used by tests
 *   and by the engine’s harness. The reference implementation in
 *   `tokenizer.c` supports a whitespace splitter with hashed-token placeholders.
 *   See @ref ie_vocab_load, @ref ie_tok_encode and @ref ie_tok_decode.
 *
 * Goals:
 * - Stable C interface that compiles cleanly with `-Wall -Wextra -Werror`.
 * - No heap ownership in public structs; zero-init + explicit close/free.
 */

#ifndef IE_IO_H_
#define IE_IO_H_

/* Request POSIX features for some libc prototypes used by implementations
 * (e.g., pread). This is safe in headers as it only widens feature visibility.
 */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  /* size_t */
#include <stdint.h>  /* uint32_t */

/* ============================================================================
 * Status codes
 * ========================================================================== */
/**
 * @defgroup ie_io_status Status Codes
 * @brief Error/status values shared by the I/O helpers.
 * @{
 */

/**
 * @enum ie_io_status_e
 * @brief Status codes for I/O helpers and lightweight subsystems.
 *
 * These values are returned by the IEBIN loader and related utilities.
 */
typedef enum ie_io_status_e {
  /** Success. */
  IE_IO_OK = 0,

  /** Invalid arguments (NULL pointer, bad sizes, etc.). */
  IE_IO_ERR_ARGS = -1,

  /** Failed to read/parse JSON metadata. */
  IE_IO_ERR_JSON = -2,

  /** JSON missing `bin` key and no override provided. */
  IE_IO_ERR_BIN_UNSPEC = -3,

  /** Failed to stat() the weights file. */
  IE_IO_ERR_STAT = -4,

  /** Failed to open the weights file. */
  IE_IO_ERR_OPEN = -5,

  /** Failed to read from the weights file. */
  IE_IO_ERR_READ = -6,

  /** Allocation failure (OOM or NULL from malloc/calloc/realloc). */
  IE_IO_ERR_ALLOC = -7,

  /** Decode/format error (e.g., quantized weights decode failure). */
  IE_IO_ERR_DECODE = -8
} ie_io_status_t;
/** @} */ /* end of ie_io_status */

/* ============================================================================
 * Weights (IEBIN v1)
 * ========================================================================== */
/**
 * @defgroup ie_io_weights IEBIN v1 Weights Loader
 * @brief Open/inspect lightweight model container files.
 * @{
 */

/**
 * @struct ie_weights_s
 * @brief In-memory descriptor for IEBIN v1 metadata and resolved paths.
 *
 * @details
 * Instances are trivially copyable; no owned heap pointers are stored here.
 * Use @ref ie_weights_open to fill the structure, and @ref ie_weights_close
 * for symmetry. The binary is not kept open by this descriptor.
 */
typedef struct ie_weights_s {
  /* ---- JSON header ---- */

  /** Parsed header `version` (>= 1). */
  int      version;

  /** Parsed header `dtype` (e.g., "fp32", "int8", "mixed"). */
  char     dtype[16];

  /* ---- Resolved locations ---- */

  /** Absolute or canonical path to the JSON actually opened. */
  char     json_path[512];

  /** Resolved path to the weights binary (`model.ie.bin`). */
  char     weights_path[512];

  /* ---- Binary info ---- */

  /** Size of `weights_path` in bytes (from stat). */
  size_t   bin_size_bytes;

  /* ---- Bookkeeping ---- */

  /** Non-zero if @ref ie_weights_open succeeded. */
  int      loaded;
} ie_weights_t;

/**
 * @brief Open and parse IEBIN v1 metadata and resolve the weights path.
 *
 * @param json_path Path to `model.ie.json` (must be readable).
 * @param bin_path  Optional override path for the binary (`model.ie.bin`);
 *                  if NULL, the JSON `bin` field (or default) is used.
 * @param out       Output descriptor; zero-initialized on entry by the function.
 * @retval IE_IO_OK             on success.
 * @retval IE_IO_ERR_ARGS       if arguments are invalid.
 * @retval IE_IO_ERR_JSON       if JSON cannot be read/parsed.
 * @retval IE_IO_ERR_BIN_UNSPEC if a weights path cannot be determined.
 * @retval IE_IO_ERR_STAT       if the binary cannot be `stat()`-ed.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out);

/**
 * @brief Touch the weights binary to verify readability (and optionally warm caches).
 *
 * @details
 * Performs a small positional read near offset 0 and near the end of the file
 * (if large enough). Intended as a fast sanity check during startup or tests.
 * The implementation may be a no-op on platforms without `pread`.
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
 * @param w Weights descriptor previously opened with @ref ie_weights_open (may be NULL).
 */
void ie_weights_close(ie_weights_t *w);
/** @} */ /* end of ie_io_weights */

/* ============================================================================
 * Tokenizer (stub API used by tests and harness)
 * ========================================================================== */
/**
 * @defgroup ie_io_tokenizer Tokenizer (Stub)
 * @brief Lightweight, dependency-free tokenizer used in tests and harness.
 * @{
 */

/**
 * @struct ie_vocab_s
 * @brief Lightweight vocabulary descriptor.
 *
 * @details
 * The reference implementation accepts either a JSON-like file or a simple text
 * list; when absent, it falls back to a stub vocabulary with a positive
 * `vocab_size`. Internal fields are intentionally omitted to keep the public
 * surface small and stable.
 */
typedef struct ie_vocab_s {
  /** Number of entries; stub sets this to a positive value. */
  int vocab_size;
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
 * @param v         Loaded vocabulary (must have positive @c vocab_size).
 * @param text      NUL-terminated UTF-8 input string.
 * @param ids       Output buffer of length `*out_count` (or NULL for size query).
 * @param out_count In: capacity of @p ids when `ids != NULL`.  
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
/** @} */ /* end of ie_io_tokenizer */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_IO_H_ */
