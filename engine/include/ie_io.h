/* File: engine/include/ie_io.h
 * -----------------------------------------------------------------------------
 * @file ie_io.h
 * @brief Public I/O interfaces: IEBIN v1 loader and lightweight tokenizer.
 *
 * @details
 * Exposes:
 *  - IEBIN v1 weights loader: open/touch/close of model.ie.json + model.ie.bin
 *  - Lightweight tokenizer used by tests/harness
 *
 * The weights API is intentionally narrow:
 *  - The caller provides JSON and (optionally) BIN paths.
 *  - The loader resolves the effective binary path and basic metadata.
 *  - The loader may open optional "dedup artifacts" and stores an opaque handle
 *    in the descriptor, but does not expose the internal representation.
 *
 * No heap ownership is exposed through this public header. Implementations are
 * expected to keep ownership internal and only publish stable, fixed-size
 * descriptors to callers.
 */

#ifndef IE_IO_H_
#define IE_IO_H_

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * Status codes
 * ========================================================================== */

/**
 * @enum ie_io_status_e
 * @brief Status codes for I/O helpers and lightweight subsystems.
 *
 * @details
 * Negative values indicate failure categories that are stable enough for
 * diagnostics in the CLI/harness. Implementations may additionally set errno
 * for OS-level failures (open/stat/read), but errno is not required for every
 * IE_IO_ERR_* return code.
 */
typedef enum ie_io_status_e {
  IE_IO_OK = 0,             /**< Success. */
  IE_IO_ERR_ARGS = -1,      /**< Invalid arguments. */
  IE_IO_ERR_JSON = -2,      /**< JSON parse error or missing required fields. */
  IE_IO_ERR_BIN_UNSPEC = -3,/**< Binary path unspecified and could not be inferred. */
  IE_IO_ERR_STAT = -4,      /**< stat(2) failure or metadata mismatch. */
  IE_IO_ERR_OPEN = -5,      /**< open(2) failure. */
  IE_IO_ERR_READ = -6,      /**< read(2) failure or short read. */
  IE_IO_ERR_ALLOC = -7,     /**< Allocation failure. */
  IE_IO_ERR_DECODE = -8     /**< Decode failure (format mismatch/corruption). */
} ie_io_status_t;

/* ============================================================================
 * Weights (IEBIN v1)
 * ========================================================================== */

/**
 * @struct ie_weights_s
 * @brief In-memory descriptor for IEBIN v1 metadata and resolved paths.
 *
 * @details
 * The descriptor is designed for:
 *  - stack allocation,
 *  - deterministic cleanup via ie_weights_close(),
 *  - stable inspection by the CLI (dtype, resolved paths, bin size).
 *
 * The loader may optionally open dedup artifacts and store an opaque handle.
 * The handle is owned by the descriptor and must be released by ie_weights_close().
 */
typedef struct ie_weights_s {
  int    version;             /**< Parsed header version (>= 1). */
  char   dtype[16];           /**< Parsed dtype string (e.g., "fp32"). */

  char   json_path[512];      /**< Canonical path to opened JSON. */
  char   weights_path[512];   /**< Resolved weights binary path. */

  size_t bin_size_bytes;      /**< Size of weights binary in bytes. */
  int    loaded;              /**< Non-zero if open succeeded. */

  int    is_dedup;            /**< Non-zero if dedup artifacts were opened. */
  void  *dedup_handle;        /**< Opaque handle owned by this descriptor. */
} ie_weights_t;

/**
 * @brief Open and parse IEBIN v1 metadata and resolve the weights path.
 *
 * @details
 * On success:
 *  - out->loaded is non-zero,
 *  - out->json_path and out->weights_path are filled with resolved paths,
 *  - out->bin_size_bytes reflects the weights binary size,
 *  - out->dtype carries the declared precision/dtype for diagnostics.
 *
 * On failure:
 *  - a negative ie_io_status_t is returned,
 *  - out is left in a safe-to-close state (implementations should either memset
 *    to zero or ensure ie_weights_close() is idempotent).
 *
 * @param json_path Path to model.ie.json (readable).
 * @param bin_path  Optional override for model.ie.bin; may be NULL.
 * @param out       Output descriptor (written on success).
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out);

/**
 * @brief Touch the weights binary to verify readability (optional warm-up).
 *
 * @details
 * Implementations typically perform a fast sequential or page-stride read to
 * force the OS to fault pages into memory and catch permission/corruption issues
 * early. This function should be safe to call multiple times.
 *
 * @param w Opened descriptor.
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_touch(const ie_weights_t *w);

/**
 * @brief Close weights resources and release optional dedup handle.
 *
 * @details
 * This function must be idempotent: calling it on a zeroed or partially-opened
 * descriptor must be safe.
 *
 * @param w Descriptor to close (may be NULL).
 */
void ie_weights_close(ie_weights_t *w);

/* ============================================================================
 * Tokenizer (stub API used by tests and harness)
 * ========================================================================== */

/**
 * @struct ie_vocab_s
 * @brief Lightweight vocabulary descriptor.
 *
 * @details
 * This is intentionally minimal; the engine does not require a full tokenizer
 * for core benchmarking paths. Implementations can remain a stub/harness tool.
 */
typedef struct ie_vocab_s {
  int vocab_size; /**< Number of entries. */
} ie_vocab_t;

/**
 * @brief Load a vocabulary from vocab_path or fall back to a stub.
 *
 * @details
 * If vocab_path is NULL, the implementation may fall back to a simple stub
 * vocabulary suitable for tests and the benchmark harness.
 *
 * @param vocab_path Path to vocab file; may be NULL.
 * @param out        Output vocab.
 * @return 0 on success, negative on unrecoverable failure.
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out);

/**
 * @brief Encode UTF-8 text into token IDs (whitespace split, hashed IDs).
 *
 * @details
 * This is a stub tokenizer intended for harness/testing. It is not meant to be
 * compatible with a production BPE. The goal is stable behavior across runs.
 *
 * Size-only mode:
 *  - Pass ids==NULL to compute required token count in *out_count.
 *
 * @param v         Vocabulary descriptor.
 * @param text      NUL-terminated UTF-8 input.
 * @param ids       Output buffer or NULL for size query.
 * @param out_count In: capacity; Out: tokens written/needed.
 * @return 0 on success, negative on error.
 */
int ie_tok_encode(const ie_vocab_t *v,
                  const char *text,
                  uint32_t *ids,
                  uint32_t *out_count);

/**
 * @brief Decode token IDs into a printable placeholder string.
 *
 * @details
 * This function exists for tests and diagnostic tooling. It does not attempt
 * to reconstruct original text; it only provides a stable printable form.
 *
 * @param v      Vocabulary descriptor.
 * @param ids    Token IDs.
 * @param count  Number of IDs.
 * @param out    Output buffer.
 * @param out_sz Output buffer capacity in bytes.
 * @return 0 on success, negative on error.
 */
int ie_tok_decode(const ie_vocab_t *v,
                  const uint32_t *ids,
                  uint32_t count,
                  char *out,
                  size_t out_sz);

/**
 * @brief Release vocabulary resources (currently a no-op).
 *
 * @details
 * Present for API symmetry; implementations may later allocate resources.
 *
 * @param v Vocabulary pointer (may be NULL).
 */
void ie_vocab_free(ie_vocab_t *v);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_IO_H_ */

