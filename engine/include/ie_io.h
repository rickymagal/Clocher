/* ============================================================================
 * File: engine/include/ie_io.h
 * ============================================================================
 */
/**
 * @file ie_io.h
 * @brief Public I/O interfaces for model artifacts and lightweight tooling.
 *
 * @details
 * This header defines the **only stable public interface** for:
 *
 *  - IEBIN v1 weight loading (model.ie.json + model.ie.bin)
 *  - Optional deduplicated-weight artifact handling
 *  - INT4 metadata inspection helpers
 *  - Lightweight tokenizer / vocabulary helpers used by tests and harnesses
 *
 * ### Design constraints
 * - No heap ownership is exposed to callers.
 * - All descriptors are POD structs suitable for stack allocation.
 * - Cleanup is deterministic and idempotent.
 * - No third-party dependencies (no JSON libraries).
 *
 * ### What this header deliberately does NOT provide
 * - No tensor lookup by name (handled by tensor_map).
 * - No inference-time APIs.
 * - No HuggingFace tokenizer (handled by tokenizer_hf).
 *
 * This file is part of **Path 2 (real inference)** but remains usable by
 * diagnostic tools, benchmarks, and CI without CUDA or inference enabled.
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
 * @brief Status codes returned by I/O and artifact-loading helpers.
 *
 * @details
 * All non-zero values indicate failure. The categories are stable and suitable
 * for:
 *  - CLI diagnostics
 *  - benchmark harness error reporting
 *
 * Implementations may set `errno` for OS-level failures, but this is not
 * guaranteed for every failure mode.
 */
typedef enum ie_io_status_e {
  IE_IO_OK = 0,              /**< Success. */
  IE_IO_ERR_ARGS = -1,       /**< Invalid arguments. */
  IE_IO_ERR_JSON = -2,       /**< JSON parse failure or missing required fields. */
  IE_IO_ERR_BIN_UNSPEC = -3, /**< Binary path unspecified and not inferable. */
  IE_IO_ERR_STAT = -4,       /**< stat(2) failure or invalid file metadata. */
  IE_IO_ERR_OPEN = -5,       /**< open(2) failure. */
  IE_IO_ERR_READ = -6,       /**< read(2) failure or short read. */
  IE_IO_ERR_ALLOC = -7,      /**< Memory allocation failure. */
  IE_IO_ERR_DECODE = -8      /**< Decode failure (corrupt or incompatible data). */
} ie_io_status_t;

/* ============================================================================
 * Weights (IEBIN v1)
 * ========================================================================== */

/**
 * @struct ie_weights_s
 * @brief Descriptor for an opened IEBIN v1 model.
 *
 * @details
 * This structure represents **resolved metadata**, not live tensor memory.
 * It is safe for:
 *  - stack allocation,
 *  - shallow copying,
 *  - deterministic cleanup via ie_weights_close().
 *
 * Optional deduplication artifacts are opened lazily and stored via an
 * **opaque handle**. The internal representation is intentionally hidden.
 */
typedef struct ie_weights_s {
  int    version;            /**< Parsed IEBIN version (>= 1). */
  char   dtype[16];          /**< Declared model dtype (e.g. "float32", "int4"). */

  char   json_path[512];     /**< Canonical path to model.ie.json. */
  char   weights_path[512];  /**< Resolved path to model.ie.bin. */

  size_t bin_size_bytes;     /**< Size of the weights binary in bytes. */
  int    loaded;             /**< Non-zero if open succeeded. */

  int    is_dedup;           /**< Non-zero if dedup artifacts were opened. */
  void  *dedup_handle;       /**< Opaque dedup handle (owned by this struct). */
} ie_weights_t;

/**
 * @brief Open and parse an IEBIN v1 model description.
 *
 * @details
 * On success:
 *  - `out->loaded` is set to non-zero,
 *  - JSON and BIN paths are resolved and stored,
 *  - basic metadata (dtype, version, size) is available for diagnostics.
 *
 * Optional deduplication artifacts are opened if enabled via environment flags.
 *
 * @param json_path Path to model.ie.json (must be readable).
 * @param bin_path  Optional override for model.ie.bin; may be NULL.
 * @param out       Output descriptor (written on success).
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_open(const char *json_path,
                    const char *bin_path,
                    ie_weights_t *out);

/**
 * @brief Touch the weights binary to verify readability and warm OS caches.
 *
 * @details
 * Implementations perform small positional reads near the start and end of the
 * file to:
 *  - detect permission or corruption issues early,
 *  - optionally fault pages into memory.
 *
 * When deduplication is enabled, this also verifies the presence of auxiliary
 * artifacts (defaults, exceptions, masks).
 *
 * @param w Opened weights descriptor.
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_touch(const ie_weights_t *w);

/**
 * @brief Close an opened weights descriptor.
 *
 * @details
 * This function releases any optional deduplication handles.
 * It is **idempotent** and safe to call on zero-initialized or partially
 * initialized descriptors.
 *
 * @param w Descriptor to close (may be NULL).
 */
void ie_weights_close(ie_weights_t *w);

/* ============================================================================
 * Lightweight tokenizer / vocabulary (test & harness only)
 * ========================================================================== */

/**
 * @struct ie_vocab_s
 * @brief Minimal vocabulary descriptor.
 *
 * @details
 * This structure exists solely for test and benchmark harness support.
 * It is **not** used by real inference paths (tokenizer_hf replaces it).
 */
typedef struct ie_vocab_s {
  int vocab_size; /**< Number of entries in the vocabulary. */
} ie_vocab_t;

/**
 * @brief Load a vocabulary file or fall back to a stub.
 *
 * @details
 * If @p vocab_path is NULL or unreadable, a deterministic stub vocabulary is
 * provided to keep tests and benchmarks operational.
 *
 * @param vocab_path Optional path to vocabulary file.
 * @param out        Output vocabulary descriptor.
 * @return 0 on success, negative on unrecoverable failure.
 */
int ie_vocab_load(const char *vocab_path, ie_vocab_t *out);

/**
 * @brief Encode UTF-8 text into token IDs (stub implementation).
 *
 * @details
 * This tokenizer:
 *  - splits on ASCII whitespace,
 *  - hashes tokens to stable integer IDs.
 *
 * It is **not** compatible with BPE or HF tokenizers and must never be used
 * for real inference.
 *
 * @param v         Vocabulary descriptor.
 * @param text      NUL-terminated UTF-8 input.
 * @param ids       Output buffer, or NULL for size-only query.
 * @param out_count In: capacity; Out: tokens written or required.
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
 * Produces a deterministic, human-readable representation for diagnostics.
 *
 * @param v      Vocabulary descriptor.
 * @param ids    Token ID array.
 * @param count  Number of token IDs.
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
 * @brief Release vocabulary resources.
 *
 * @details
 * Currently a no-op; present for API symmetry and future extensibility.
 *
 * @param v Vocabulary descriptor (may be NULL).
 */
void ie_vocab_free(ie_vocab_t *v);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_IO_H_ */
