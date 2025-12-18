/* File: engine/include/ie_io.h
 * -----------------------------------------------------------------------------
 * @file ie_io.h
 * @brief Public I/O interfaces: IEBIN v1 loader and lightweight tokenizer.
 *
 * @details
 * Exposes:
 * - IEBIN v1 weights loader: open/touch/close of model.ie.json + model.ie.bin
 * - Lightweight tokenizer used by tests/harness
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
 */
typedef enum ie_io_status_e {
  IE_IO_OK = 0,
  IE_IO_ERR_ARGS = -1,
  IE_IO_ERR_JSON = -2,
  IE_IO_ERR_BIN_UNSPEC = -3,
  IE_IO_ERR_STAT = -4,
  IE_IO_ERR_OPEN = -5,
  IE_IO_ERR_READ = -6,
  IE_IO_ERR_ALLOC = -7,
  IE_IO_ERR_DECODE = -8
} ie_io_status_t;

/* ============================================================================
 * Weights (IEBIN v1)
 * ========================================================================== */

/**
 * @struct ie_weights_s
 * @brief In-memory descriptor for IEBIN v1 metadata and resolved paths.
 *
 * @details
 * This struct stores only fixed-size fields and an optional opaque dedup handle.
 * No heap ownership is exposed through the public surface.
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
 * @param json_path Path to model.ie.json (readable).
 * @param bin_path  Optional override for model.ie.bin; may be NULL.
 * @param out       Output descriptor (written on success).
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out);

/**
 * @brief Touch the weights binary to verify readability (optional warm-up).
 *
 * @param w Opened descriptor.
 * @return IE_IO_OK on success, negative ie_io_status_t otherwise.
 */
int ie_weights_touch(const ie_weights_t *w);

/**
 * @brief Close weights resources and release optional dedup handle.
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
 */
typedef struct ie_vocab_s {
  int vocab_size; /**< Number of entries. */
} ie_vocab_t;

/**
 * @brief Load a vocabulary from vocab_path or fall back to a stub.
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
 * Size-only mode: pass ids==NULL to compute required token count in *out_count.
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
 * @param v Vocabulary pointer (may be NULL).
 */
void ie_vocab_free(ie_vocab_t *v);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_IO_H_ */

