/**
 * @file spec.h
 * @brief Lossless deduplication specification (binary) for INT4 weight blobs.
 *
 * This module defines a compact, mmap-friendly "spec" format describing how to
 * reconstruct original weight blobs from:
 *  - a shared "defaults" blob file (per group, stored once), and
 *  - per-tensor exception masks + exception data blocks (stored only for deltas).
 *
 * Design goals:
 *  - Zero third-party dependencies.
 *  - Deterministic, stable binary layout (little-endian).
 *  - Runtime-friendly: reconstruction can be a tight loop over fixed-size blocks.
 *  - Lossless: every byte of the original tensor payload is reproducible.
 *
 * The extractor (dedup_extract.c) produces:
 *  - dedup.spec.bin        (this spec format)
 *  - dedup.defaults.bin    (concatenated default blobs)
 *  - dedup.exceptions.bin  (concatenated exception blocks)
 *  - dedup.masks.bin       (concatenated bitset masks)
 *
 * A "tensor blob" here is an opaque byte array representing the packed weights for
 * a tensor (or sub-tensor) exactly as they appear on disk (e.g., q4 blocks, scales, etc.).
 */

#ifndef DEDUP_SPEC_H
#define DEDUP_SPEC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/** @brief Current on-disk spec version. */
#define DEDUP_SPEC_VERSION 1u

/** @brief File magic for dedup spec ("DEDSPEC\0"). */
#define DEDUP_SPEC_MAGIC0 0x44u /* 'D' */
#define DEDUP_SPEC_MAGIC1 0x45u /* 'E' */
#define DEDUP_SPEC_MAGIC2 0x44u /* 'D' */
#define DEDUP_SPEC_MAGIC3 0x53u /* 'S' */
#define DEDUP_SPEC_MAGIC4 0x50u /* 'P' */
#define DEDUP_SPEC_MAGIC5 0x45u /* 'E' */
#define DEDUP_SPEC_MAGIC6 0x43u /* 'C' */
#define DEDUP_SPEC_MAGIC7 0x00u /* '\0' */

/**
 * @brief Kind of payload described by a tensor entry.
 *
 * This is deliberately generic: "blob" can represent int4 blocks, int4 scales,
 * fp16 scales, or any other packed stream, as long as the extractor and runtime
 * agree on interpretation and block sizing.
 */
typedef enum dedup_payload_kind_e {
  DEDUP_PAYLOAD_UNKNOWN = 0,
  DEDUP_PAYLOAD_INT4_BLOCKS = 1,
  DEDUP_PAYLOAD_SCALES = 2,
  DEDUP_PAYLOAD_OTHER = 3
} dedup_payload_kind_t;

/**
 * @brief On-disk header for dedup.spec.bin.
 *
 * All offsets are file-relative offsets into the spec file itself.
 * All other streams (defaults/exceptions/masks) are separate files; their offsets
 * are relative to their own file starts.
 *
 * The spec file stores:
 *  - header
 *  - entry table [entry_count]
 *  - string table (NUL-terminated names)
 *
 * Each entry references its name via name_off (offset into string table).
 */
typedef struct dedup_spec_header_s {
  uint8_t  magic[8];        /**< Must match DEDUP_SPEC_MAGIC*. */
  uint32_t version;         /**< Must match DEDUP_SPEC_VERSION. */
  uint32_t reserved0;       /**< Reserved (must be 0). */

  uint64_t entry_count;     /**< Number of entries in the entry table. */
  uint64_t entries_off;     /**< Offset in this file to entry table. */
  uint64_t strings_off;     /**< Offset in this file to string table. */
  uint64_t strings_bytes;   /**< Total bytes of string table. */

  uint64_t defaults_bytes;  /**< Total bytes in defaults file (informational). */
  uint64_t exceptions_bytes;/**< Total bytes in exceptions file (informational). */
  uint64_t masks_bytes;     /**< Total bytes in masks file (informational). */

  uint64_t reserved1[8];    /**< Reserved (must be 0). */
} dedup_spec_header_t;

/**
 * @brief A single dedup mapping entry for one tensor blob.
 *
 * Reconstruction:
 *  - The blob is partitioned into fixed-size blocks: block_bytes.
 *  - For block i:
 *      if mask[i] == 0: copy block i from defaults at (defaults_off + i*block_bytes)
 *      else:            copy block i from exceptions at exceptions_off + ex_index(i)*block_bytes
 *
 * Where ex_index(i) is the number of set bits in mask[0..i-1].
 *
 * Masks are stored as a packed bitset in masks.bin:
 *  - bit i corresponds to block i
 *  - LSB-first within each byte (bit0 is block 0)
 *
 * Notes:
 *  - defaults_off points to a per-group default blob; multiple entries may share it.
 *  - original_* fields are informational and can be used for validation/debug.
 */
typedef struct dedup_spec_entry_s {
  uint64_t name_off;          /**< Offset into string table (NUL-terminated). */

  uint32_t payload_kind;      /**< dedup_payload_kind_t */
  uint32_t reserved0;         /**< Reserved (must be 0). */

  uint64_t group_id;          /**< Group identifier (same defaults blob). */

  uint64_t original_bytes;    /**< Total bytes in original blob. */
  uint64_t block_bytes;       /**< Fixed block size in bytes. Must divide original_bytes. */
  uint64_t block_count;       /**< original_bytes / block_bytes. */

  uint64_t defaults_off;      /**< Offset into defaults.bin for default blob. */
  uint64_t exceptions_off;    /**< Offset into exceptions.bin where exception blocks start. */
  uint64_t exceptions_blocks; /**< Number of exception blocks stored for this entry. */

  uint64_t mask_off;          /**< Offset into masks.bin for this entry's bitset. */
  uint64_t mask_bytes;        /**< Bytes of mask bitset: ceil(block_count/8). */

  uint64_t reserved1[4];      /**< Reserved (must be 0). */
} dedup_spec_entry_t;

/**
 * @brief Opaque builder used by the extractor to assemble and write spec files.
 */
typedef struct dedup_spec_builder_s dedup_spec_builder_t;

/**
 * @brief Create a new spec builder.
 *
 * @return Pointer to a new builder, or NULL on allocation failure.
 */
dedup_spec_builder_t* dedup_spec_builder_create(void);

/**
 * @brief Destroy a spec builder and release memory.
 *
 * @param b Builder pointer (may be NULL).
 */
void dedup_spec_builder_destroy(dedup_spec_builder_t* b);

/**
 * @brief Add (or reuse) a string in the builder's string table.
 *
 * @param b Builder.
 * @param s NUL-terminated string.
 * @return Offset into the eventual string table (name_off), or 0 on failure.
 *
 * @note Offset 0 is valid if the string table starts at offset 0; however this builder
 *       uses a leading '\0' sentinel so offset 0 is reserved, and failures return 0.
 */
uint64_t dedup_spec_builder_intern_string(dedup_spec_builder_t* b, const char* s);

/**
 * @brief Append a spec entry.
 *
 * @param b Builder.
 * @param e Entry to append (copied by value).
 * @return 0 on success, non-zero on failure.
 */
int dedup_spec_builder_add_entry(dedup_spec_builder_t* b, const dedup_spec_entry_t* e);

/**
 * @brief Write the spec binary file.
 *
 * @param b Builder.
 * @param path Output path (e.g., "dedup.spec.bin").
 * @param defaults_bytes Total bytes written to defaults.bin (for header info).
 * @param exceptions_bytes Total bytes written to exceptions.bin (for header info).
 * @param masks_bytes Total bytes written to masks.bin (for header info).
 * @return 0 on success, non-zero on error.
 */
int dedup_spec_builder_write_file(
  dedup_spec_builder_t* b,
  const char* path,
  uint64_t defaults_bytes,
  uint64_t exceptions_bytes,
  uint64_t masks_bytes
);

/**
 * @brief Compute a 64-bit FNV-1a hash for bytes (helper).
 *
 * @param data Byte pointer.
 * @param n Bytes length.
 * @return 64-bit hash.
 */
uint64_t dedup_fnv1a64(const void* data, size_t n);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DEDUP_SPEC_H */

