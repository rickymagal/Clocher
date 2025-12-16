#ifndef IE_DEDUP_SPEC_H
#define IE_DEDUP_SPEC_H

/**
 * @file dedup_spec.h
 * @brief Lossless deduplication specification for "defaults + masks + exceptions".
 *
 * This header defines the in-memory representation of a deduplication specification and
 * the binary file layout used to store deduplicated weights losslessly.
 *
 * The design is intentionally byte-oriented:
 * - A "tensor" is treated as a raw byte blob (works for FP16, BF16, INT8, packed INT4, etc.).
 * - The mask has 1 bit per byte in the tensor blob.
 * - The exception payload stores the exact original bytes at masked positions in ascending byte index order.
 *
 * Reconstruction rule for a target tensor:
 *   target_bytes = default_bytes (same size) patched by (mask, exceptions).
 *
 * Files (relative to model directory):
 * - model.defaults.bin    concatenated default blobs (raw bytes)
 * - model.masks.bin       concatenated bitmasks (ceil(nbytes/8) bytes each)
 * - model.exceptions.bin  concatenated exception bytes (popcount(mask) bytes each)
 *
 * The JSON spec (model.dedup.json) links logical tensors to slices of those files.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Magic value for validating a parsed spec in memory ("DUDP"). */
#define IE_DEDUP_MAGIC 0x44554450u

/** @brief Spec version supported by this header. */
#define IE_DEDUP_VERSION 1u

/**
 * @struct ie_dedup_target_t
 * @brief One deduplicated target tensor entry.
 *
 * Each target tensor is reconstructed by taking a default blob and applying a sparse patch.
 * All offsets are byte offsets into their respective files.
 */
typedef struct ie_dedup_target_s {
  /** @brief Tensor index in the engine's tensor table (caller-defined mapping). */
  uint32_t tensor_index;

  /** @brief Tensor size in bytes. Must equal the group's default_nbytes. */
  uint64_t nbytes;

  /** @brief Offset into masks.bin where this target's mask starts. */
  uint64_t mask_off;

  /** @brief Size of this target's mask in bytes (ceil(nbytes/8)). */
  uint64_t mask_nbytes;

  /** @brief Offset into exceptions.bin where this target's exception bytes start. */
  uint64_t exc_off;

  /** @brief Size of this target's exception byte payload (popcount(mask)). */
  uint64_t exc_nbytes;
} ie_dedup_target_t;

/**
 * @struct ie_dedup_group_t
 * @brief A group of targets that share one default tensor blob.
 *
 * A group defines:
 * - one default blob slice in defaults.bin
 * - N target tensors each with their own (mask, exceptions) slices
 */
typedef struct ie_dedup_group_s {
  /** @brief Tensor index of the default reference tensor (caller-defined mapping). */
  uint32_t default_tensor_index;

  /** @brief Size of the default tensor blob in bytes. */
  uint64_t default_nbytes;

  /** @brief Offset into defaults.bin where this group's default blob starts. */
  uint64_t default_off;

  /** @brief Number of targets in this group. */
  uint32_t ntargets;

  /** @brief Pointer to the first target entry (owned by the spec). */
  const ie_dedup_target_t* targets;
} ie_dedup_group_t;

/**
 * @struct ie_dedup_spec_t
 * @brief Top-level parsed dedup spec.
 *
 * Ownership model:
 * - groups and targets_flat are owned by the spec parser / arena.
 * - pointers in groups point into targets_flat.
 *
 * The engine typically keeps one spec per model directory.
 */
typedef struct ie_dedup_spec_s {
  /** @brief Must be IE_DEDUP_MAGIC after successful parsing/validation. */
  uint32_t magic;

  /** @brief Must be IE_DEDUP_VERSION after successful parsing/validation. */
  uint32_t version;

  /** @brief Number of groups. */
  uint32_t ngroups;

  /** @brief Pointer to groups array (owned by the spec). */
  const ie_dedup_group_t* groups;

  /** @brief Flat storage of all targets (owned by the spec). */
  const ie_dedup_target_t* targets_flat;

  /** @brief Number of entries in targets_flat. */
  uint32_t targets_flat_count;

  /** @brief Total file sizes in bytes for sanity checks (optional but recommended). */
  uint64_t defaults_size;

  /** @brief Total masks.bin size in bytes for sanity checks. */
  uint64_t masks_size;

  /** @brief Total exceptions.bin size in bytes for sanity checks. */
  uint64_t exceptions_size;
} ie_dedup_spec_t;

/**
 * @struct ie_dedup_spec_arena_t
 * @brief Simple bump allocator used by a spec parser to avoid heap fragmentation.
 *
 * This is intentionally minimal: allocate from a caller-provided buffer.
 */
typedef struct ie_dedup_spec_arena_s {
  /** @brief Base address of the arena storage. */
  void* base;

  /** @brief Total arena capacity in bytes. */
  size_t cap;

  /** @brief Bytes already allocated (bump pointer). */
  size_t len;
} ie_dedup_spec_arena_t;

/**
 * @brief Initialize a spec arena using caller-provided storage.
 * @param a Arena instance.
 * @param mem Backing storage.
 * @param cap Capacity in bytes.
 */
static inline void ie_dedup_arena_init(ie_dedup_spec_arena_t* a, void* mem, size_t cap) {
  a->base = mem;
  a->cap = cap;
  a->len = 0;
}

/**
 * @brief Allocate aligned memory from a bump arena.
 *
 * @param a Arena instance.
 * @param nbytes Number of bytes requested.
 * @param align Alignment (power of two).
 * @return Pointer to allocated space, or NULL on out-of-memory.
 */
void* ie_dedup_arena_alloc(ie_dedup_spec_arena_t* a, size_t nbytes, size_t align);

/**
 * @brief Compute mask size in bytes for a tensor byte length.
 * @param nbytes Tensor size in bytes.
 * @return ceil(nbytes/8).
 */
static inline uint64_t ie_dedup_mask_nbytes(uint64_t nbytes) {
  return (nbytes + 7u) / 8u;
}

/**
 * @brief Validate basic invariants of a parsed spec.
 *
 * This checks only structural constraints; it does not require opening files.
 *
 * @param s Spec pointer.
 * @return 0 on success, non-zero on validation error.
 */
int ie_dedup_spec_validate(const ie_dedup_spec_t* s);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_DEDUP_SPEC_H */

