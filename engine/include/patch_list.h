#ifndef IE_PATCH_LIST_H
#define IE_PATCH_LIST_H

/**
 * @file patch_list.h
 * @brief Patch-list representation and fast application for dedup reconstruction.
 *
 * This module provides an alternative to bitmask-based exceptions:
 * instead of storing a 1-bit-per-byte mask, it stores a compact list of
 * byte indices that differ from the default plus their corresponding bytes.
 *
 * Use cases:
 * - Extremely sparse exceptions (low exception density).
 * - Faster iteration than scanning a full mask for large tensors when
 *   popcount(mask) is small.
 *
 * Contract (lossless, byte-oriented):
 * - Indices refer to byte offsets in the tensor blob.
 * - Values are the exact original bytes at those offsets.
 * - Applying the patch list to a default blob produces the exact target blob.
 *
 * Storage format (binary, little-endian):
 *
 *   struct ie_patch_list_hdr {
 *     uint32_t magic;        // 'PLST'
 *     uint32_t version;      // 1
 *     uint64_t nbytes;       // tensor byte length
 *     uint64_t nitems;       // number of patched bytes
 *     // followed by:
 *     //   uint32_t idx[nitems];   (byte indices, strictly increasing)
 *     //   uint8_t  val[nitems];   (patched byte values)
 *   };
 *
 * Notes:
 * - idx is uint32_t to reduce size. If tensors exceed 4 GiB, switch to uint64_t.
 * - Strictly increasing indices allow validation and optional SIMD-friendly
 *   streaming writes.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Magic value for patch list headers: 'PLST'. */
#define IE_PATCH_LIST_MAGIC 0x54534C50u

/** @brief Patch list version supported by this module. */
#define IE_PATCH_LIST_VERSION 1u

/**
 * @struct ie_patch_list_t
 * @brief View over an in-memory patch list.
 *
 * This struct does not own memory. It points into a buffer that must remain valid.
 */
typedef struct ie_patch_list_s {
  /** @brief Tensor byte length that this patch list targets. */
  uint64_t nbytes;

  /** @brief Number of patched positions. */
  uint64_t nitems;

  /** @brief Pointer to byte indices array (strictly increasing). */
  const uint32_t* idx;

  /** @brief Pointer to byte values array (same length as idx). */
  const uint8_t* val;
} ie_patch_list_t;

/**
 * @brief Compute the required buffer size (in bytes) to store a patch list.
 *
 * @param nitems Number of patched bytes.
 * @return Total bytes for header + indices + values.
 */
size_t ie_patch_list_bytes(uint64_t nitems);

/**
 * @brief Initialize a patch list view from a serialized buffer.
 *
 * The buffer must begin with an ie_patch_list_hdr as described in the file comment.
 * This function validates the header and sets the ie_patch_list_t pointers.
 *
 * @param out Output patch list view.
 * @param buf Serialized buffer.
 * @param buf_len Buffer length in bytes.
 * @return 0 on success, non-zero on failure.
 */
int ie_patch_list_view_init(ie_patch_list_t* out, const void* buf, size_t buf_len);

/**
 * @brief Serialize a patch list into a caller-provided buffer.
 *
 * The caller provides idx and val arrays. Indices must be strictly increasing and
 * each must be < nbytes.
 *
 * @param dst Destination buffer.
 * @param dst_len Destination buffer capacity.
 * @param nbytes Tensor byte length.
 * @param nitems Number of patch items.
 * @param idx Byte indices array (strictly increasing).
 * @param val Byte values array.
 * @return 0 on success, non-zero on error (including insufficient dst_len).
 */
int ie_patch_list_serialize(
  void* dst,
  size_t dst_len,
  uint64_t nbytes,
  uint64_t nitems,
  const uint32_t* idx,
  const uint8_t* val);

/**
 * @brief Apply a patch list to an existing tensor buffer in-place.
 *
 * Typical usage:
 *   memcpy(dst, default_bytes, nbytes);
 *   ie_patch_list_apply_inplace(dst, nbytes, &pl);
 *
 * @param dst Destination buffer to modify (must have at least nbytes bytes).
 * @param nbytes Destination buffer size in bytes.
 * @param pl Patch list view.
 * @return 0 on success, non-zero on error (bounds, invalid indices).
 */
int ie_patch_list_apply_inplace(uint8_t* dst, uint64_t nbytes, const ie_patch_list_t* pl);

/**
 * @brief Convert a bitmask + exception stream into a patch list (indices + values).
 *
 * This is useful for hybrid formats:
 * - Authoring uses mask+exceptions (cheap to compute).
 * - Runtime uses patch list when exception density is low.
 *
 * Mask semantics:
 * - 1 bit per byte (LSB-first within each mask byte).
 * - exceptions stream provides the patched bytes in ascending byte index order.
 *
 * Output:
 * - idx_out receives indices (uint32_t per item).
 * - val_out receives values (uint8_t per item).
 *
 * The caller must allocate idx_out/val_out to hold exactly popcount(mask) items.
 *
 * @param nbytes Tensor byte length.
 * @param mask Pointer to mask bytes (ceil(nbytes/8)).
 * @param exceptions Pointer to exception bytes (popcount(mask)).
 * @param idx_out Output indices array.
 * @param val_out Output values array.
 * @param out_nitems Output number of items produced.
 * @return 0 on success, non-zero on error.
 */
int ie_patch_list_from_mask_exceptions(
  uint64_t nbytes,
  const uint8_t* mask,
  const uint8_t* exceptions,
  uint32_t* idx_out,
  uint8_t* val_out,
  uint64_t* out_nitems);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_PATCH_LIST_H */

