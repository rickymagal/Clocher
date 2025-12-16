/**
 * @file patch_list.c
 * @brief Patch list implementation: serialization, validation, and application.
 */

#include "patch_list.h"

#include <string.h>

/**
 * @struct ie_patch_list_hdr_t
 * @brief Serialized header for a patch list buffer.
 */
typedef struct ie_patch_list_hdr_s {
  uint32_t magic;
  uint32_t version;
  uint64_t nbytes;
  uint64_t nitems;
} ie_patch_list_hdr_t;

/**
 * @brief Read a little-endian 32-bit value from an unaligned pointer.
 * @param p Pointer to 4 bytes.
 * @return Parsed uint32_t.
 */
static uint32_t ie_load_u32(const uint8_t* p) {
  return (uint32_t)p[0]
       | ((uint32_t)p[1] << 8)
       | ((uint32_t)p[2] << 16)
       | ((uint32_t)p[3] << 24);
}

/**
 * @brief Read a little-endian 64-bit value from an unaligned pointer.
 * @param p Pointer to 8 bytes.
 * @return Parsed uint64_t.
 */
static uint64_t ie_load_u64(const uint8_t* p) {
  return (uint64_t)p[0]
       | ((uint64_t)p[1] << 8)
       | ((uint64_t)p[2] << 16)
       | ((uint64_t)p[3] << 24)
       | ((uint64_t)p[4] << 32)
       | ((uint64_t)p[5] << 40)
       | ((uint64_t)p[6] << 48)
       | ((uint64_t)p[7] << 56);
}

/**
 * @brief Store a little-endian 32-bit value to an unaligned pointer.
 * @param p Destination pointer to 4 bytes.
 * @param v Value to store.
 */
static void ie_store_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)((v >> 8) & 0xFFu);
  p[2] = (uint8_t)((v >> 16) & 0xFFu);
  p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

/**
 * @brief Store a little-endian 64-bit value to an unaligned pointer.
 * @param p Destination pointer to 8 bytes.
 * @param v Value to store.
 */
static void ie_store_u64(uint8_t* p, uint64_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)((v >> 8) & 0xFFu);
  p[2] = (uint8_t)((v >> 16) & 0xFFu);
  p[3] = (uint8_t)((v >> 24) & 0xFFu);
  p[4] = (uint8_t)((v >> 32) & 0xFFu);
  p[5] = (uint8_t)((v >> 40) & 0xFFu);
  p[6] = (uint8_t)((v >> 48) & 0xFFu);
  p[7] = (uint8_t)((v >> 56) & 0xFFu);
}

size_t ie_patch_list_bytes(uint64_t nitems) {
  const size_t hdr = sizeof(ie_patch_list_hdr_t);
  const size_t idx = (size_t)(nitems * (uint64_t)sizeof(uint32_t));
  const size_t val = (size_t)(nitems * (uint64_t)sizeof(uint8_t));
  return hdr + idx + val;
}

int ie_patch_list_view_init(ie_patch_list_t* out, const void* buf, size_t buf_len) {
  if (!out || !buf) return -1;
  if (buf_len < sizeof(ie_patch_list_hdr_t)) return -2;

  const uint8_t* p = (const uint8_t*)buf;

  const uint32_t magic = ie_load_u32(p + 0);
  const uint32_t version = ie_load_u32(p + 4);
  const uint64_t nbytes = ie_load_u64(p + 8);
  const uint64_t nitems = ie_load_u64(p + 16);

  if (magic != IE_PATCH_LIST_MAGIC) return -3;
  if (version != IE_PATCH_LIST_VERSION) return -4;

  const size_t need = ie_patch_list_bytes(nitems);
  if (buf_len < need) return -5;

  const uint8_t* idxp = p + sizeof(ie_patch_list_hdr_t);
  const uint8_t* valp = idxp + (size_t)(nitems * (uint64_t)sizeof(uint32_t));

  out->nbytes = nbytes;
  out->nitems = nitems;
  out->idx = (const uint32_t*)idxp;
  out->val = (const uint8_t*)valp;

  /* Validate strict monotonicity and bounds. */
  uint32_t prev = 0;
  for (uint64_t i = 0; i < nitems; ++i) {
    const uint32_t cur = out->idx[i];
    if ((uint64_t)cur >= nbytes) return -6;
    if (i > 0 && cur <= prev) return -7;
    prev = cur;
  }

  return 0;
}

int ie_patch_list_serialize(
  void* dst,
  size_t dst_len,
  uint64_t nbytes,
  uint64_t nitems,
  const uint32_t* idx,
  const uint8_t* val)
{
  if (!dst || (!idx && nitems) || (!val && nitems)) return -1;

  const size_t need = ie_patch_list_bytes(nitems);
  if (dst_len < need) return -2;

  /* Validate input indices. */
  uint32_t prev = 0;
  for (uint64_t i = 0; i < nitems; ++i) {
    const uint32_t cur = idx[i];
    if ((uint64_t)cur >= nbytes) return -3;
    if (i > 0 && cur <= prev) return -4;
    prev = cur;
  }

  uint8_t* p = (uint8_t*)dst;
  ie_store_u32(p + 0, IE_PATCH_LIST_MAGIC);
  ie_store_u32(p + 4, IE_PATCH_LIST_VERSION);
  ie_store_u64(p + 8, nbytes);
  ie_store_u64(p + 16, nitems);

  uint8_t* idxp = p + sizeof(ie_patch_list_hdr_t);
  uint8_t* valp = idxp + (size_t)(nitems * (uint64_t)sizeof(uint32_t));

  memcpy(idxp, idx, (size_t)(nitems * (uint64_t)sizeof(uint32_t)));
  memcpy(valp, val, (size_t)(nitems * (uint64_t)sizeof(uint8_t)));

  return 0;
}

int ie_patch_list_apply_inplace(uint8_t* dst, uint64_t nbytes, const ie_patch_list_t* pl) {
  if (!dst || !pl) return -1;
  if (pl->nbytes != nbytes) return -2;
  if (pl->nitems == 0) return 0;
  if (!pl->idx || !pl->val) return -3;

  for (uint64_t i = 0; i < pl->nitems; ++i) {
    const uint32_t idx = pl->idx[i];
    if ((uint64_t)idx >= nbytes) return -4;
    dst[idx] = pl->val[i];
  }

  return 0;
}

int ie_patch_list_from_mask_exceptions(
  uint64_t nbytes,
  const uint8_t* mask,
  const uint8_t* exceptions,
  uint32_t* idx_out,
  uint8_t* val_out,
  uint64_t* out_nitems)
{
  if (out_nitems) *out_nitems = 0;
  if (!mask || (!exceptions && nbytes != 0) || !idx_out || !val_out || !out_nitems) return -1;

  const uint64_t mask_n = (nbytes + 7u) / 8u;
  uint64_t exc_i = 0;
  uint64_t out_i = 0;
  uint64_t byte_base = 0;

  for (uint64_t mi = 0; mi < mask_n; ++mi) {
    const uint8_t m = mask[mi];
    if (m == 0) {
      byte_base += 8;
      continue;
    }

    for (uint32_t bit = 0; bit < 8; ++bit) {
      const uint64_t idx64 = byte_base + (uint64_t)bit;
      if (idx64 >= nbytes) break;
      if ((m >> bit) & 1u) {
        idx_out[out_i] = (uint32_t)idx64;
        val_out[out_i] = exceptions[exc_i];
        out_i++;
        exc_i++;
      }
    }

    byte_base += 8;
  }

  *out_nitems = out_i;
  return 0;
}
