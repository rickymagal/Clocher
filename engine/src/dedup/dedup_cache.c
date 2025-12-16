/**
 * @file dedup_cache.c
 * @brief Implementation of lossless reconstruction and an optional fixed-capacity cache.
 */

#include "dedup_cache.h"

#include <stdlib.h>
#include <string.h>

/**
 * @brief Align a size up to a power-of-two boundary.
 * @param x Input value.
 * @param a Alignment (power of two).
 * @return x rounded up to a multiple of a.
 */
static size_t ie_align_up(size_t x, size_t a) {
  return (x + (a - 1u)) & ~(a - 1u);
}

void* ie_dedup_arena_alloc(ie_dedup_spec_arena_t* a, size_t nbytes, size_t align) {
  if (!a || !a->base || align == 0u) return NULL;
  size_t p = ie_align_up(a->len, align);
  if (p > a->cap) return NULL;
  if (nbytes > (a->cap - p)) return NULL;
  void* out = (uint8_t*)a->base + p;
  a->len = p + nbytes;
  return out;
}

int ie_dedup_spec_validate(const ie_dedup_spec_t* s) {
  if (!s) return -1;
  if (s->magic != IE_DEDUP_MAGIC) return -2;
  if (s->version != IE_DEDUP_VERSION) return -3;
  if (s->ngroups == 0 || !s->groups) return -4;

  uint64_t seen_targets = 0;
  for (uint32_t gi = 0; gi < s->ngroups; ++gi) {
    const ie_dedup_group_t* g = &s->groups[gi];
    if (g->default_nbytes == 0) return -5;
    if (g->ntargets == 0 || !g->targets) return -6;
    for (uint32_t ti = 0; ti < g->ntargets; ++ti) {
      const ie_dedup_target_t* t = &g->targets[ti];
      if (t->nbytes != g->default_nbytes) return -7;
      if (t->mask_nbytes != ie_dedup_mask_nbytes(t->nbytes)) return -8;
      seen_targets++;
    }
  }
  if (s->targets_flat_count != (uint32_t)seen_targets) return -9;
  return 0;
}

/**
 * @brief Count set bits in a byte using a precomputed nibble table.
 * @param b Input byte.
 * @return Popcount of b.
 */
static inline uint8_t ie_popcount_u8(uint8_t b) {
  static const uint8_t pc4[16] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
  };
  return (uint8_t)(pc4[b & 0x0Fu] + pc4[(b >> 4) & 0x0Fu]);
}

int ie_dedup_reconstruct_into(
  uint8_t* dst,
  const uint8_t* default_bytes,
  const uint8_t* mask,
  const uint8_t* exceptions,
  uint64_t nbytes)
{
  if (!dst || !default_bytes || !mask || (!exceptions && nbytes != 0)) return -1;
  if (nbytes == 0) return 0;

  /* Copy default into dst. */
  memcpy(dst, default_bytes, (size_t)nbytes);

  const uint64_t mask_n = ie_dedup_mask_nbytes(nbytes);

  /* Patch exceptions. */
  uint64_t exc_i = 0;
  uint64_t byte_i = 0;

  for (uint64_t mi = 0; mi < mask_n; ++mi) {
    const uint8_t m = mask[mi];
    if (m == 0) {
      byte_i += 8;
      continue;
    }

    /* Process the 8 bits in this mask byte. */
    for (uint32_t bit = 0; bit < 8; ++bit) {
      const uint64_t idx = byte_i + (uint64_t)bit;
      if (idx >= nbytes) break;
      if ((m >> bit) & 1u) {
        dst[idx] = exceptions[exc_i++];
      }
    }

    byte_i += 8;
  }

  /* Optional: caller can compare exc_i with expected exc_nbytes separately. */
  return 0;
}

/**
 * @brief Find the best eviction candidate (lowest stamp).
 * @param c Cache instance.
 * @return Entry index.
 */
static uint32_t ie_cache_find_evict(const ie_dedup_cache_t* c) {
  uint32_t best = 0;
  uint64_t best_stamp = UINT64_MAX;
  for (uint32_t i = 0; i < c->entry_cap; ++i) {
    const ie_dedup_cache_entry_t* e = &c->entries[i];
    if (!e->data) return i; /* empty slot */
    if (e->stamp < best_stamp) {
      best_stamp = e->stamp;
      best = i;
    }
  }
  return best;
}

/**
 * @brief Free a cache entry's owned buffer and reset it.
 * @param c Cache instance.
 * @param e Entry to clear.
 */
static void ie_cache_entry_clear(ie_dedup_cache_t* c, ie_dedup_cache_entry_t* e) {
  if (e->data) {
    free(e->data);
    if (c->bytes_cached >= e->nbytes) c->bytes_cached -= e->nbytes;
  }
  e->tensor_index = 0;
  e->nbytes = 0;
  e->data = NULL;
  e->stamp = 0;
}

int ie_dedup_cache_init(
  ie_dedup_cache_t* c,
  const ie_dedup_spec_t* spec,
  const ie_dedup_files_t* files,
  uint32_t max_tensor_index,
  uint32_t entry_cap,
  uint64_t bytes_limit)
{
  if (!c || !spec || !files) return -1;
  memset(c, 0, sizeof(*c));

  c->spec = spec;
  c->files = *files;

  if (ie_dedup_spec_validate(spec) != 0) return -2;

  /* Allocate target_by_index lookup. */
  c->target_by_index_len = max_tensor_index + 1u;
  c->target_by_index = (const ie_dedup_target_t**)calloc((size_t)c->target_by_index_len, sizeof(void*));
  if (!c->target_by_index) return -3;

  /* Build target lookup table. */
  for (uint32_t gi = 0; gi < spec->ngroups; ++gi) {
    const ie_dedup_group_t* g = &spec->groups[gi];
    for (uint32_t ti = 0; ti < g->ntargets; ++ti) {
      const ie_dedup_target_t* t = &g->targets[ti];
      if (t->tensor_index <= max_tensor_index) {
        c->target_by_index[t->tensor_index] = t;
      }
    }
  }

  /* Cache configuration. */
  c->bytes_limit = bytes_limit;
  c->entry_cap = (bytes_limit > 0 && entry_cap > 0) ? entry_cap : 0;

  if (c->entry_cap > 0) {
    c->entries = (ie_dedup_cache_entry_t*)calloc((size_t)c->entry_cap, sizeof(ie_dedup_cache_entry_t));
    if (!c->entries) {
      free(c->target_by_index);
      memset(c, 0, sizeof(*c));
      return -4;
    }
  }

  return 0;
}

void ie_dedup_cache_destroy(ie_dedup_cache_t* c) {
  if (!c) return;
  if (c->entries) {
    for (uint32_t i = 0; i < c->entry_cap; ++i) {
      ie_cache_entry_clear(c, &c->entries[i]);
    }
    free(c->entries);
  }
  free(c->target_by_index);
  memset(c, 0, sizeof(*c));
}

const ie_dedup_target_t* ie_dedup_cache_find_target(const ie_dedup_cache_t* c, uint32_t tensor_index) {
  if (!c || !c->target_by_index || tensor_index >= c->target_by_index_len) return NULL;
  return c->target_by_index[tensor_index];
}

const ie_dedup_group_t* ie_dedup_cache_find_group_for_target(const ie_dedup_cache_t* c, const ie_dedup_target_t* t) {
  if (!c || !c->spec || !t) return NULL;
  for (uint32_t gi = 0; gi < c->spec->ngroups; ++gi) {
    const ie_dedup_group_t* g = &c->spec->groups[gi];
    for (uint32_t ti = 0; ti < g->ntargets; ++ti) {
      if (&g->targets[ti] == t) return g;
    }
  }
  return NULL;
}

/**
 * @brief Compute the expected exception byte count from a mask.
 * @param mask Mask bytes.
 * @param mask_nbytes Mask size.
 * @return Popcount over all mask bytes.
 */
static uint64_t ie_mask_popcount_bytes(const uint8_t* mask, uint64_t mask_nbytes) {
  uint64_t pc = 0;
  for (uint64_t i = 0; i < mask_nbytes; ++i) pc += (uint64_t)ie_popcount_u8(mask[i]);
  return pc;
}

const uint8_t* ie_dedup_cache_get_bytes(
  ie_dedup_cache_t* c,
  uint32_t tensor_index,
  uint8_t* scratch,
  uint64_t scratch_cap,
  uint64_t* out_nbytes)
{
  if (out_nbytes) *out_nbytes = 0;
  if (!c || !c->spec) return NULL;

  const ie_dedup_target_t* t = ie_dedup_cache_find_target(c, tensor_index);
  if (!t) return NULL;

  const ie_dedup_group_t* g = ie_dedup_cache_find_group_for_target(c, t);
  if (!g) return NULL;

  const uint64_t nbytes = t->nbytes;

  /* Validate file slices are within bounds. */
  if (g->default_off + nbytes > c->files.defaults_size) return NULL;
  if (t->mask_off + t->mask_nbytes > c->files.masks_size) return NULL;
  if (t->exc_off + t->exc_nbytes > c->files.exceptions_size) return NULL;

  const uint8_t* defp = c->files.defaults + g->default_off;
  const uint8_t* maskp = c->files.masks + t->mask_off;
  const uint8_t* excp = c->files.exceptions + t->exc_off;

  /* Sanity check: popcount(mask) must match exc_nbytes. */
  {
    const uint64_t expected = ie_mask_popcount_bytes(maskp, t->mask_nbytes);
    if (expected != t->exc_nbytes) return NULL;
  }

  /* If caching disabled, reconstruct into scratch. */
  if (c->bytes_limit == 0 || c->entry_cap == 0) {
    if (!scratch || scratch_cap < nbytes) return NULL;
    if (ie_dedup_reconstruct_into(scratch, defp, maskp, excp, nbytes) != 0) return NULL;
    if (out_nbytes) *out_nbytes = nbytes;
    return scratch;
  }

  /* Check cache hit. */
  for (uint32_t i = 0; i < c->entry_cap; ++i) {
    ie_dedup_cache_entry_t* e = &c->entries[i];
    if (e->data && e->tensor_index == tensor_index && e->nbytes == nbytes) {
      e->stamp = ++c->stamp_now;
      if (out_nbytes) *out_nbytes = nbytes;
      return e->data;
    }
  }

  /* Cache miss: evict if needed, then allocate and reconstruct. */
  uint32_t slot = ie_cache_find_evict(c);
  ie_dedup_cache_entry_t* e = &c->entries[slot];

  /* Free entry if occupied. */
  if (e->data) ie_cache_entry_clear(c, e);

  /* Enforce soft byte limit by evicting more if required. */
  while (c->bytes_cached + nbytes > c->bytes_limit) {
    uint32_t victim = ie_cache_find_evict(c);
    if (!c->entries[victim].data) break;
    ie_cache_entry_clear(c, &c->entries[victim]);
  }

  e->data = (uint8_t*)malloc((size_t)nbytes);
  if (!e->data) {
    ie_cache_entry_clear(c, e);
    return NULL;
  }

  if (ie_dedup_reconstruct_into(e->data, defp, maskp, excp, nbytes) != 0) {
    ie_cache_entry_clear(c, e);
    return NULL;
  }

  e->tensor_index = tensor_index;
  e->nbytes = nbytes;
  e->stamp = ++c->stamp_now;
  c->bytes_cached += nbytes;

  if (out_nbytes) *out_nbytes = nbytes;
  return e->data;
}
