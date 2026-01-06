/**
 * @file dedup_cache.c
 * @brief Simple in-memory cache for reconstructed tensors (string-keyed).
 *
 * This module implements the public API declared in dedup_cache.h.
 * It is intentionally generic: it does not parse specs and does not
 * know how to reconstruct bytes. Callers materialize tensors elsewhere
 * and use this cache to retain the resulting contiguous buffers.
 */

#include "dedup_cache.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct ie_dedup_cache_entry {
  char* key;
  uint8_t* data;
  size_t size;
  uint64_t stamp;
} ie_dedup_cache_entry_t;

struct ie_dedup_cache {
  size_t bytes_limit;
  size_t bytes_used;
  int enable_safety_checks;

  uint64_t stamp_now;

  ie_dedup_cache_entry_t* entries;
  size_t entry_count;
  size_t entry_cap;
};

static void ie_entry_free(ie_dedup_cache_entry_t* e) {
  if (!e) return;
  free(e->key);
  free(e->data);
  e->key = NULL;
  e->data = NULL;
  e->size = 0;
  e->stamp = 0;
}

static size_t ie_find_entry(const ie_dedup_cache_t* c, const char* key, int* out_found) {
  if (out_found) *out_found = 0;
  if (!c || !key) return 0;

  for (size_t i = 0; i < c->entry_count; ++i) {
    const ie_dedup_cache_entry_t* e = &c->entries[i];
    if (e->key && strcmp(e->key, key) == 0) {
      if (out_found) *out_found = 1;
      return i;
    }
  }
  return 0;
}

static int ie_ensure_capacity(ie_dedup_cache_t* c, size_t want_cap) {
  if (!c) return -1;
  if (c->entry_cap >= want_cap) return 0;

  size_t new_cap = (c->entry_cap == 0) ? 16u : c->entry_cap;
  while (new_cap < want_cap) {
    if (new_cap > (SIZE_MAX / 2u)) return -2;
    new_cap *= 2u;
  }

  ie_dedup_cache_entry_t* p = (ie_dedup_cache_entry_t*)realloc(c->entries, new_cap * sizeof(*p));
  if (!p) return -3;

  for (size_t i = c->entry_cap; i < new_cap; ++i) {
    p[i].key = NULL;
    p[i].data = NULL;
    p[i].size = 0;
    p[i].stamp = 0;
  }

  c->entries = p;
  c->entry_cap = new_cap;
  return 0;
}

static size_t ie_find_lru_victim(const ie_dedup_cache_t* c) {
  size_t best_i = 0;
  uint64_t best_stamp = UINT64_MAX;

  for (size_t i = 0; i < c->entry_count; ++i) {
    const ie_dedup_cache_entry_t* e = &c->entries[i];
    if (e->data == NULL) return i;
    if (e->stamp < best_stamp) {
      best_stamp = e->stamp;
      best_i = i;
    }
  }
  return best_i;
}

static void ie_evict_index(ie_dedup_cache_t* c, size_t idx) {
  if (!c || idx >= c->entry_count) return;

  ie_dedup_cache_entry_t* e = &c->entries[idx];
  if (e->data) {
    if (c->bytes_used >= e->size) c->bytes_used -= e->size;
    else c->bytes_used = 0;
  }

  ie_entry_free(e);

  if (idx + 1u < c->entry_count) {
    c->entries[idx] = c->entries[c->entry_count - 1u];
    c->entries[c->entry_count - 1u].key = NULL;
    c->entries[c->entry_count - 1u].data = NULL;
    c->entries[c->entry_count - 1u].size = 0;
    c->entries[c->entry_count - 1u].stamp = 0;
  }
  c->entry_count--;
}

ie_dedup_cache_t* ie_dedup_cache_create(const ie_dedup_cache_opts_t* opts) {
  ie_dedup_cache_t* c = (ie_dedup_cache_t*)calloc(1, sizeof(*c));
  if (!c) return NULL;

  c->bytes_limit = 0;
  c->bytes_used = 0;
  c->enable_safety_checks = 1;
  c->stamp_now = 0;
  c->entries = NULL;
  c->entry_count = 0;
  c->entry_cap = 0;

  if (opts) {
    c->bytes_limit = opts->bytes_limit;
    c->enable_safety_checks = (opts->enable_safety_checks != 0);
  }

  return c;
}

void ie_dedup_cache_destroy(ie_dedup_cache_t* c) {
  if (!c) return;

  for (size_t i = 0; i < c->entry_count; ++i) {
    ie_entry_free(&c->entries[i]);
  }
  free(c->entries);

  free(c);
}

size_t ie_dedup_cache_bytes_used(const ie_dedup_cache_t* c) {
  return c ? c->bytes_used : 0;
}

size_t ie_dedup_cache_bytes_limit(const ie_dedup_cache_t* c) {
  return c ? c->bytes_limit : 0;
}

void ie_dedup_cache_clear(ie_dedup_cache_t* c) {
  if (!c) return;

  for (size_t i = 0; i < c->entry_count; ++i) {
    ie_entry_free(&c->entries[i]);
  }
  c->entry_count = 0;
  c->bytes_used = 0;
  c->stamp_now = 0;
}

int ie_dedup_cache_get(const ie_dedup_cache_t* c,
                       const char* key,
                       const void** out_ptr,
                       size_t* out_size)
{
  if (out_ptr) *out_ptr = NULL;
  if (out_size) *out_size = 0;

  if (!c || !key || !out_ptr || !out_size) return 0;
  if (c->bytes_limit == 0) return 0;

  int found = 0;
  size_t idx = ie_find_entry(c, key, &found);
  if (!found) return 0;

  const ie_dedup_cache_entry_t* e = &c->entries[idx];
  if (!e->data) return 0;

  *out_ptr = e->data;
  *out_size = e->size;

  return 1;
}

int ie_dedup_cache_put(ie_dedup_cache_t* c,
                       const char* key,
                       const void* data,
                       size_t size)
{
  if (!c || !key || (!data && size != 0)) return -1;

  if (c->bytes_limit == 0) {
    return 0;
  }

  if (c->enable_safety_checks) {
    if (size > c->bytes_limit) return -2;
  }

  uint8_t* new_data = NULL;
  if (size != 0) {
    new_data = (uint8_t*)malloc(size);
    if (!new_data) return -3;
    memcpy(new_data, data, size);
  }

  int found = 0;
  size_t idx = ie_find_entry(c, key, &found);

  if (found) {
    ie_dedup_cache_entry_t* e = &c->entries[idx];

    while (c->enable_safety_checks && (c->bytes_used - e->size + size) > c->bytes_limit) {
      size_t victim = ie_find_lru_victim(c);
      if (victim == idx) break;
      ie_evict_index(c, victim);
      if (idx >= c->entry_count) {
        idx = ie_find_entry(c, key, &found);
        if (!found) break;
      }
      e = &c->entries[idx];
    }

    if (c->enable_safety_checks && (c->bytes_used - e->size + size) > c->bytes_limit) {
      free(new_data);
      return -2;
    }

    if (c->bytes_used >= e->size) c->bytes_used -= e->size;
    else c->bytes_used = 0;

    free(e->data);
    e->data = new_data;
    e->size = size;
    e->stamp = ++c->stamp_now;

    c->bytes_used += size;
    return 0;
  }

  while (c->enable_safety_checks && (c->bytes_used + size) > c->bytes_limit) {
    if (c->entry_count == 0) break;
    size_t victim = ie_find_lru_victim(c);
    ie_evict_index(c, victim);
  }

  if (c->enable_safety_checks && (c->bytes_used + size) > c->bytes_limit) {
    free(new_data);
    return -2;
  }

  if (ie_ensure_capacity(c, c->entry_count + 1u) != 0) {
    free(new_data);
    return -3;
  }

  char* key_copy = (char*)malloc(strlen(key) + 1u);
  if (!key_copy) {
    free(new_data);
    return -3;
  }
  memcpy(key_copy, key, strlen(key) + 1u);

  ie_dedup_cache_entry_t* e = &c->entries[c->entry_count++];
  e->key = key_copy;
  e->data = new_data;
  e->size = size;
  e->stamp = ++c->stamp_now;

  c->bytes_used += size;
  return 0;
}
