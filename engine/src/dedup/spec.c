/**
 * @file spec.c
 * @brief Implementation of the dedup spec builder and helpers.
 */

#include "spec.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/**
 * @brief Builder internal representation.
 *
 * The builder keeps:
 *  - dynamic array of entries
 *  - a single growing string table (with a leading '\0' sentinel)
 *
 * This is intentionally simple: no hash table for string interning (Step 1 scope).
 * The extractor typically interns O(10^2) names, so linear lookup is acceptable.
 */
struct dedup_spec_builder_s {
  dedup_spec_entry_t* entries;
  uint64_t entry_count;
  uint64_t entry_cap;

  char* strings;
  uint64_t strings_len;
  uint64_t strings_cap;
};

/** @brief Ensure capacity for the entry array. */
static int ensure_entries_cap(dedup_spec_builder_t* b, uint64_t need) {
  if (need <= b->entry_cap) return 0;
  uint64_t new_cap = (b->entry_cap == 0) ? 64u : b->entry_cap;
  while (new_cap < need) new_cap *= 2u;
  dedup_spec_entry_t* p = (dedup_spec_entry_t*)realloc(b->entries, (size_t)new_cap * sizeof(dedup_spec_entry_t));
  if (!p) return -1;
  b->entries = p;
  b->entry_cap = new_cap;
  return 0;
}

/** @brief Ensure capacity for the string table. */
static int ensure_strings_cap(dedup_spec_builder_t* b, uint64_t need) {
  if (need <= b->strings_cap) return 0;
  uint64_t new_cap = (b->strings_cap == 0) ? 4096u : b->strings_cap;
  while (new_cap < need) new_cap *= 2u;
  char* p = (char*)realloc(b->strings, (size_t)new_cap);
  if (!p) return -1;
  b->strings = p;
  b->strings_cap = new_cap;
  return 0;
}

dedup_spec_builder_t* dedup_spec_builder_create(void) {
  dedup_spec_builder_t* b = (dedup_spec_builder_t*)calloc(1, sizeof(dedup_spec_builder_t));
  if (!b) return NULL;

  /* Reserve offset 0 as failure/sentinel by starting with a '\0'. */
  if (ensure_strings_cap(b, 1u) != 0) {
    free(b);
    return NULL;
  }
  b->strings[0] = '\0';
  b->strings_len = 1u;
  return b;
}

void dedup_spec_builder_destroy(dedup_spec_builder_t* b) {
  if (!b) return;
  free(b->entries);
  free(b->strings);
  free(b);
}

uint64_t dedup_spec_builder_intern_string(dedup_spec_builder_t* b, const char* s) {
  if (!b || !s) return 0;
  const uint64_t slen = (uint64_t)strlen(s) + 1u;

  /* Linear scan for identical string. */
  uint64_t off = 1u;
  while (off < b->strings_len) {
    const char* cur = b->strings + off;
    if (strcmp(cur, s) == 0) return off;
    off += (uint64_t)strlen(cur) + 1u;
  }

  const uint64_t need = b->strings_len + slen;
  if (ensure_strings_cap(b, need) != 0) return 0;

  const uint64_t out_off = b->strings_len;
  memcpy(b->strings + b->strings_len, s, (size_t)slen);
  b->strings_len += slen;
  return out_off;
}

int dedup_spec_builder_add_entry(dedup_spec_builder_t* b, const dedup_spec_entry_t* e) {
  if (!b || !e) return -1;
  if (ensure_entries_cap(b, b->entry_count + 1u) != 0) return -1;
  b->entries[b->entry_count++] = *e;
  return 0;
}

/** @brief Write bytes to a stream or fail. */
static int write_all(FILE* f, const void* p, size_t n) {
  if (n == 0) return 0;
  return (fwrite(p, 1, n, f) == n) ? 0 : -1;
}

int dedup_spec_builder_write_file(
  dedup_spec_builder_t* b,
  const char* path,
  uint64_t defaults_bytes,
  uint64_t exceptions_bytes,
  uint64_t masks_bytes
) {
  if (!b || !path) return -1;

  FILE* f = fopen(path, "wb");
  if (!f) return -1;

  dedup_spec_header_t hdr;
  memset(&hdr, 0, sizeof(hdr));
  hdr.magic[0] = (uint8_t)DEDUP_SPEC_MAGIC0;
  hdr.magic[1] = (uint8_t)DEDUP_SPEC_MAGIC1;
  hdr.magic[2] = (uint8_t)DEDUP_SPEC_MAGIC2;
  hdr.magic[3] = (uint8_t)DEDUP_SPEC_MAGIC3;
  hdr.magic[4] = (uint8_t)DEDUP_SPEC_MAGIC4;
  hdr.magic[5] = (uint8_t)DEDUP_SPEC_MAGIC5;
  hdr.magic[6] = (uint8_t)DEDUP_SPEC_MAGIC6;
  hdr.magic[7] = (uint8_t)DEDUP_SPEC_MAGIC7;
  hdr.version = DEDUP_SPEC_VERSION;
  hdr.entry_count = b->entry_count;

  hdr.entries_off = (uint64_t)sizeof(dedup_spec_header_t);
  hdr.strings_off = hdr.entries_off + b->entry_count * (uint64_t)sizeof(dedup_spec_entry_t);
  hdr.strings_bytes = b->strings_len;

  hdr.defaults_bytes = defaults_bytes;
  hdr.exceptions_bytes = exceptions_bytes;
  hdr.masks_bytes = masks_bytes;

  if (write_all(f, &hdr, sizeof(hdr)) != 0) { fclose(f); return -1; }
  if (b->entry_count) {
    if (write_all(f, b->entries, (size_t)b->entry_count * sizeof(dedup_spec_entry_t)) != 0) { fclose(f); return -1; }
  }
  if (b->strings_len) {
    if (write_all(f, b->strings, (size_t)b->strings_len) != 0) { fclose(f); return -1; }
  }

  fclose(f);
  return 0;
}

uint64_t dedup_fnv1a64(const void* data, size_t n) {
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

