/* ==========================================================================
 * File: engine/src/io/tokenizer_gptoss.c
 * ==========================================================================
 */
/**
 * @file tokenizer_gptoss.c
 * @brief GPT-OSS tokenizer supporting:
 *  - HuggingFace `tokenizer.json` (GPT-2 byte-level BPE)
 *  - Packed `IETOK1` (recommended)
 *  - OpenAI-style `.tiktoken` ranks (base64 token bytes + rank)
 *
 * Notes:
 *  - Correctness is prioritized over speed.
 *  - `.tiktoken` tokens are binary-safe and stored as raw bytes.
 *  - `.tiktoken` encode uses byte-level BPE per conservative segments:
 *    runs of whitespace and runs of non-whitespace.
 */

#include "ie_tokenizer_gptoss.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Small utilities
 * ========================================================================== */

static uint64_t fnv1a64(const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

static uint64_t fnv1a64_two(const void *a, size_t an, const void *b, size_t bn) {
  const unsigned char *p = (const unsigned char *)a;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < an; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  p = (const unsigned char *)b;
  for (size_t i = 0; i < bn; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

static uint32_t fnv1a32(const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;
  }
  return h;
}

static void *xcalloc(size_t n, size_t sz) {
  if (n == 0 || sz == 0) return NULL;
  if (n > (SIZE_MAX / sz)) return NULL;
  return calloc(n, sz);
}

static void *xrealloc(void *p, size_t n) {
  if (n == 0) return NULL;
  return realloc(p, n);
}

static char *xstrdup_n(const char *s, size_t n) {
  char *p = (char *)malloc(n + 1);
  if (!p) return NULL;
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

static int read_all_bytes(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf || !out_len) return -1;
  *out_buf = NULL;
  *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return -1; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return -1; }

  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if (rd != (size_t)sz) { free(buf); return -1; }

  buf[sz] = '\0';
  *out_buf = buf;
  *out_len = (size_t)sz;
  return 0;
}

static int path_ends_with(const char *path, const char *suf) {
  if (!path || !suf) return 0;
  size_t a = strlen(path);
  size_t b = strlen(suf);
  if (b > a) return 0;
  return memcmp(path + (a - b), suf, b) == 0;
}

/* ============================================================================
 * Minimal UTF-8 decode/encode
 * ========================================================================== */

static int utf8_next_cp(const char *s, size_t n, size_t *io_i, uint32_t *out_cp) {
  if (!s || !io_i || !out_cp) return -1;
  size_t i = *io_i;
  if (i >= n) return 1;

  unsigned char c0 = (unsigned char)s[i++];
  if (c0 < 0x80) { *out_cp = (uint32_t)c0; *io_i = i; return 0; }

  int need = 0;
  uint32_t cp = 0;
  if ((c0 & 0xE0) == 0xC0) { need = 1; cp = (uint32_t)(c0 & 0x1F); }
  else if ((c0 & 0xF0) == 0xE0) { need = 2; cp = (uint32_t)(c0 & 0x0F); }
  else if ((c0 & 0xF8) == 0xF0) { need = 3; cp = (uint32_t)(c0 & 0x07); }
  else return -1;

  if (i + (size_t)need > n) return -1;
  for (int k = 0; k < need; ++k) {
    unsigned char cx = (unsigned char)s[i++];
    if ((cx & 0xC0) != 0x80) return -1;
    cp = (cp << 6) | (uint32_t)(cx & 0x3F);
  }

  *out_cp = cp;
  *io_i = i;
  return 0;
}

static size_t utf8_put_cp(char *dst, size_t cap, uint32_t cp) {
  if (!dst || cap == 0) return 0;
  if (cp < 0x80) {
    if (cap < 1) return 0;
    dst[0] = (char)cp;
    return 1;
  } else if (cp < 0x800) {
    if (cap < 2) return 0;
    dst[0] = (char)(0xC0 | (cp >> 6));
    dst[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  } else if (cp < 0x10000) {
    if (cap < 3) return 0;
    dst[0] = (char)(0xE0 | (cp >> 12));
    dst[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    dst[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  } else {
    if (cap < 4) return 0;
    dst[0] = (char)(0xF0 | (cp >> 18));
    dst[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    dst[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    dst[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
  }
}

/* ============================================================================
 * JSON scanner tailored for tokenizer.json
 * ========================================================================== */

static const char *skip_ws(const char *p) {
  while (*p && (unsigned char)*p <= 0x20) ++p;
  return p;
}

static int scan_json_string(const char **pp, char **out_s, size_t *out_n) {
  const char *p = skip_ws(*pp);
  if (*p != '"') return -1;
  ++p;

  const char *start = p;
  while (*p) {
    if (*p == '\\') { ++p; if (!*p) return -1; ++p; continue; }
    if (*p == '"') break;
    ++p;
  }
  if (*p != '"') return -1;

  const char *end = p;
  ++p;

  char *raw = xstrdup_n(start, (size_t)(end - start));
  if (!raw) return -1;

  *out_s = raw;
  if (out_n) *out_n = (size_t)(end - start);
  *pp = p;
  return 0;
}

static int json_unescape_inplace(char *s) {
  if (!s) return -1;
  char *w = s;
  for (char *r = s; *r; ++r) {
    if (*r != '\\') { *w++ = *r; continue; }
    ++r;
    if (!*r) return -1;
    switch (*r) {
      case '"': *w++ = '"'; break;
      case '\\': *w++ = '\\'; break;
      case '/': *w++ = '/'; break;
      case 'b': *w++ = '\b'; break;
      case 'f': *w++ = '\f'; break;
      case 'n': *w++ = '\n'; break;
      case 'r': *w++ = '\r'; break;
      case 't': *w++ = '\t'; break;
      case 'u': {
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
          ++r;
          char c = *r;
          if (!c) return -1;
          v <<= 4;
          if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
          else if (c >= 'a' && c <= 'f') v |= (uint32_t)(10 + (c - 'a'));
          else if (c >= 'A' && c <= 'F') v |= (uint32_t)(10 + (c - 'A'));
          else return -1;
        }
        char tmp[4];
        size_t n = utf8_put_cp(tmp, sizeof(tmp), v);
        if (n == 0) return -1;
        for (size_t i = 0; i < n; ++i) *w++ = tmp[i];
      } break;
      default:
        return -1;
    }
  }
  *w = '\0';
  return 0;
}

static int scan_json_int(const char **pp, int *out_val) {
  const char *p = skip_ws(*pp);
  int neg = 0;
  if (*p == '-') { neg = 1; ++p; }
  if (!isdigit((unsigned char)*p)) return -1;
  long v = 0;
  while (isdigit((unsigned char)*p)) {
    v = v * 10 + (long)(*p - '0');
    if (v > INT_MAX) return -1;
    ++p;
  }
  *out_val = neg ? -(int)v : (int)v;
  *pp = p;
  return 0;
}

static const char *find_key_in_object(const char *json, const char *key) {
  if (!json || !key) return NULL;
  size_t klen = strlen(key);

  const char *p = json;
  while ((p = strstr(p, "\"")) != NULL) {
    ++p;
    if (strncmp(p, key, klen) == 0 && p[klen] == '"') {
      const char *q = p + klen + 1;
      q = skip_ws(q);
      if (*q == ':') return q + 1;
    }
    ++p;
  }
  return NULL;
}

static int skip_json_value(const char **pp) {
  const char *p = skip_ws(*pp);
  if (!p || *p == '\0') return -1;

  if (*p == '"') {
    char *tmp = NULL;
    if (scan_json_string(&p, &tmp, NULL) != 0) return -1;
    free(tmp);
    *pp = p;
    return 0;
  }

  if (*p == '{') {
    int depth = 0, in_str = 0, esc = 0;
    for (; *p; p++) {
      char c = *p;
      if (in_str) {
        if (esc) esc = 0;
        else if (c == '\\') esc = 1;
        else if (c == '"') in_str = 0;
        continue;
      }
      if (c == '"') { in_str = 1; continue; }
      if (c == '{') depth++;
      else if (c == '}') { depth--; if (depth == 0) { p++; *pp = p; return 0; } }
    }
    return -1;
  }

  if (*p == '[') {
    int depth = 0, in_str = 0, esc = 0;
    for (; *p; p++) {
      char c = *p;
      if (in_str) {
        if (esc) esc = 0;
        else if (c == '\\') esc = 1;
        else if (c == '"') in_str = 0;
        continue;
      }
      if (c == '"') { in_str = 1; continue; }
      if (c == '[') depth++;
      else if (c == ']') { depth--; if (depth == 0) { p++; *pp = p; return 0; } }
    }
    return -1;
  }

  if (*p == '-' || (*p >= '0' && *p <= '9')) {
    p++;
    while (*p) {
      char c = *p;
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { p++; continue; }
      break;
    }
    *pp = p;
    return 0;
  }

  if (strncmp(p, "true", 4) == 0) { *pp = p + 4; return 0; }
  if (strncmp(p, "false", 5) == 0) { *pp = p + 5; return 0; }
  if (strncmp(p, "null", 4) == 0) { *pp = p + 4; return 0; }

  return -1;
}

/* ============================================================================
 * Hash maps: string->id and pair->rank (GPT-2 JSON backend)
 * ========================================================================== */

typedef struct strid_ent_s {
  uint64_t h;
  const char *k;
  uint32_t v;
  int used;
} strid_ent_t;

typedef struct pairrank_ent_s {
  uint64_t key;
  uint32_t rank;
  int used;
} pairrank_ent_t;

static size_t next_pow2(size_t x) {
  size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

static int strid_map_init(strid_ent_t **out, size_t *out_cap, size_t want) {
  size_t cap = next_pow2(want * 2 + 64);
  strid_ent_t *m = (strid_ent_t *)xcalloc(cap, sizeof(*m));
  if (!m) return -1;
  *out = m;
  *out_cap = cap;
  return 0;
}

static int strid_map_put(strid_ent_t *m, size_t cap, const char *k, uint32_t v) {
  if (!m || cap == 0 || !k) return -1;
  uint64_t h = fnv1a64(k, strlen(k));
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) {
      m[i].used = 1;
      m[i].h = h;
      m[i].k = k;
      m[i].v = v;
      return 0;
    }
    if (m[i].h == h && strcmp(m[i].k, k) == 0) { m[i].v = v; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

static int strid_map_get(const strid_ent_t *m, size_t cap, const char *k, uint32_t *out_v) {
  if (!m || cap == 0 || !k || !out_v) return -1;
  uint64_t h = fnv1a64(k, strlen(k));
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].h == h && strcmp(m[i].k, k) == 0) { *out_v = m[i].v; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

static int pairrank_map_init(pairrank_ent_t **out, size_t *out_cap, size_t want) {
  size_t cap = next_pow2(want * 2 + 64);
  pairrank_ent_t *m = (pairrank_ent_t *)xcalloc(cap, sizeof(*m));
  if (!m) return -1;
  *out = m;
  *out_cap = cap;
  return 0;
}

static int pairrank_map_put(pairrank_ent_t *m, size_t cap, uint64_t key, uint32_t rank) {
  if (!m || cap == 0) return -1;
  size_t mask = cap - 1;
  size_t i = (size_t)key & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) { m[i].used = 1; m[i].key = key; m[i].rank = rank; return 0; }
    if (m[i].key == key) { m[i].rank = rank; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

static int pairrank_map_get(const pairrank_ent_t *m, size_t cap, uint64_t key, uint32_t *out_rank) {
  if (!m || cap == 0 || !out_rank) return -1;
  size_t mask = cap - 1;
  size_t i = (size_t)key & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].key == key) { *out_rank = m[i].rank; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

/* ============================================================================
 * Binary hash map: (bytes,len)->id for .tiktoken backend
 * ========================================================================== */

typedef struct bytesid_ent_s {
  uint64_t h;
  const uint8_t *p;
  uint32_t n;
  uint32_t v;
  int used;
} bytesid_ent_t;

static int bytesid_map_init(bytesid_ent_t **out, size_t *out_cap, size_t want) {
  size_t cap = next_pow2(want * 2 + 64);
  bytesid_ent_t *m = (bytesid_ent_t *)xcalloc(cap, sizeof(*m));
  if (!m) return -1;
  *out = m;
  *out_cap = cap;
  return 0;
}

static int bytesid_map_put(bytesid_ent_t *m, size_t cap, const uint8_t *p, uint32_t n, uint32_t v) {
  if (!m || cap == 0 || (!p && n)) return -1;
  uint64_t h = fnv1a64(p, (size_t)n);
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) {
      m[i].used = 1;
      m[i].h = h;
      m[i].p = p;
      m[i].n = n;
      m[i].v = v;
      return 0;
    }
    if (m[i].h == h && m[i].n == n && (n == 0 || memcmp(m[i].p, p, n) == 0)) {
      m[i].p = p;
      m[i].v = v;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

static int bytesid_map_get(const bytesid_ent_t *m, size_t cap, const uint8_t *p, uint32_t n, uint32_t *out_v) {
  if (!m || cap == 0 || (!p && n) || !out_v) return -1;
  uint64_t h = fnv1a64(p, (size_t)n);
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].h == h && m[i].n == n && (n == 0 || memcmp(m[i].p, p, n) == 0)) { *out_v = m[i].v; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

static int bytesid_map_get_concat(const bytesid_ent_t *m, size_t cap,
                                  const uint8_t *a, uint32_t an,
                                  const uint8_t *b, uint32_t bn,
                                  uint32_t *out_v) {
  if (!m || cap == 0 || (!a && an) || (!b && bn) || !out_v) return -1;
  uint32_t n = an + bn;
  uint64_t h = fnv1a64_two(a, (size_t)an, b, (size_t)bn);
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].h == h && m[i].n == n) {
      if (n == 0) { *out_v = m[i].v; return 0; }
      if (an == 0) {
        if (memcmp(m[i].p, b, bn) == 0) { *out_v = m[i].v; return 0; }
      } else if (bn == 0) {
        if (memcmp(m[i].p, a, an) == 0) { *out_v = m[i].v; return 0; }
      } else {
        if (memcmp(m[i].p, a, an) == 0 && memcmp(m[i].p + an, b, bn) == 0) { *out_v = m[i].v; return 0; }
      }
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/* ============================================================================
 * String interning for GPT-2 merges
 * ========================================================================== */

typedef struct intern_ent_s {
  uint32_t h;
  const char *s;
  uint32_t id;
  int used;
} intern_ent_t;

typedef struct intern_s {
  intern_ent_t *m;
  size_t cap;
  char *arena;
  size_t arena_sz;
  size_t arena_used;
  uint32_t next_id;
} intern_t;

static int intern_init(intern_t *in, size_t want_syms, size_t arena_hint) {
  if (!in) return -1;
  memset(in, 0, sizeof(*in));
  in->cap = next_pow2(want_syms * 2 + 64);
  in->m = (intern_ent_t *)xcalloc(in->cap, sizeof(*in->m));
  if (!in->m) return -1;
  in->arena_sz = (arena_hint ? arena_hint : (1u << 20));
  in->arena = (char *)malloc(in->arena_sz);
  if (!in->arena) { free(in->m); memset(in, 0, sizeof(*in)); return -1; }
  in->arena_used = 0;
  in->next_id = 1;
  return 0;
}

static void intern_free(intern_t *in) {
  if (!in) return;
  free(in->m);
  free(in->arena);
  memset(in, 0, sizeof(*in));
}

static const char *intern_store(intern_t *in, const char *s, size_t n) {
  if (!in || !s) return NULL;
  if (n + 1 > in->arena_sz - in->arena_used) {
    size_t new_sz = in->arena_sz ? in->arena_sz * 2 : (1u << 20);
    while (new_sz < in->arena_used + n + 1) new_sz *= 2;
    char *na = (char *)realloc(in->arena, new_sz);
    if (!na) return NULL;
    in->arena = na;
    in->arena_sz = new_sz;
  }
  char *dst = in->arena + in->arena_used;
  memcpy(dst, s, n);
  dst[n] = '\0';
  in->arena_used += n + 1;
  return dst;
}

static int intern_get(const intern_t *in, const char *s, uint32_t *out_id) {
  if (!in || !s || !out_id || in->cap == 0) return -1;
  uint32_t h = fnv1a32(s, strlen(s));
  size_t mask = in->cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < in->cap; ++step) {
    if (!in->m[i].used) return -1;
    if (in->m[i].h == h && strcmp(in->m[i].s, s) == 0) { *out_id = in->m[i].id; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

static int intern_get_or_add(intern_t *in, const char *s, uint32_t *out_id) {
  if (!in || !s || !out_id) return -1;
  uint32_t h = fnv1a32(s, strlen(s));
  size_t mask = in->cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < in->cap; ++step) {
    if (!in->m[i].used) {
      const char *st = intern_store(in, s, strlen(s));
      if (!st) return -1;
      uint32_t id = in->next_id++;
      in->m[i].used = 1;
      in->m[i].h = h;
      in->m[i].s = st;
      in->m[i].id = id;
      *out_id = id;
      return 0;
    }
    if (in->m[i].h == h && strcmp(in->m[i].s, s) == 0) { *out_id = in->m[i].id; return 0; }
    i = (i + 1) & mask;
  }
  return -1;
}

/* ============================================================================
 * GPT-2 bytes_to_unicode mapping (byte-level BPE)
 * ========================================================================== */

typedef struct byteunicode_s {
  uint32_t byte_to_cp[256];
  int32_t cp_to_byte[2048];
  uint32_t cp_to_byte_cap;
} byteunicode_t;

static void byteunicode_init(byteunicode_t *m) {
  memset(m, 0, sizeof(*m));
  for (size_t i = 0; i < 2048; ++i) m->cp_to_byte[i] = -1;
  m->cp_to_byte_cap = 2048;

  int used[256];
  for (int i = 0; i < 256; ++i) used[i] = 0;

  for (int b = 33; b <= 126; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }
  for (int b = 161; b <= 172; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }
  for (int b = 174; b <= 255; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }

  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (!used[b]) { m->byte_to_cp[b] = (uint32_t)(256 + extra); extra++; }
  }

  for (int b = 0; b < 256; ++b) {
    uint32_t cp = m->byte_to_cp[b];
    if (cp < m->cp_to_byte_cap) m->cp_to_byte[cp] = b;
  }
}

static int byteunicode_cp_to_byte(const byteunicode_t *m, uint32_t cp, uint8_t *out_b) {
  if (!m || !out_b) return -1;
  if (cp < m->cp_to_byte_cap && m->cp_to_byte[cp] >= 0) { *out_b = (uint8_t)m->cp_to_byte[cp]; return 0; }
  return -1;
}

/* ============================================================================
 * Buffers
 * ========================================================================== */

typedef struct u32buf_s {
  uint32_t *v;
  size_t n;
  size_t cap;
} u32buf_t;

static int u32buf_push(u32buf_t *b, uint32_t x) {
  if (!b) return -1;
  if (b->n == b->cap) {
    size_t nc = (b->cap ? b->cap * 2 : 64);
    uint32_t *nv = (uint32_t *)xrealloc(b->v, nc * sizeof(uint32_t));
    if (!nv) return -1;
    b->v = nv;
    b->cap = nc;
  }
  b->v[b->n++] = x;
  return 0;
}

typedef struct strbuf_s {
  char *v;
  size_t n;
  size_t cap;
} strbuf_t;

static int strbuf_reserve(strbuf_t *b, size_t add) {
  if (!b) return -1;
  if (b->n + add <= b->cap) return 0;
  size_t nc = (b->cap ? b->cap * 2 : 256);
  while (nc < b->n + add) nc *= 2;
  char *nv = (char *)xrealloc(b->v, nc);
  if (!nv) return -1;
  b->v = nv;
  b->cap = nc;
  return 0;
}

static int strbuf_append_bytes(strbuf_t *b, const char *s, size_t n) {
  if (!b || (!s && n)) return -1;
  if (strbuf_reserve(b, n) != 0) return -1;
  memcpy(b->v + b->n, s, n);
  b->n += n;
  return 0;
}

static void strbuf_free(strbuf_t *b) {
  if (!b) return;
  free(b->v);
  memset(b, 0, sizeof(*b));
}

/* ============================================================================
 * Byte arenas (stable pointers)
 * ========================================================================== */

typedef struct bytearena_s {
  uint8_t **blocks;
  size_t *sizes;
  size_t nblocks;
  size_t cap;
  uint8_t *cur;
  size_t cur_sz;
  size_t cur_used;
} bytearena_t;

static void bytearena_free(bytearena_t *a) {
  if (!a) return;
  if (a->blocks) {
    for (size_t i = 0; i < a->nblocks; ++i) free(a->blocks[i]);
  }
  free(a->blocks);
  free(a->sizes);
  memset(a, 0, sizeof(*a));
}

static int bytearena_new_block(bytearena_t *a, size_t need) {
  size_t blk = (a->cur_sz ? a->cur_sz * 2 : (1u << 20));
  if (blk < need) blk = need;
  uint8_t *mem = (uint8_t *)malloc(blk);
  if (!mem) return -1;

  if (a->nblocks == a->cap) {
    size_t nc = a->cap ? a->cap * 2 : 8;
    uint8_t **nb = (uint8_t **)realloc(a->blocks, nc * sizeof(*nb));
    size_t *ns = (size_t *)realloc(a->sizes, nc * sizeof(*ns));
    if (!nb || !ns) { free(mem); free(nb); free(ns); return -1; }
    a->blocks = nb;
    a->sizes = ns;
    a->cap = nc;
  }

  a->blocks[a->nblocks] = mem;
  a->sizes[a->nblocks] = blk;
  a->nblocks++;

  a->cur = mem;
  a->cur_sz = blk;
  a->cur_used = 0;
  return 0;
}

static uint8_t *bytearena_alloc(bytearena_t *a, size_t n) {
  if (!a) return NULL;
  if (n == 0) return (uint8_t *)"";
  if (!a->cur || a->cur_used + n > a->cur_sz) {
    if (bytearena_new_block(a, n) != 0) return NULL;
  }
  uint8_t *p = a->cur + a->cur_used;
  a->cur_used += n;
  return p;
}

/* ============================================================================
 * Tokenizer state
 * ========================================================================== */

typedef enum ie_tok_mode_e {
  IE_TOK_MODE_GPT2 = 0,
  IE_TOK_MODE_TIKTOKEN = 1
} ie_tok_mode_t;

typedef struct bytes_view_s {
  const uint8_t *p;
  uint32_t n;
} bytes_view_t;

struct ie_tok_gptoss_s {
  ie_tok_mode_t mode;
  uint32_t vocab_size;

  /* GPT-2 JSON / packed backend */
  char **id_to_tok;
  strid_ent_t *tok_to_id;
  size_t tok_to_id_cap;
  pairrank_ent_t *pair_to_rank;
  size_t pair_to_rank_cap;
  intern_t intern;
  byteunicode_t bu;

  /* .tiktoken backend */
  bytes_view_t *id_to_bytes;
  bytesid_ent_t *bytes_to_id;
  size_t bytes_to_id_cap;
  bytearena_t tt_arena;
  uint32_t tt_byte_id[256];
  int tt_have_byte_id;
};

/* ============================================================================
 * tokenizer.json parsing (GPT-2 backend)
 * ========================================================================== */

static int parse_vocab_object(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "vocab");
  if (!p) return IE_TOK_GPTOSS_ERR_JSON;

  p = skip_ws(p);
  if (*p != '{') return IE_TOK_GPTOSS_ERR_JSON;
  ++p;

  uint32_t max_id = 0;
  const char *q = p;
  while (*q) {
    q = skip_ws(q);
    if (*q == '}') break;
    char *kraw = NULL;
    if (scan_json_string(&q, &kraw, NULL) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    if (json_unescape_inplace(kraw) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
    q = skip_ws(q);
    if (*q != ':') { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
    ++q;
    int id = 0;
    if (scan_json_int(&q, &id) != 0 || id < 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
    if ((uint32_t)id > max_id) max_id = (uint32_t)id;
    free(kraw);
    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  tok->vocab_size = max_id + 1;
  tok->id_to_tok = (char **)xcalloc(tok->vocab_size, sizeof(char *));
  if (!tok->id_to_tok) return IE_TOK_GPTOSS_ERR_NOMEM;

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  q = p;
  while (*q) {
    q = skip_ws(q);
    if (*q == '}') break;
    char *kraw = NULL;
    if (scan_json_string(&q, &kraw, NULL) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    if (json_unescape_inplace(kraw) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
    q = skip_ws(q);
    if (*q != ':') { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
    ++q;
    int id = 0;
    if (scan_json_int(&q, &id) != 0 || id < 0 || (uint32_t)id >= tok->vocab_size) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }

    tok->id_to_tok[id] = kraw;
    (void)strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[id], (uint32_t)id);

    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  return IE_TOK_GPTOSS_OK;
}

static int rebuild_tok_to_id(ie_tok_gptoss_t *tok) {
  if (!tok) return -1;

  free(tok->tok_to_id);
  tok->tok_to_id = NULL;
  tok->tok_to_id_cap = 0;

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) return -1;

  for (size_t i = 0; i < tok->vocab_size; i++) {
    const char *s = tok->id_to_tok[i];
    if (!s) continue;
    if (strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, s, (uint32_t)i) != 0) return -1;
  }
  return 0;
}

static int parse_added_tokens_array(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "added_tokens");
  if (!p) return IE_TOK_GPTOSS_OK;

  p = skip_ws(p);
  if (!p || *p != '[') return IE_TOK_GPTOSS_ERR_JSON;
  p++;

  uint32_t max_id = (tok->vocab_size > 0) ? (uint32_t)(tok->vocab_size - 1) : 0;

  const char *q = p;
  for (;;) {
    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ']') break;
    if (*q != '{') return IE_TOK_GPTOSS_ERR_JSON;
    q++;

    int got_id = 0;
    uint32_t id = 0;

    for (;;) {
      q = skip_ws(q);
      if (!q) return IE_TOK_GPTOSS_ERR_JSON;
      if (*q == '}') { q++; break; }

      char *kraw = NULL;
      if (scan_json_string(&q, &kraw, NULL) != 0) return IE_TOK_GPTOSS_ERR_JSON;
      if (json_unescape_inplace(kraw) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }

      q = skip_ws(q);
      if (!q || *q != ':') { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
      q++;

      if (strcmp(kraw, "id") == 0) {
        int v = 0;
        if (scan_json_int(&q, &v) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
        if (v < 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
        id = (uint32_t)v;
        got_id = 1;
      } else {
        if (skip_json_value(&q) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
      }

      free(kraw);

      q = skip_ws(q);
      if (!q) return IE_TOK_GPTOSS_ERR_JSON;
      if (*q == ',') { q++; continue; }
      if (*q == '}') { q++; break; }
    }

    if (got_id && id > max_id) max_id = id;

    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ',') { q++; continue; }
    if (*q == ']') break;
  }

  size_t need = (size_t)max_id + 1;
  if (need > tok->vocab_size) {
    size_t old = tok->vocab_size;
    char **nt = (char **)realloc(tok->id_to_tok, need * sizeof(char *));
    if (!nt) return IE_TOK_GPTOSS_ERR_NOMEM;
    tok->id_to_tok = nt;
    for (size_t i = old; i < need; i++) tok->id_to_tok[i] = NULL;
    tok->vocab_size = (uint32_t)need;
    if (rebuild_tok_to_id(tok) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  q = p;
  for (;;) {
    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ']') break;
    if (*q != '{') return IE_TOK_GPTOSS_ERR_JSON;
    q++;

    int got_id = 0;
    uint32_t id = 0;
    char *content = NULL;

    for (;;) {
      q = skip_ws(q);
      if (!q) { if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      if (*q == '}') { q++; break; }

      char *kraw = NULL;
      if (scan_json_string(&q, &kraw, NULL) != 0) { if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      if (json_unescape_inplace(kraw) != 0) { free(kraw); if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }

      q = skip_ws(q);
      if (!q || *q != ':') { free(kraw); if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      q++;

      if (strcmp(kraw, "id") == 0) {
        int v = 0;
        if (scan_json_int(&q, &v) != 0) { free(kraw); if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
        if (v < 0) { free(kraw); if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
        id = (uint32_t)v;
        got_id = 1;
      } else if (strcmp(kraw, "content") == 0) {
        if (content) { free(content); content = NULL; }
        if (scan_json_string(&q, &content, NULL) != 0) { free(kraw); return IE_TOK_GPTOSS_ERR_JSON; }
        if (json_unescape_inplace(content) != 0) { free(kraw); free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      } else {
        if (skip_json_value(&q) != 0) { free(kraw); if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      }

      free(kraw);

      q = skip_ws(q);
      if (!q) { if (content) free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      if (*q == ',') { q++; continue; }
      if (*q == '}') { q++; break; }
    }

    if (got_id && content) {
      if (id >= tok->vocab_size) { free(content); return IE_TOK_GPTOSS_ERR_JSON; }
      if (tok->id_to_tok[id] == NULL) tok->id_to_tok[id] = content;
      else free(content);
      if (strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[id], id) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
    } else if (content) {
      free(content);
    }

    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ',') { q++; continue; }
    if (*q == ']') break;
  }

  return IE_TOK_GPTOSS_OK;
}

static int scan_merge_entry(const char **pp, char **a, char **b) {
  const char *p = skip_ws(*pp);
  if (!p || *p == '\0') return -1;

  *a = NULL;
  *b = NULL;

  if (*p == '"') {
    char *sraw = NULL;
    if (scan_json_string(&p, &sraw, NULL) != 0) return -1;
    if (json_unescape_inplace(sraw) != 0) { free(sraw); return -1; }

    char *sp = strchr(sraw, ' ');
    if (!sp) { free(sraw); return -1; }
    *sp = '\0';

    char *lhs = xstrdup_n(sraw, strlen(sraw));
    char *rhs = xstrdup_n(sp + 1, strlen(sp + 1));
    free(sraw);

    if (!lhs || !rhs) { free(lhs); free(rhs); return -1; }

    *a = lhs;
    *b = rhs;
    *pp = p;
    return 0;
  }

  if (*p == '[') {
    p++;
    p = skip_ws(p);

    char *lhs = NULL;
    char *rhs = NULL;

    if (scan_json_string(&p, &lhs, NULL) != 0) return -1;
    if (json_unescape_inplace(lhs) != 0) { free(lhs); return -1; }

    p = skip_ws(p);
    if (!p || *p != ',') { free(lhs); return -1; }
    p++;
    p = skip_ws(p);

    if (scan_json_string(&p, &rhs, NULL) != 0) { free(lhs); return -1; }
    if (json_unescape_inplace(rhs) != 0) { free(lhs); free(rhs); return -1; }

    p = skip_ws(p);
    if (!p || *p != ']') { free(lhs); free(rhs); return -1; }
    p++;

    *a = lhs;
    *b = rhs;
    *pp = p;
    return 0;
  }

  return -1;
}

static int parse_merges_array(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "merges");
  if (!p) return IE_TOK_GPTOSS_ERR_JSON;

  p = skip_ws(p);
  if (!p || *p != '[') return IE_TOK_GPTOSS_ERR_JSON;
  ++p;

  size_t merges = 0;
  const char *q = p;
  while (*q) {
    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ']') break;

    char *lhs = NULL;
    char *rhs = NULL;
    if (scan_merge_entry(&q, &lhs, &rhs) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    free(lhs);
    free(rhs);
    merges++;

    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ',') ++q;
  }

  if (intern_init(&tok->intern, merges * 4 + 1024, (size_t)(1u << 20)) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
  if (pairrank_map_init(&tok->pair_to_rank, &tok->pair_to_rank_cap, merges) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  q = p;
  uint32_t rank = 0;
  while (*q) {
    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ']') break;

    char *lhs = NULL;
    char *rhs = NULL;
    if (scan_merge_entry(&q, &lhs, &rhs) != 0) return IE_TOK_GPTOSS_ERR_JSON;

    uint32_t ida = 0, idb = 0;
    if (intern_get_or_add(&tok->intern, lhs, &ida) != 0 || intern_get_or_add(&tok->intern, rhs, &idb) != 0) {
      free(lhs); free(rhs);
      return IE_TOK_GPTOSS_ERR_NOMEM;
    }

    uint64_t key = ((uint64_t)ida << 32) | (uint64_t)idb;
    (void)pairrank_map_put(tok->pair_to_rank, tok->pair_to_rank_cap, key, rank++);

    free(lhs);
    free(rhs);

    q = skip_ws(q);
    if (!q) return IE_TOK_GPTOSS_ERR_JSON;
    if (*q == ',') ++q;
  }

  return IE_TOK_GPTOSS_OK;
}

static int parse_tokenizer_json(const char *json, ie_tok_gptoss_t *tok) {
  if (!json || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  int rc = parse_vocab_object(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  rc = parse_added_tokens_array(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  rc = parse_merges_array(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  tok->mode = IE_TOK_MODE_GPT2;
  return IE_TOK_GPTOSS_OK;
}

/* ============================================================================
 * Packed tokenizer format (IETOK1)
 * ========================================================================== */

static int rd_u32le(const unsigned char *buf, size_t n, size_t *io_off, uint32_t *out) {
  if (!buf || !io_off || !out) return -1;
  size_t off = *io_off;
  if (off + 4 > n) return -1;
  uint32_t v = 0;
  v |= (uint32_t)buf[off + 0];
  v |= (uint32_t)buf[off + 1] << 8;
  v |= (uint32_t)buf[off + 2] << 16;
  v |= (uint32_t)buf[off + 3] << 24;
  *io_off = off + 4;
  *out = v;
  return 0;
}

static int rd_u16le(const unsigned char *buf, size_t n, size_t *io_off, uint16_t *out) {
  if (!buf || !io_off || !out) return -1;
  size_t off = *io_off;
  if (off + 2 > n) return -1;
  uint16_t v = 0;
  v |= (uint16_t)buf[off + 0];
  v |= (uint16_t)buf[off + 1] << 8;
  *io_off = off + 2;
  *out = v;
  return 0;
}

static int rd_lp_string(const unsigned char *buf, size_t n, size_t *io_off, char **out_s) {
  if (!buf || !io_off || !out_s) return -1;
  uint32_t len = 0;
  if (rd_u32le(buf, n, io_off, &len) != 0) return -1;
  size_t off = *io_off;
  if (off + (size_t)len > n) return -1;
  char *s = (char *)malloc((size_t)len + 1);
  if (!s) return -1;
  memcpy(s, buf + off, (size_t)len);
  s[len] = '\0';
  *io_off = off + (size_t)len;
  *out_s = s;
  return 0;
}

static int parse_tokenizer_packed(const unsigned char *buf, size_t n, ie_tok_gptoss_t *tok) {
  if (!buf || n < 32 || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  size_t off = 0;
  if (n < 6) return IE_TOK_GPTOSS_ERR_JSON;
  if (memcmp(buf, "IETOK1", 6) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  off += 6;

  uint16_t version = 0;
  if (rd_u16le(buf, n, &off, &version) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (version != 1u) return IE_TOK_GPTOSS_ERR_JSON;

  uint32_t vocab_size = 0, merges_count = 0, off_vocab = 0, off_merges = 0, off_special = 0, reserved = 0;
  if (rd_u32le(buf, n, &off, &vocab_size) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (rd_u32le(buf, n, &off, &merges_count) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (rd_u32le(buf, n, &off, &off_vocab) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (rd_u32le(buf, n, &off, &off_merges) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (rd_u32le(buf, n, &off, &off_special) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (rd_u32le(buf, n, &off, &reserved) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  (void)reserved;

  if (vocab_size == 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (off_vocab >= n || off_merges > n || off_special > n) return IE_TOK_GPTOSS_ERR_JSON;
  if (!(off_vocab <= off_merges && off_merges <= off_special && off_special <= n)) return IE_TOK_GPTOSS_ERR_JSON;

  tok->vocab_size = vocab_size;
  tok->id_to_tok = (char **)xcalloc(tok->vocab_size, sizeof(char *));
  if (!tok->id_to_tok) return IE_TOK_GPTOSS_ERR_NOMEM;

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  off = (size_t)off_vocab;
  for (uint32_t i = 0; i < tok->vocab_size; ++i) {
    char *s = NULL;
    if (rd_lp_string(buf, n, &off, &s) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    tok->id_to_tok[i] = s;
    (void)strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[i], i);
  }

  if (intern_init(&tok->intern, (size_t)merges_count * 4 + 1024, (size_t)(1u << 20)) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
  if (pairrank_map_init(&tok->pair_to_rank, &tok->pair_to_rank_cap, (size_t)merges_count) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  off = (size_t)off_merges;
  for (uint32_t r = 0; r < merges_count; ++r) {
    char *a = NULL;
    char *b = NULL;
    if (rd_lp_string(buf, n, &off, &a) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    if (rd_lp_string(buf, n, &off, &b) != 0) { free(a); return IE_TOK_GPTOSS_ERR_JSON; }

    uint32_t ida = 0, idb = 0;
    if (intern_get_or_add(&tok->intern, a, &ida) != 0 || intern_get_or_add(&tok->intern, b, &idb) != 0) {
      free(a); free(b);
      return IE_TOK_GPTOSS_ERR_NOMEM;
    }

    uint64_t key = ((uint64_t)ida << 32) | (uint64_t)idb;
    (void)pairrank_map_put(tok->pair_to_rank, tok->pair_to_rank_cap, key, r);

    free(a);
    free(b);
  }

  tok->mode = IE_TOK_MODE_GPT2;
  return IE_TOK_GPTOSS_OK;
}

static int is_packed_tokenizer(const unsigned char *buf, size_t n) {
  if (!buf || n < 6) return 0;
  return memcmp(buf, "IETOK1", 6) == 0;
}

/* ============================================================================
 * .tiktoken parsing
 * ========================================================================== */

static int b64_val(int c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return 26 + (c - 'a');
  if (c >= '0' && c <= '9') return 52 + (c - '0');
  if (c == '+') return 62;
  if (c == '/') return 63;
  if (c == '-') return 62;
  if (c == '_') return 63;
  return -1;
}

static int b64_decode(const char *s, size_t n, uint8_t **out, size_t *out_n) {
  if (!s || !out || !out_n) return -1;
  *out = NULL;
  *out_n = 0;

  size_t max_out = (n / 4 + 1) * 3;
  uint8_t *buf = (uint8_t *)malloc(max_out ? max_out : 1);
  if (!buf) return -1;

  size_t wi = 0;
  int quad[4];
  int qn = 0;

  for (size_t i = 0; i < n; ++i) {
    unsigned char c = (unsigned char)s[i];
    if (c == '=') {
      quad[qn++] = -2;
    } else {
      int v = b64_val((int)c);
      if (v < 0) { free(buf); return -1; }
      quad[qn++] = v;
    }

    if (qn == 4) {
      if (quad[0] < 0 || quad[1] < 0) { free(buf); return -1; }

      uint32_t x = ((uint32_t)quad[0] << 18) | ((uint32_t)quad[1] << 12);
      uint32_t b2 = (quad[2] >= 0) ? (uint32_t)quad[2] : 0;
      uint32_t b3 = (quad[3] >= 0) ? (uint32_t)quad[3] : 0;
      x |= (b2 << 6) | b3;

      buf[wi++] = (uint8_t)((x >> 16) & 0xFF);
      if (quad[2] != -2) buf[wi++] = (uint8_t)((x >> 8) & 0xFF);
      if (quad[3] != -2) buf[wi++] = (uint8_t)(x & 0xFF);

      qn = 0;
    }
  }

  if (qn != 0) { free(buf); return -1; }

  *out = buf;
  *out_n = wi;
  return 0;
}

static int parse_uint32_str(const char *s, const char *e, uint32_t *out) {
  if (!s || !e || !out) return -1;
  while (s < e && (*s == ' ' || *s == '\t' || *s == '\r')) s++;
  if (s >= e) return -1;
  uint64_t v = 0;
  for (const char *p = s; p < e; ++p) {
    if (*p < '0' || *p > '9') return -1;
    v = v * 10 + (uint64_t)(*p - '0');
    if (v > 0xFFFFFFFFu) return -1;
  }
  *out = (uint32_t)v;
  return 0;
}

static int ensure_id_to_bytes(ie_tok_gptoss_t *tok, uint32_t need) {
  if (!tok) return -1;
  if (need <= tok->vocab_size && tok->id_to_bytes) return 0;

  uint32_t old = tok->vocab_size;
  uint32_t nv = (old ? old : 1024);
  while (nv < need) nv *= 2;

  bytes_view_t *nb = (bytes_view_t *)realloc(tok->id_to_bytes, (size_t)nv * sizeof(*nb));
  if (!nb) return -1;
  tok->id_to_bytes = nb;
  for (uint32_t i = old; i < nv; ++i) {
    tok->id_to_bytes[i].p = NULL;
    tok->id_to_bytes[i].n = 0;
  }
  tok->vocab_size = nv;
  return 0;
}

static int finalize_tt_vocab_size(ie_tok_gptoss_t *tok, uint32_t max_id_plus1) {
  if (!tok) return -1;
  if (max_id_plus1 == 0) return -1;
  bytes_view_t *nb = (bytes_view_t *)realloc(tok->id_to_bytes, (size_t)max_id_plus1 * sizeof(*nb));
  if (!nb) return -1;
  tok->id_to_bytes = nb;
  tok->vocab_size = max_id_plus1;
  return 0;
}

static int parse_tokenizer_tiktoken(const char *text, size_t len, ie_tok_gptoss_t *tok) {
  if (!text || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  size_t lines = 0;
  for (size_t i = 0; i < len; ++i) if (text[i] == '\n') lines++;
  if (lines < 128) lines = 128;

  if (bytesid_map_init(&tok->bytes_to_id, &tok->bytes_to_id_cap, lines) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  tok->id_to_bytes = NULL;
  tok->vocab_size = 0;

  uint32_t max_id = 0;
  size_t pos = 0;

  while (pos < len) {
    size_t line_start = pos;
    while (pos < len && text[pos] != '\n') pos++;
    size_t line_end = pos;
    if (pos < len && text[pos] == '\n') pos++;

    while (line_start < line_end && (text[line_start] == ' ' || text[line_start] == '\t' || text[line_start] == '\r')) line_start++;
    while (line_end > line_start && (text[line_end - 1] == ' ' || text[line_end - 1] == '\t' || text[line_end - 1] == '\r')) line_end--;

    if (line_end <= line_start) continue;
    if (text[line_start] == '#') continue;

    size_t sp = line_start;
    while (sp < line_end && text[sp] != ' ' && text[sp] != '\t') sp++;
    if (sp == line_start || sp >= line_end) return IE_TOK_GPTOSS_ERR_JSON;

    size_t b64_start = line_start;
    size_t b64_end = sp;
    while (sp < line_end && (text[sp] == ' ' || text[sp] == '\t')) sp++;
    if (sp >= line_end) return IE_TOK_GPTOSS_ERR_JSON;

    uint32_t id = 0;
    if (parse_uint32_str(text + sp, text + line_end, &id) != 0) return IE_TOK_GPTOSS_ERR_JSON;

    if (ensure_id_to_bytes(tok, id + 1) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
    if (id > max_id) max_id = id;

    uint8_t *decoded = NULL;
    size_t decoded_n = 0;
    if (b64_decode(text + b64_start, b64_end - b64_start, &decoded, &decoded_n) != 0) return IE_TOK_GPTOSS_ERR_JSON;

    uint8_t *dst = bytearena_alloc(&tok->tt_arena, decoded_n ? decoded_n : 1);
    if (!dst) { free(decoded); return IE_TOK_GPTOSS_ERR_NOMEM; }
    if (decoded_n) memcpy(dst, decoded, decoded_n);
    free(decoded);

    tok->id_to_bytes[id].p = dst;
    tok->id_to_bytes[id].n = (uint32_t)decoded_n;

    if (bytesid_map_put(tok->bytes_to_id, tok->bytes_to_id_cap, dst, (uint32_t)decoded_n, id) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  if (finalize_tt_vocab_size(tok, max_id + 1) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;

  for (uint32_t i = 0; i < tok->vocab_size; ++i) {
    if (!tok->id_to_bytes[i].p && tok->id_to_bytes[i].n == 0) return IE_TOK_GPTOSS_ERR_JSON;
  }

  tok->tt_have_byte_id = 1;
  for (int b = 0; b < 256; ++b) {
    uint8_t bb = (uint8_t)b;
    uint32_t id = 0;
    if (bytesid_map_get(tok->bytes_to_id, tok->bytes_to_id_cap, &bb, 1, &id) != 0) { tok->tt_have_byte_id = 0; break; }
    tok->tt_byte_id[b] = id;
  }
  if (!tok->tt_have_byte_id) return IE_TOK_GPTOSS_ERR_JSON;

  tok->mode = IE_TOK_MODE_TIKTOKEN;
  return IE_TOK_GPTOSS_OK;
}

/* ============================================================================
 * BPE encode helpers
 * ========================================================================== */

static int to_bytelevel_unicode(const byteunicode_t *bu, const char *s, size_t n, strbuf_t *out) {
  if (!bu || (!s && n) || !out) return -1;
  for (size_t i = 0; i < n; ++i) {
    uint8_t b = (uint8_t)(unsigned char)s[i];
    uint32_t cp = bu->byte_to_cp[b];
    char tmp[4];
    size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
    if (wn == 0) return -1;
    if (strbuf_append_bytes(out, tmp, wn) != 0) return -1;
  }
  return 0;
}

static int pretokenize_basic(const char *text, char ***out_parts, size_t *out_n) {
  if (!text || !out_parts || !out_n) return -1;
  *out_parts = NULL;
  *out_n = 0;

  size_t cap = 0, n = 0;
  char **parts = NULL;
  const char *p = text;

  while (*p) {
    if (isspace((unsigned char)*p)) {
      const char *s = p;
      while (*p && isspace((unsigned char)*p)) ++p;
      size_t len = (size_t)(p - s);
      char *seg = xstrdup_n(s, len);
      if (!seg) goto fail;
      if (n == cap) {
        size_t nc = cap ? cap * 2 : 64;
        char **np = (char **)realloc(parts, nc * sizeof(*np));
        if (!np) { free(seg); goto fail; }
        parts = np; cap = nc;
      }
      parts[n++] = seg;
      continue;
    }

    const char *s = p;
    while (*p && !isspace((unsigned char)*p)) ++p;
    size_t len = (size_t)(p - s);

    int have_lead_space = 0;
    if (n > 0 && parts[n - 1] && parts[n - 1][0] != '\0') {
      char *w = parts[n - 1];
      size_t wl = strlen(w);
      if (wl > 0 && w[wl - 1] == ' ') { w[wl - 1] = '\0'; have_lead_space = 1; }
    }

    strbuf_t sb = {0};
    if (have_lead_space) {
      char sp = ' ';
      if (strbuf_append_bytes(&sb, &sp, 1) != 0) { strbuf_free(&sb); goto fail; }
    }
    if (strbuf_append_bytes(&sb, s, len) != 0) { strbuf_free(&sb); goto fail; }
    if (strbuf_reserve(&sb, 1) != 0) { strbuf_free(&sb); goto fail; }
    sb.v[sb.n] = '\0';

    if (n == cap) {
      size_t nc = cap ? cap * 2 : 64;
      char **np = (char **)realloc(parts, nc * sizeof(*np));
      if (!np) { strbuf_free(&sb); goto fail; }
      parts = np; cap = nc;
    }
    parts[n++] = sb.v;
  }

  size_t w = 0;
  for (size_t i = 0; i < n; ++i) {
    if (parts[i] && parts[i][0] == '\0') { free(parts[i]); parts[i] = NULL; continue; }
    parts[w++] = parts[i];
  }
  n = w;

  *out_parts = parts;
  *out_n = n;
  return 0;

fail:
  if (parts) {
    for (size_t i = 0; i < n; ++i) free(parts[i]);
    free(parts);
  }
  return -1;
}

/* GPT-2: BPE over interned unicode symbols (existing behavior) */
static int bpe_encode_to_ids_gpt2(const ie_tok_gptoss_t *tok, const char *in, u32buf_t *out_ids) {
  if (!tok || !in || !out_ids) return -1;

  typedef struct sym_s { const char *s; uint32_t id; } sym_t;

  const size_t in_len = strlen(in);
  size_t i = 0;

  sym_t *syms = NULL;
  size_t nsyms = 0, cap = 0;
  strbuf_t scratch = {0};

  while (i < in_len) {
    uint32_t cp = 0;
    size_t old = i;
    if (utf8_next_cp(in, in_len, &i, &cp) != 0) { strbuf_free(&scratch); free(syms); return -1; }

    const char *slice = in + old;
    size_t slen = i - old;

    if (strbuf_reserve(&scratch, slen + 1) != 0) { strbuf_free(&scratch); free(syms); return -1; }
    char *dst = scratch.v + scratch.n;
    memcpy(dst, slice, slen);
    dst[slen] = '\0';
    scratch.n += slen + 1;

    if (nsyms == cap) {
      size_t nc = cap ? cap * 2 : 64;
      sym_t *ns = (sym_t *)realloc(syms, nc * sizeof(*ns));
      if (!ns) { strbuf_free(&scratch); free(syms); return -1; }
      syms = ns;
      cap = nc;
    }

    uint32_t sid = 0;
    if (intern_get(&tok->intern, dst, &sid) != 0) sid = 0;
    syms[nsyms].s = dst;
    syms[nsyms].id = sid;
    nsyms++;
  }

  if (nsyms == 0) { strbuf_free(&scratch); free(syms); return 0; }

  for (;;) {
    uint32_t best_rank = UINT32_MAX;
    size_t best_i = (size_t)-1;

    for (size_t k = 0; k + 1 < nsyms; ++k) {
      if (syms[k].id == 0 || syms[k + 1].id == 0) continue;
      uint64_t key = ((uint64_t)syms[k].id << 32) | (uint64_t)syms[k + 1].id;
      uint32_t r = 0;
      if (pairrank_map_get(tok->pair_to_rank, tok->pair_to_rank_cap, key, &r) == 0) {
        if (r < best_rank) { best_rank = r; best_i = k; }
      }
    }

    if (best_i == (size_t)-1) break;

    const char *a = syms[best_i].s;
    const char *b = syms[best_i + 1].s;

    size_t al = strlen(a), bl = strlen(b);
    if (strbuf_reserve(&scratch, al + bl + 1) != 0) { strbuf_free(&scratch); free(syms); return -1; }
    char *dst = scratch.v + scratch.n;
    memcpy(dst, a, al);
    memcpy(dst + al, b, bl);
    dst[al + bl] = '\0';
    scratch.n += al + bl + 1;

    uint32_t sid = 0;
    if (intern_get(&tok->intern, dst, &sid) != 0) sid = 0;

    syms[best_i].s = dst;
    syms[best_i].id = sid;

    for (size_t k = best_i + 1; k + 1 < nsyms; ++k) syms[k] = syms[k + 1];
    nsyms--;
  }

  for (size_t k = 0; k < nsyms; ++k) {
    const char *sym = syms[k].s;
    uint32_t id = 0;
    if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, sym, &id) == 0) {
      if (u32buf_push(out_ids, id) != 0) { strbuf_free(&scratch); free(syms); return -1; }
      continue;
    }

    const size_t slen = strlen(sym);
    size_t pos = 0;
    while (pos < slen) {
      size_t old = pos;
      uint32_t cp = 0;
      if (utf8_next_cp(sym, slen, &pos, &cp) != 0) break;
      (void)cp;
      char *one = xstrdup_n(sym + old, pos - old);
      if (!one) { strbuf_free(&scratch); free(syms); return -1; }
      uint32_t tid = 0;
      int ok = (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, one, &tid) == 0);
      free(one);
      if (!ok) { strbuf_free(&scratch); free(syms); return -1; }
      if (u32buf_push(out_ids, tid) != 0) { strbuf_free(&scratch); free(syms); return -1; }
    }
  }

  strbuf_free(&scratch);
  free(syms);
  return 0;
}

/* .tiktoken: byte-level BPE over raw bytes */
typedef struct ttpiece_s {
  const uint8_t *p;
  uint32_t n;
  uint32_t id;
} ttpiece_t;

static int tt_bpe_encode_segment(const ie_tok_gptoss_t *tok,
                                 const uint8_t *seg, uint32_t seg_n,
                                 u32buf_t *out_ids) {
  if (!tok || !out_ids) return -1;
  if (seg_n == 0) return 0;
  if (!seg) return -1;

  ttpiece_t *pieces = (ttpiece_t *)malloc((size_t)seg_n * sizeof(*pieces));
  if (!pieces) return -1;

  for (uint32_t i = 0; i < seg_n; ++i) {
    pieces[i].p = seg + i;
    pieces[i].n = 1;
    pieces[i].id = tok->tt_byte_id[seg[i]];
  }

  uint32_t np = seg_n;
  bytearena_t tmp = {0};

  while (np >= 2) {
    uint32_t best_rank = UINT32_MAX;
    uint32_t best_id = 0;
    uint32_t best_i = UINT32_MAX;

    for (uint32_t i = 0; i + 1 < np; ++i) {
      uint32_t cand_id = 0;
      if (bytesid_map_get_concat(tok->bytes_to_id, tok->bytes_to_id_cap,
                                 pieces[i].p, pieces[i].n,
                                 pieces[i + 1].p, pieces[i + 1].n,
                                 &cand_id) == 0) {
        uint32_t cand_rank = cand_id;
        if (cand_rank < best_rank) {
          best_rank = cand_rank;
          best_id = cand_id;
          best_i = i;
        }
      }
    }

    if (best_i == UINT32_MAX) break;

    uint32_t an = pieces[best_i].n;
    uint32_t bn = pieces[best_i + 1].n;
    uint32_t mn = an + bn;

    uint8_t *dst = bytearena_alloc(&tmp, (size_t)mn);
    if (!dst) { bytearena_free(&tmp); free(pieces); return -1; }
    memcpy(dst, pieces[best_i].p, an);
    memcpy(dst + an, pieces[best_i + 1].p, bn);

    pieces[best_i].p = dst;
    pieces[best_i].n = mn;
    pieces[best_i].id = best_id;

    for (uint32_t j = best_i + 1; j + 1 < np; ++j) pieces[j] = pieces[j + 1];
    np--;
  }

  for (uint32_t i = 0; i < np; ++i) {
    if (u32buf_push(out_ids, pieces[i].id) != 0) { bytearena_free(&tmp); free(pieces); return -1; }
  }

  bytearena_free(&tmp);
  free(pieces);
  return 0;
}

static int tt_encode_text(const ie_tok_gptoss_t *tok, const char *text, u32buf_t *out_ids) {
  if (!tok || !text || !out_ids) return -1;
  const uint8_t *p = (const uint8_t *)text;
  size_t n = strlen(text);

  size_t i = 0;
  while (i < n) {
    size_t j = i;
    int ws = isspace((unsigned char)p[i]) ? 1 : 0;
    while (j < n) {
      int ws2 = isspace((unsigned char)p[j]) ? 1 : 0;
      if (ws2 != ws) break;
      j++;
    }
    uint32_t seg_n = (uint32_t)(j - i);
    if (tt_bpe_encode_segment(tok, p + i, seg_n, out_ids) != 0) return -1;
    i = j;
  }
  return 0;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

int ie_tok_gptoss_open(const char *tokenizer_path, ie_tok_gptoss_t **out_tok) {
  if (!tokenizer_path || !out_tok) return IE_TOK_GPTOSS_ERR_ARGS;
  *out_tok = NULL;

  char *buf = NULL;
  size_t len = 0;
  if (read_all_bytes(tokenizer_path, &buf, &len) != 0) return IE_TOK_GPTOSS_ERR_IO;

  ie_tok_gptoss_t *tok = (ie_tok_gptoss_t *)xcalloc(1, sizeof(*tok));
  if (!tok) { free(buf); return IE_TOK_GPTOSS_ERR_NOMEM; }

  byteunicode_init(&tok->bu);

  int rc = IE_TOK_GPTOSS_ERR_JSON;

  if (is_packed_tokenizer((const unsigned char *)buf, len)) {
    rc = parse_tokenizer_packed((const unsigned char *)buf, len, tok);
  } else if (path_ends_with(tokenizer_path, ".tiktoken")) {
    rc = parse_tokenizer_tiktoken(buf, len, tok);
  } else {
    rc = parse_tokenizer_json(buf, tok);
  }

  free(buf);

  if (rc != IE_TOK_GPTOSS_OK) {
    ie_tok_gptoss_close(tok);
    return rc;
  }

  *out_tok = tok;
  return IE_TOK_GPTOSS_OK;
}

void ie_tok_gptoss_close(ie_tok_gptoss_t *tok) {
  if (!tok) return;

  if (tok->id_to_tok) {
    for (uint32_t i = 0; i < tok->vocab_size; ++i) free(tok->id_to_tok[i]);
    free(tok->id_to_tok);
  }
  free(tok->tok_to_id);
  free(tok->pair_to_rank);
  intern_free(&tok->intern);

  free(tok->id_to_bytes);
  free(tok->bytes_to_id);
  bytearena_free(&tok->tt_arena);

  free(tok);
}

uint32_t ie_tok_gptoss_vocab_size(const ie_tok_gptoss_t *tok) {
  if (!tok) return 0;
  return tok->vocab_size;
}

int ie_tok_gptoss_decode(const ie_tok_gptoss_t *tok,
                         const uint32_t *ids,
                         uint32_t count,
                         char *out,
                         size_t *inout_bytes) {
  if (!tok || (!ids && count) || !inout_bytes) return IE_TOK_GPTOSS_ERR_ARGS;

  strbuf_t bytes = {0};

  if (tok->mode == IE_TOK_MODE_TIKTOKEN) {
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t id = ids[i];
      if (id >= tok->vocab_size) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_RANGE; }
      const uint8_t *p = tok->id_to_bytes[id].p;
      uint32_t n = tok->id_to_bytes[id].n;
      if (n && (!p)) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_INTERNAL; }
      if (n) {
        if (strbuf_append_bytes(&bytes, (const char *)p, (size_t)n) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
      }
    }

    if (strbuf_reserve(&bytes, 1) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
    bytes.v[bytes.n] = '\0';

    size_t need = bytes.n + 1;
    if (!out || *inout_bytes == 0) { *inout_bytes = need; strbuf_free(&bytes); return IE_TOK_GPTOSS_OK; }
    if (*inout_bytes < need) { *inout_bytes = need; strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_RANGE; }

    memcpy(out, bytes.v, need);
    *inout_bytes = need;
    strbuf_free(&bytes);
    return IE_TOK_GPTOSS_OK;
  }

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t id = ids[i];
    if (id >= tok->vocab_size) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_RANGE; }
    const char *t = tok->id_to_tok[id];
    if (!t) t = "";

    const size_t tlen = strlen(t);
    size_t pos = 0;
    while (pos < tlen) {
      uint32_t cp = 0;
      int st = utf8_next_cp(t, tlen, &pos, &cp);
      if (st != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_INTERNAL; }

      uint8_t b = 0;
      if (byteunicode_cp_to_byte(&tok->bu, cp, &b) == 0) {
        char c = (char)b;
        if (strbuf_append_bytes(&bytes, &c, 1) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
      } else {
        char tmp[4];
        size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
        if (wn == 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_INTERNAL; }
        if (strbuf_append_bytes(&bytes, tmp, wn) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
      }
    }
  }

  if (strbuf_reserve(&bytes, 1) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
  bytes.v[bytes.n] = '\0';

  size_t need = bytes.n + 1;
  if (!out || *inout_bytes == 0) { *inout_bytes = need; strbuf_free(&bytes); return IE_TOK_GPTOSS_OK; }
  if (*inout_bytes < need) { *inout_bytes = need; strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_RANGE; }

  memcpy(out, bytes.v, need);
  *inout_bytes = need;
  strbuf_free(&bytes);
  return IE_TOK_GPTOSS_OK;
}

int ie_tok_gptoss_encode(const ie_tok_gptoss_t *tok,
                         const char *text,
                         uint32_t *ids,
                         uint32_t *inout_count) {
  if (!tok || !text || !inout_count) return IE_TOK_GPTOSS_ERR_ARGS;

  u32buf_t out_ids = {0};

  if (tok->mode == IE_TOK_MODE_TIKTOKEN) {
    if (tt_encode_text(tok, text, &out_ids) != 0) { free(out_ids.v); return IE_TOK_GPTOSS_ERR_INTERNAL; }
  } else {
    char **parts = NULL;
    size_t nparts = 0;
    if (pretokenize_basic(text, &parts, &nparts) != 0) { free(out_ids.v); return IE_TOK_GPTOSS_ERR_INTERNAL; }

    for (size_t i = 0; i < nparts; ++i) {
      const char *seg = parts[i];
      if (!seg || !*seg) { free(parts[i]); continue; }

      strbuf_t bl = {0};
      if (to_bytelevel_unicode(&tok->bu, seg, strlen(seg), &bl) != 0) { strbuf_free(&bl); goto fail_gpt2; }
      if (strbuf_reserve(&bl, 1) != 0) { strbuf_free(&bl); goto fail_gpt2; }
      bl.v[bl.n] = '\0';

      if (bpe_encode_to_ids_gpt2(tok, bl.v, &out_ids) != 0) { strbuf_free(&bl); goto fail_gpt2; }
      strbuf_free(&bl);
      free(parts[i]);
    }

    free(parts);
    parts = NULL;
    nparts = 0;
  }

  uint32_t need = (uint32_t)out_ids.n;
  if (!ids || *inout_count == 0) {
    *inout_count = need;
    free(out_ids.v);
    return IE_TOK_GPTOSS_OK;
  }

  if (*inout_count < need) {
    *inout_count = need;
    free(out_ids.v);
    return IE_TOK_GPTOSS_ERR_RANGE;
  }

  memcpy(ids, out_ids.v, (size_t)need * sizeof(uint32_t));
  *inout_count = need;
  free(out_ids.v);
  return IE_TOK_GPTOSS_OK;

fail_gpt2:
  free(out_ids.v);
  return IE_TOK_GPTOSS_ERR_INTERNAL;
}
