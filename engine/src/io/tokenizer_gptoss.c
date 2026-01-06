/* ==========================================================================
 * File: engine/src/io/tokenizer_gptoss.c
 * ==========================================================================
 */
/**
 * @file tokenizer_gptoss.c
 * @brief GPT-OSS (GPT-2 style) byte-level BPE tokenizer.
 *
 * This implementation supports two on-disk formats:
 *  - HuggingFace `tokenizer.json` (slow to load, JSON scanned with a minimal parser)
 *  - Packed tokenizer produced by `scripts/pack_tokenizer.py` (recommended)
 *
 * Features:
 *  - Load vocab (id <-> token string)
 *  - Load merges (pair ranks)
 *  - GPT-2 `bytes_to_unicode` mapping
 *  - Encode: UTF-8 text -> token ids
 *  - Decode: token ids -> UTF-8 text
 *
 * Notes:
 *  - Correctness is prioritized over speed.
 *  - Pretokenization is a lightweight approximation of the GPT-2 regex.
 *    It attaches a single leading space to a token when present.
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

/**
 * @brief FNV-1a 64-bit hash.
 *
 * @param[in] data Pointer to bytes.
 * @param[in] n    Number of bytes.
 * @return 64-bit hash.
 */
static uint64_t fnv1a64(const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint64_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

/**
 * @brief FNV-1a 32-bit hash.
 *
 * @param[in] data Pointer to bytes.
 * @param[in] n    Number of bytes.
 * @return 32-bit hash.
 */
static uint32_t fnv1a32(const void *data, size_t n) {
  const unsigned char *p = (const unsigned char *)data;
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint32_t)p[i];
    h *= 16777619u;
  }
  return h;
}

/**
 * @brief calloc wrapper with overflow guard.
 */
static void *xcalloc(size_t n, size_t sz) {
  if (n == 0 || sz == 0) return NULL;
  if (n > (SIZE_MAX / sz)) return NULL;
  return calloc(n, sz);
}

/**
 * @brief realloc wrapper.
 */
static void *xrealloc(void *p, size_t n) {
  if (n == 0) return NULL;
  return realloc(p, n);
}

/**
 * @brief Duplicate a byte slice as a NUL-terminated string.
 *
 * @param[in] s Input pointer.
 * @param[in] n Number of bytes.
 * @return Newly allocated string or NULL.
 */
static char *xstrdup_n(const char *s, size_t n) {
  char *p = (char *)malloc(n + 1);
  if (!p) return NULL;
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

/**
 * @brief Read the entire file into memory.
 *
 * The returned buffer is NUL-terminated for convenience.
 */
static int read_all_bytes(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf || !out_len) return -1;
  *out_buf = NULL;
  *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -1;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return -1;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -1;
  }

  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return -1;
  }
  size_t rd = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if (rd != (size_t)sz) {
    free(buf);
    return -1;
  }
  buf[sz] = '\0';
  *out_buf = buf;
  *out_len = (size_t)sz;
  return 0;
}

/* ============================================================================
 * Minimal UTF-8 decode/encode (codepoint-based)
 * ========================================================================== */

/**
 * @brief Decode the next UTF-8 codepoint.
 *
 * @param[in]     s      UTF-8 bytes.
 * @param[in]     n      Buffer length.
 * @param[in,out] io_i   Cursor (byte index) advanced on success.
 * @param[out]    out_cp Codepoint.
 * @return 0 on success, 1 on end-of-buffer, -1 on malformed UTF-8.
 */
static int utf8_next_cp(const char *s, size_t n, size_t *io_i, uint32_t *out_cp) {
  if (!s || !io_i || !out_cp) return -1;
  size_t i = *io_i;
  if (i >= n) return 1;

  unsigned char c0 = (unsigned char)s[i++];
  if (c0 < 0x80) {
    *out_cp = (uint32_t)c0;
    *io_i = i;
    return 0;
  }

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

/**
 * @brief Encode a single codepoint as UTF-8.
 *
 * @param[out] dst Output buffer.
 * @param[in]  cap Capacity in bytes.
 * @param[in]  cp  Codepoint.
 * @return Number of bytes written (1..4), or 0 on insufficient capacity.
 */
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

/**
 * @brief Skip ASCII whitespace.
 */
static const char *skip_ws(const char *p) {
  while (*p && (unsigned char)*p <= 0x20) ++p;
  return p;
}

/**
 * @brief Scan a JSON string (raw, with escapes preserved).
 *
 * The returned string must be freed by the caller.
 */
static int scan_json_string(const char **pp, char **out_s, size_t *out_n) {
  const char *p = skip_ws(*pp);
  if (*p != '"') return -1;
  ++p;

  const char *start = p;
  while (*p) {
    if (*p == '\\') {
      ++p;
      if (!*p) return -1;
      ++p;
      continue;
    }
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

/**
 * @brief Unescape a JSON string in-place.
 *
 * Supports common escapes plus minimal BMP-only \uXXXX decoding.
 */
static int json_unescape_inplace(char *s) {
  if (!s) return -1;
  char *w = s;
  for (char *r = s; *r; ++r) {
    if (*r != '\\') {
      *w++ = *r;
      continue;
    }
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

/**
 * @brief Scan a JSON integer.
 */
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

/**
 * @brief Find a JSON key ("key": ...) anywhere in a document.
 *
 * This is a pragmatic scanner: it does not build a full JSON AST.
 */
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

/**
 * @brief Skip a JSON value starting at *pp.
 *
 * This supports strings, numbers, objects, arrays, and true/false/null.
 */
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
    int depth = 0;
    int in_str = 0;
    int esc = 0;
    for (; *p; p++) {
      const char c = *p;
      if (in_str) {
        if (esc) esc = 0;
        else if (c == '\\') esc = 1;
        else if (c == '"') in_str = 0;
        continue;
      }
      if (c == '"') { in_str = 1; continue; }
      if (c == '{') depth++;
      else if (c == '}') {
        depth--;
        if (depth == 0) { p++; *pp = p; return 0; }
      }
    }
    return -1;
  }

  if (*p == '[') {
    int depth = 0;
    int in_str = 0;
    int esc = 0;
    for (; *p; p++) {
      const char c = *p;
      if (in_str) {
        if (esc) esc = 0;
        else if (c == '\\') esc = 1;
        else if (c == '"') in_str = 0;
        continue;
      }
      if (c == '"') { in_str = 1; continue; }
      if (c == '[') depth++;
      else if (c == ']') {
        depth--;
        if (depth == 0) { p++; *pp = p; return 0; }
      }
    }
    return -1;
  }

  if (*p == '-' || (*p >= '0' && *p <= '9')) {
    p++;
    while (*p) {
      const char c = *p;
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        p++;
        continue;
      }
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
 * Hash maps: string->id and pair->rank
 * ========================================================================== */

/**
 * @brief Entry for token string -> id map.
 */
typedef struct strid_ent_s {
  uint64_t h;
  const char *k;
  uint32_t v;
  int used;
} strid_ent_t;

/**
 * @brief Entry for merge pair -> rank map.
 */
typedef struct pairrank_ent_s {
  uint64_t key;
  uint32_t rank;
  int used;
} pairrank_ent_t;

/**
 * @brief Compute next power-of-two >= x.
 */
static size_t next_pow2(size_t x) {
  size_t p = 1;
  while (p < x) p <<= 1;
  return p;
}

/**
 * @brief Initialize token->id hash map.
 */
static int strid_map_init(strid_ent_t **out, size_t *out_cap, size_t want) {
  size_t cap = next_pow2(want * 2 + 64);
  strid_ent_t *m = (strid_ent_t *)xcalloc(cap, sizeof(*m));
  if (!m) return -1;
  *out = m;
  *out_cap = cap;
  return 0;
}

/**
 * @brief Insert into token->id map.
 */
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
    if (m[i].h == h && strcmp(m[i].k, k) == 0) {
      m[i].v = v;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/**
 * @brief Lookup in token->id map.
 */
static int strid_map_get(const strid_ent_t *m, size_t cap, const char *k, uint32_t *out_v) {
  if (!m || cap == 0 || !k || !out_v) return -1;
  uint64_t h = fnv1a64(k, strlen(k));
  size_t mask = cap - 1;
  size_t i = (size_t)h & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].h == h && strcmp(m[i].k, k) == 0) {
      *out_v = m[i].v;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/**
 * @brief Initialize pair->rank map.
 */
static int pairrank_map_init(pairrank_ent_t **out, size_t *out_cap, size_t want) {
  size_t cap = next_pow2(want * 2 + 64);
  pairrank_ent_t *m = (pairrank_ent_t *)xcalloc(cap, sizeof(*m));
  if (!m) return -1;
  *out = m;
  *out_cap = cap;
  return 0;
}

/**
 * @brief Insert into pair->rank map.
 */
static int pairrank_map_put(pairrank_ent_t *m, size_t cap, uint64_t key, uint32_t rank) {
  if (!m || cap == 0) return -1;
  size_t mask = cap - 1;
  size_t i = (size_t)key & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) {
      m[i].used = 1;
      m[i].key = key;
      m[i].rank = rank;
      return 0;
    }
    if (m[i].key == key) {
      m[i].rank = rank;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/**
 * @brief Lookup in pair->rank map.
 */
static int pairrank_map_get(const pairrank_ent_t *m, size_t cap, uint64_t key, uint32_t *out_rank) {
  if (!m || cap == 0 || !out_rank) return -1;
  size_t mask = cap - 1;
  size_t i = (size_t)key & mask;
  for (size_t step = 0; step < cap; ++step) {
    if (!m[i].used) return -1;
    if (m[i].key == key) {
      *out_rank = m[i].rank;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/* ============================================================================
 * String interning for BPE symbols
 * ========================================================================== */

/**
 * @brief Intern table entry.
 */
typedef struct intern_ent_s {
  uint32_t h;
  const char *s;
  uint32_t id;
  int used;
} intern_ent_t;

/**
 * @brief Intern table with an arena backing store.
 */
typedef struct intern_s {
  intern_ent_t *m;
  size_t cap;
  char *arena;
  size_t arena_sz;
  size_t arena_used;
  uint32_t next_id;
} intern_t;

/**
 * @brief Initialize intern table.
 */
static int intern_init(intern_t *in, size_t want_syms, size_t arena_hint) {
  if (!in) return -1;
  memset(in, 0, sizeof(*in));
  in->cap = next_pow2(want_syms * 2 + 64);
  in->m = (intern_ent_t *)xcalloc(in->cap, sizeof(*in->m));
  if (!in->m) return -1;
  in->arena_sz = (arena_hint ? arena_hint : (1u << 20));
  in->arena = (char *)malloc(in->arena_sz);
  if (!in->arena) {
    free(in->m);
    memset(in, 0, sizeof(*in));
    return -1;
  }
  in->arena_used = 0;
  in->next_id = 1;
  return 0;
}

/**
 * @brief Free intern table.
 */
static void intern_free(intern_t *in) {
  if (!in) return;
  free(in->m);
  free(in->arena);
  memset(in, 0, sizeof(*in));
}

/**
 * @brief Store a string into the intern arena.
 */
static const char *intern_store(intern_t *in, const char *s, size_t n) {
  if (!in || !s) return NULL;
  if (n + 1 > in->arena_sz - in->arena_used) {
    size_t new_sz = in->arena_sz * 2;
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

/**
 * @brief Get or add an interned string ID.
 */
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
    if (in->m[i].h == h && strcmp(in->m[i].s, s) == 0) {
      *out_id = in->m[i].id;
      return 0;
    }
    i = (i + 1) & mask;
  }
  return -1;
}

/* ============================================================================
 * GPT-2 bytes_to_unicode mapping (byte-level BPE)
 * ========================================================================== */

/**
 * @brief Byte-to-unicode mapping state.
 */
typedef struct byteunicode_s {
  uint32_t byte_to_cp[256];
  int32_t cp_to_byte[2048];
  uint32_t cp_to_byte_cap;
} byteunicode_t;

/**
 * @brief Initialize GPT-2 bytes_to_unicode mapping.
 */
static void byteunicode_init(byteunicode_t *m) {
  memset(m, 0, sizeof(*m));
  for (size_t i = 0; i < 2048; ++i) m->cp_to_byte[i] = -1;
  m->cp_to_byte_cap = 2048;

  int used[256];
  for (int i = 0; i < 256; ++i) used[i] = 0;

  /* Map selected bytes to themselves. */
  for (int b = 33; b <= 126; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }
  for (int b = 161; b <= 172; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }
  for (int b = 174; b <= 255; ++b) { m->byte_to_cp[b] = (uint32_t)b; used[b] = 1; }

  /* Assign remaining bytes to codepoints starting at 256. */
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (!used[b]) {
      m->byte_to_cp[b] = (uint32_t)(256 + extra);
      extra++;
    }
  }

  /* Invert mapping for decoding. */
  for (int b = 0; b < 256; ++b) {
    uint32_t cp = m->byte_to_cp[b];
    if (cp < m->cp_to_byte_cap) m->cp_to_byte[cp] = b;
  }
}

/**
 * @brief Convert a unicode codepoint from bytes_to_unicode back to a byte.
 */
static int byteunicode_cp_to_byte(const byteunicode_t *m, uint32_t cp, uint8_t *out_b) {
  if (!m || !out_b) return -1;
  if (cp < m->cp_to_byte_cap && m->cp_to_byte[cp] >= 0) {
    *out_b = (uint8_t)m->cp_to_byte[cp];
    return 0;
  }
  return -1;
}

/* ============================================================================
 * Tokenizer state
 * ========================================================================== */

struct ie_tok_gptoss_s {
  uint32_t vocab_size;

  /* id -> token string (UTF-8). */
  char **id_to_tok;

  /* token string -> id (string pointers are owned by id_to_tok). */
  strid_ent_t *tok_to_id;
  size_t tok_to_id_cap;

  /* BPE merges: pair(sym_id_a, sym_id_b) -> rank (smaller rank = earlier merge). */
  pairrank_ent_t *pair_to_rank;
  size_t pair_to_rank_cap;

  /* Intern table for BPE symbol strings. */
  intern_t intern;

  /* Byte-level mapping. */
  byteunicode_t bu;
};

/* ============================================================================
 * tokenizer.json parsing
 * ========================================================================== */

/**
 * @brief Parse the vocab object (token -> id) from tokenizer.json.
 */
static int parse_vocab_object(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "vocab");
  if (!p) return IE_TOK_GPTOSS_ERR_JSON;

  p = skip_ws(p);
  if (*p != '{') return IE_TOK_GPTOSS_ERR_JSON;
  ++p;

  /* First pass: discover max id to size the arrays. */
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

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  /* Second pass: fill. */
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
    if (scan_json_int(&q, &id) != 0 || id < 0 || (uint32_t)id >= tok->vocab_size) {
      free(kraw);
      return IE_TOK_GPTOSS_ERR_JSON;
    }

    tok->id_to_tok[id] = kraw;
    (void)strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[id], (uint32_t)id);

    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  return IE_TOK_GPTOSS_OK;
}

/**
 * @brief Rebuild tok_to_id from id_to_tok after vocab expansion.
 */
static int rebuild_tok_to_id(ie_tok_gptoss_t *tok) {
  if (!tok) return -1;

  free(tok->tok_to_id);
  tok->tok_to_id = NULL;
  tok->tok_to_id_cap = 0;

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) return -1;

  for (size_t i = 0; i < tok->vocab_size; i++) {
    const char *s = tok->id_to_tok[i];
    if (!s) continue;
    if (strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, s, (uint32_t)i) != 0) {
      free(tok->tok_to_id);
      tok->tok_to_id = NULL;
      tok->tok_to_id_cap = 0;
      return -1;
    }
  }

  return 0;
}

/**
 * @brief Parse optional top-level "added_tokens" entries into vocab.
 */
static int parse_added_tokens_array(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "added_tokens");
  if (!p) return IE_TOK_GPTOSS_OK;

  p = skip_ws(p);
  if (!p || *p != '[') return IE_TOK_GPTOSS_ERR_JSON;
  p++;

  uint32_t max_id = (tok->vocab_size > 0) ? (uint32_t)(tok->vocab_size - 1) : 0;

  /* Pass 1: compute max id. */
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

  /* Expand vocab if needed. */
  const size_t need = (size_t)max_id + 1;
  if (need > tok->vocab_size) {
    const size_t old = tok->vocab_size;
    char **nt = (char **)realloc(tok->id_to_tok, need * sizeof(char *));
    if (!nt) return IE_TOK_GPTOSS_ERR_NOMEM;
    tok->id_to_tok = nt;
    for (size_t i = old; i < need; i++) tok->id_to_tok[i] = NULL;
    tok->vocab_size = (uint32_t)need;
    if (rebuild_tok_to_id(tok) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  /* Pass 2: insert tokens. */
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
      if (tok->id_to_tok[id] == NULL) {
        tok->id_to_tok[id] = content;
      } else {
        free(content);
      }
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

/**
 * @brief Scan a merge entry which can be either a string ("A B") or a pair array ["A","B"].
 */
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

/**
 * @brief Parse the merges array (pair ranks) from tokenizer.json.
 */
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

  if (intern_init(&tok->intern, merges * 4 + 1024, (size_t)(1u << 20)) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  if (pairrank_map_init(&tok->pair_to_rank, &tok->pair_to_rank_cap, merges) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

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
    if (intern_get_or_add(&tok->intern, lhs, &ida) != 0 ||
        intern_get_or_add(&tok->intern, rhs, &idb) != 0) {
      free(lhs);
      free(rhs);
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

/**
 * @brief Parse tokenizer.json into tokenizer state.
 */
static int parse_tokenizer_json(const char *json, ie_tok_gptoss_t *tok) {
  if (!json || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  int rc = parse_vocab_object(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  rc = parse_added_tokens_array(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  rc = parse_merges_array(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  return IE_TOK_GPTOSS_OK;
}

/* ============================================================================
 * Packed tokenizer format (IETOK1)
 * ========================================================================== */

/**
 * @brief Read a little-endian u32 from a byte buffer.
 */
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

/**
 * @brief Read a little-endian u16 from a byte buffer.
 */
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

/**
 * @brief Read a length-prefixed string from a packed tokenizer buffer.
 */
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

/**
 * @brief Parse packed tokenizer file produced by scripts/pack_tokenizer.py.
 */
static int parse_tokenizer_packed(const unsigned char *buf, size_t n, ie_tok_gptoss_t *tok) {
  if (!buf || n < 32 || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  /* Header layout matches scripts/pack_tokenizer.py:
     magic[6] = "IETOK1"
     u16 version
     u32 vocab_size
     u32 merges_count
     u32 off_vocab
     u32 off_merges
     u32 off_special
     u32 reserved
   */
  size_t off = 0;
  if (n < 6) return IE_TOK_GPTOSS_ERR_JSON;
  if (memcmp(buf, "IETOK1", 6) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  off += 6;

  uint16_t version = 0;
  if (rd_u16le(buf, n, &off, &version) != 0) return IE_TOK_GPTOSS_ERR_JSON;
  if (version != 1u) return IE_TOK_GPTOSS_ERR_JSON;

  uint32_t vocab_size = 0;
  uint32_t merges_count = 0;
  uint32_t off_vocab = 0;
  uint32_t off_merges = 0;
  uint32_t off_special = 0;
  uint32_t reserved = 0;

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

  if (strid_map_init(&tok->tok_to_id, &tok->tok_to_id_cap, tok->vocab_size) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  /* Vocab section. */
  off = (size_t)off_vocab;
  for (uint32_t i = 0; i < tok->vocab_size; ++i) {
    char *s = NULL;
    if (rd_lp_string(buf, n, &off, &s) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    tok->id_to_tok[i] = s;
    (void)strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[i], i);
  }

  /* Merges section. */
  if (intern_init(&tok->intern, (size_t)merges_count * 4 + 1024, (size_t)(1u << 20)) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }
  if (pairrank_map_init(&tok->pair_to_rank, &tok->pair_to_rank_cap, (size_t)merges_count) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  off = (size_t)off_merges;
  for (uint32_t r = 0; r < merges_count; ++r) {
    char *a = NULL;
    char *b = NULL;
    if (rd_lp_string(buf, n, &off, &a) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    if (rd_lp_string(buf, n, &off, &b) != 0) { free(a); return IE_TOK_GPTOSS_ERR_JSON; }

    uint32_t ida = 0, idb = 0;
    if (intern_get_or_add(&tok->intern, a, &ida) != 0 ||
        intern_get_or_add(&tok->intern, b, &idb) != 0) {
      free(a);
      free(b);
      return IE_TOK_GPTOSS_ERR_NOMEM;
    }

    uint64_t key = ((uint64_t)ida << 32) | (uint64_t)idb;
    (void)pairrank_map_put(tok->pair_to_rank, tok->pair_to_rank_cap, key, r);

    free(a);
    free(b);
  }

  return IE_TOK_GPTOSS_OK;
}

/**
 * @brief Detect whether a buffer is a packed tokenizer.
 */
static int is_packed_tokenizer(const unsigned char *buf, size_t n) {
  if (!buf || n < 6) return 0;
  return memcmp(buf, "IETOK1", 6) == 0;
}

/* ============================================================================
 * BPE encode
 * ========================================================================== */

/**
 * @brief Simple growable u32 array.
 */
typedef struct u32buf_s {
  uint32_t *v;
  size_t n;
  size_t cap;
} u32buf_t;

/**
 * @brief Push into u32 buffer.
 */
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

/**
 * @brief Simple growable byte buffer.
 */
typedef struct strbuf_s {
  char *v;
  size_t n;
  size_t cap;
} strbuf_t;

/**
 * @brief Ensure capacity for additional bytes.
 */
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

/**
 * @brief Append raw bytes.
 */
static int strbuf_append_bytes(strbuf_t *b, const char *s, size_t n) {
  if (!b || (!s && n)) return -1;
  if (strbuf_reserve(b, n) != 0) return -1;
  memcpy(b->v + b->n, s, n);
  b->n += n;
  return 0;
}

/**
 * @brief Append a C string.
 */
static int strbuf_append_cstr(strbuf_t *b, const char *s) {
  return strbuf_append_bytes(b, s, strlen(s));
}

/**
 * @brief Free a strbuf.
 */
static void strbuf_free(strbuf_t *b) {
  if (!b) return;
  free(b->v);
  memset(b, 0, sizeof(*b));
}

/**
 * @brief Convert raw bytes to byte-level unicode string (GPT-2 bytes_to_unicode).
 */
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

/**
 * @brief Encode one BPE token (byte-level unicode) into token ids.
 *
 * This performs the GPT-2 BPE merges and emits ids for each final symbol.
 */
static int bpe_encode_to_ids(const ie_tok_gptoss_t *tok, const char *in, u32buf_t *out_ids) {
  if (!tok || !in || !out_ids) return -1;

  typedef struct sym_s { const char *s; uint32_t id; } sym_t;

  const size_t in_len = strlen(in);
  size_t i = 0;

  sym_t *syms = NULL;
  size_t nsyms = 0;
  size_t cap = 0;

  strbuf_t scratch = {0};

  /* Split input into initial symbols (one UTF-8 codepoint each). */
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
    if (intern_get_or_add((intern_t *)&tok->intern, dst, &sid) != 0) {
      strbuf_free(&scratch);
      free(syms);
      return -1;
    }
    syms[nsyms].s = dst;
    syms[nsyms].id = sid;
    nsyms++;
  }

  if (nsyms == 0) {
    strbuf_free(&scratch);
    free(syms);
    return 0;
  }

  /* Repeatedly merge the best-ranked adjacent pair. */
  for (;;) {
    uint32_t best_rank = UINT32_MAX;
    size_t best_i = (size_t)-1;

    for (size_t k = 0; k + 1 < nsyms; ++k) {
      uint64_t key = ((uint64_t)syms[k].id << 32) | (uint64_t)syms[k + 1].id;
      uint32_t r = 0;
      if (pairrank_map_get(tok->pair_to_rank, tok->pair_to_rank_cap, key, &r) == 0) {
        if (r < best_rank) {
          best_rank = r;
          best_i = k;
        }
      }
    }

    if (best_i == (size_t)-1) break;

    const char *a = syms[best_i].s;
    const char *b = syms[best_i + 1].s;

    strbuf_t merged = {0};
    if (strbuf_append_cstr(&merged, a) != 0 || strbuf_append_cstr(&merged, b) != 0) {
      strbuf_free(&merged);
      strbuf_free(&scratch);
      free(syms);
      return -1;
    }

    if (strbuf_reserve(&scratch, merged.n + 1) != 0) {
      strbuf_free(&merged);
      strbuf_free(&scratch);
      free(syms);
      return -1;
    }
    char *dst = scratch.v + scratch.n;
    memcpy(dst, merged.v, merged.n);
    dst[merged.n] = '\0';
    scratch.n += merged.n + 1;
    strbuf_free(&merged);

    uint32_t sid = 0;
    if (intern_get_or_add((intern_t *)&tok->intern, dst, &sid) != 0) {
      strbuf_free(&scratch);
      free(syms);
      return -1;
    }

    syms[best_i].s = dst;
    syms[best_i].id = sid;

    for (size_t k = best_i + 1; k + 1 < nsyms; ++k) syms[k] = syms[k + 1];
    nsyms--;
  }

  /* Emit ids for each final symbol. */
  for (size_t k = 0; k < nsyms; ++k) {
    const char *sym = syms[k].s;
    uint32_t id = 0;
    if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, sym, &id) == 0) {
      if (u32buf_push(out_ids, id) != 0) { strbuf_free(&scratch); free(syms); return -1; }
      continue;
    }

    /* Fallback: split symbol into codepoints and lookup each.
       This should be rare if vocab and merges are consistent. */
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

/**
 * @brief Pretokenize text into segments.
 *
 * This is a lightweight approximation of the GPT-2 pretokenizer:
 *  - Whitespace runs are preserved.
 *  - For a non-whitespace token that follows whitespace, one leading space is
 *    attached to the token when possible.
 */
static int pretokenize_basic(const char *text, char ***out_parts, size_t *out_n) {
  if (!text || !out_parts || !out_n) return -1;
  *out_parts = NULL;
  *out_n = 0;

  size_t cap = 0;
  size_t n = 0;
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
        parts = np;
        cap = nc;
      }
      parts[n++] = seg;
      continue;
    }

    /* Non-whitespace run. */
    const char *s = p;
    while (*p && !isspace((unsigned char)*p)) ++p;
    size_t len = (size_t)(p - s);

    /* Attach a single leading space from the previous whitespace segment, if any. */
    int have_lead_space = 0;
    if (n > 0 && parts[n - 1] && parts[n - 1][0] != '\0') {
      char *w = parts[n - 1];
      size_t wl = strlen(w);
      if (wl > 0 && w[wl - 1] == ' ') {
        w[wl - 1] = '\0';
        have_lead_space = 1;
      }
    }

    strbuf_t sb = {0};
    if (have_lead_space) {
      const char sp = ' ';
      if (strbuf_append_bytes(&sb, &sp, 1) != 0) { strbuf_free(&sb); goto fail; }
    }
    if (strbuf_append_bytes(&sb, s, len) != 0) { strbuf_free(&sb); goto fail; }
    if (strbuf_reserve(&sb, 1) != 0) { strbuf_free(&sb); goto fail; }
    sb.v[sb.n] = '\0';

    if (n == cap) {
      size_t nc = cap ? cap * 2 : 64;
      char **np = (char **)realloc(parts, nc * sizeof(*np));
      if (!np) { strbuf_free(&sb); goto fail; }
      parts = np;
      cap = nc;
    }
    parts[n++] = sb.v;
  }

  /* Drop any empty whitespace segments created by space stealing. */
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

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t id = ids[i];
    if (id >= tok->vocab_size) return IE_TOK_GPTOSS_ERR_RANGE;
    const char *t = tok->id_to_tok[id];
    if (!t) t = "";

    const size_t tlen = strlen(t);
    size_t pos = 0;
    while (pos < tlen) {
      uint32_t cp = 0;
      int st = utf8_next_cp(t, tlen, &pos, &cp);
      if (st != 0) return IE_TOK_GPTOSS_ERR_INTERNAL;

      uint8_t b = 0;
      if (byteunicode_cp_to_byte(&tok->bu, cp, &b) == 0) {
        char c = (char)b;
        if (strbuf_append_bytes(&bytes, &c, 1) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
      } else {
        char tmp[4];
        size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
        if (wn == 0) return IE_TOK_GPTOSS_ERR_INTERNAL;
        if (strbuf_append_bytes(&bytes, tmp, wn) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
      }
    }
  }

  if (strbuf_reserve(&bytes, 1) != 0) { strbuf_free(&bytes); return IE_TOK_GPTOSS_ERR_NOMEM; }
  bytes.v[bytes.n] = '\0';

  size_t need = bytes.n + 1;
  if (!out || *inout_bytes == 0) {
    *inout_bytes = need;
    strbuf_free(&bytes);
    return IE_TOK_GPTOSS_OK;
  }

  if (*inout_bytes < need) {
    *inout_bytes = need;
    strbuf_free(&bytes);
    return IE_TOK_GPTOSS_ERR_RANGE;
  }

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

  char **parts = NULL;
  size_t nparts = 0;
  if (pretokenize_basic(text, &parts, &nparts) != 0) return IE_TOK_GPTOSS_ERR_INTERNAL;

  u32buf_t out_ids = {0};

  for (size_t i = 0; i < nparts; ++i) {
    const char *seg = parts[i];
    if (!seg || !*seg) { free(parts[i]); continue; }

    /* Convert segment bytes to byte-level unicode. */
    strbuf_t bl = {0};
    if (to_bytelevel_unicode(&tok->bu, seg, strlen(seg), &bl) != 0) { strbuf_free(&bl); goto fail; }
    if (strbuf_reserve(&bl, 1) != 0) { strbuf_free(&bl); goto fail; }
    bl.v[bl.n] = '\0';

    /* Apply BPE and emit ids. */
    if (bpe_encode_to_ids(tok, bl.v, &out_ids) != 0) { strbuf_free(&bl); goto fail; }
    strbuf_free(&bl);

    free(parts[i]);
  }

  free(parts);

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

  memcpy(ids, out_ids.v, need * sizeof(uint32_t));
  *inout_count = need;
  free(out_ids.v);
  return IE_TOK_GPTOSS_OK;

fail:
  for (size_t i = 0; i < nparts; ++i) free(parts[i]);
  free(parts);
  free(out_ids.v);
  return IE_TOK_GPTOSS_ERR_INTERNAL;
}
