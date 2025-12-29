/* engine/src/io/tokenizer_gptoss.c */
/**
 * @file tokenizer_gptoss.c
 * @brief GPT-OSS (GPT-2 style) Byte-Level BPE tokenizer (tokenizer.json loader).
 *
 * This implementation supports:
 *  - Loading `model.vocab` (token string -> id) from tokenizer.json
 *  - Loading `model.merges` (pair ranks) from tokenizer.json
 *  - Byte-level (bytes_to_unicode) mapping used by GPT-2 BPE
 *  - Decode: token ids -> token strings -> bytes -> UTF-8
 *  - Encode: basic pretokenization + byte-level mapping + BPE merges + vocab lookup
 *
 * Limitations (intentional for a first pass):
 *  - Pretokenization is simplified (space-aware, preserves runs of whitespace).
 *    If you need exact HF parity, implement the HF GPT-2 regex pretokenizer.
 *  - Special tokens handling is minimal (we keep them as their literal strings).
 */

#include "ie_tokenizer_gptoss.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
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

static int match_lit(const char **pp, const char *lit) {
  const char *p = *pp;
  size_t n = strlen(lit);
  if (strncmp(p, lit, n) != 0) return 0;
  *pp = p + n;
  return 1;
}

static int scan_json_string(const char **pp, char **out_s, size_t *out_n) {
  const char *p = skip_ws(*pp);
  if (*p != '"') return -1;
  ++p;

  const char *start = p;
  size_t len = 0;

  while (*p) {
    if (*p == '\\') {
      ++p;
      if (!*p) return -1;
      ++p;
      len += 2;
      continue;
    }
    if (*p == '"') break;
    ++p;
    ++len;
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
        /* Minimal \uXXXX support (BMP only, no surrogate pairs). */
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

/* ============================================================================
 * Hash maps: string->id and pair->rank
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
    if (m[i].h == h && strcmp(m[i].k, k) == 0) {
      m[i].v = v;
      return 0;
    }
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
    if (m[i].h == h && strcmp(m[i].k, k) == 0) {
      *out_v = m[i].v;
      return 0;
    }
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
  in->arena_sz = (arena_hint ? arena_hint : 1u << 20);
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

static void intern_free(intern_t *in) {
  if (!in) return;
  free(in->m);
  free(in->arena);
  memset(in, 0, sizeof(*in));
}

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

typedef struct byteunicode_s {
  uint32_t byte_to_cp[256];
  int32_t cp_to_byte[2048]; /* covers the codepoints we generate (<= 256 + 255) plus ASCII */
  uint32_t cp_to_byte_cap;
} byteunicode_t;

static void byteunicode_init(byteunicode_t *m) {
  memset(m, 0, sizeof(*m));
  for (size_t i = 0; i < 2048; ++i) m->cp_to_byte[i] = -1;
  m->cp_to_byte_cap = 2048;

  int used[256];
  for (int i = 0; i < 256; ++i) used[i] = 0;

  int out = 0;
  for (int b = 33; b <= 126; ++b) { used[b] = 1; m->byte_to_cp[out++] = (uint32_t)b; }
  for (int b = 161; b <= 172; ++b) { used[b] = 1; m->byte_to_cp[out++] = (uint32_t)b; }
  for (int b = 174; b <= 255; ++b) { used[b] = 1; m->byte_to_cp[out++] = (uint32_t)b; }

  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    if (!used[b]) {
      m->byte_to_cp[out++] = (uint32_t)(256 + extra);
      extra++;
    }
  }

  /* Invert. */
  for (int b = 0; b < 256; ++b) {
    uint32_t cp = m->byte_to_cp[b];
    if (cp < m->cp_to_byte_cap) m->cp_to_byte[cp] = b;
  }
}

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

    /* Store token string owned by id_to_tok. */
    tok->id_to_tok[id] = kraw;
    (void)strid_map_put(tok->tok_to_id, tok->tok_to_id_cap, tok->id_to_tok[id], (uint32_t)id);

    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  return IE_TOK_GPTOSS_OK;
}

static int parse_merges_array(const char *json, ie_tok_gptoss_t *tok) {
  const char *p = find_key_in_object(json, "merges");
  if (!p) return IE_TOK_GPTOSS_ERR_JSON;

  p = skip_ws(p);
  if (*p != '[') return IE_TOK_GPTOSS_ERR_JSON;
  ++p;

  /* Count merges (approx). */
  size_t merges = 0;
  const char *q = p;
  while (*q) {
    q = skip_ws(q);
    if (*q == ']') break;
    if (*q != '"') return IE_TOK_GPTOSS_ERR_JSON;
    char *sraw = NULL;
    if (scan_json_string(&q, &sraw, NULL) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    free(sraw);
    merges++;
    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  if (intern_init(&tok->intern, merges * 4 + 1024, (size_t)(1u << 20)) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  if (pairrank_map_init(&tok->pair_to_rank, &tok->pair_to_rank_cap, merges) != 0) {
    return IE_TOK_GPTOSS_ERR_NOMEM;
  }

  /* Parse again and insert pairs. */
  q = p;
  uint32_t rank = 0;
  while (*q) {
    q = skip_ws(q);
    if (*q == ']') break;

    char *sraw = NULL;
    if (scan_json_string(&q, &sraw, NULL) != 0) return IE_TOK_GPTOSS_ERR_JSON;
    if (json_unescape_inplace(sraw) != 0) { free(sraw); return IE_TOK_GPTOSS_ERR_JSON; }

    /* Merge line format: "A B" */
    char *sp = strchr(sraw, ' ');
    if (!sp) { free(sraw); return IE_TOK_GPTOSS_ERR_JSON; }
    *sp = '\0';
    const char *a = sraw;
    const char *b = sp + 1;

    uint32_t ida = 0, idb = 0;
    if (intern_get_or_add(&tok->intern, a, &ida) != 0 ||
        intern_get_or_add(&tok->intern, b, &idb) != 0) {
      free(sraw);
      return IE_TOK_GPTOSS_ERR_NOMEM;
    }

    uint64_t key = ((uint64_t)ida << 32) | (uint64_t)idb;
    (void)pairrank_map_put(tok->pair_to_rank, tok->pair_to_rank_cap, key, rank++);

    free(sraw);

    q = skip_ws(q);
    if (*q == ',') ++q;
  }

  return IE_TOK_GPTOSS_OK;
}

static int parse_tokenizer_json(const char *json, ie_tok_gptoss_t *tok) {
  if (!json || !tok) return IE_TOK_GPTOSS_ERR_ARGS;

  /* tokenizer.json has "model": { "type": "...", "vocab": {...}, "merges": [...] }.
     We scan globally for "vocab" and "merges" keys to avoid deep object parsing. */
  int rc = parse_vocab_object(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  rc = parse_merges_array(json, tok);
  if (rc != IE_TOK_GPTOSS_OK) return rc;

  return IE_TOK_GPTOSS_OK;
}

/* ============================================================================
 * BPE encode (minimal)
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

static int strbuf_append_cstr(strbuf_t *b, const char *s) {
  return strbuf_append_bytes(b, s, strlen(s));
}

static void strbuf_free(strbuf_t *b) {
  if (!b) return;
  free(b->v);
  memset(b, 0, sizeof(*b));
}

/* Convert raw UTF-8 token piece to "byte-level" unicode string (GPT-2 bytes_to_unicode). */
static int to_bytelevel_unicode(const byteunicode_t *bu, const char *s, strbuf_t *out) {
  if (!bu || !s || !out) return -1;
  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    uint8_t b = *p++;
    uint32_t cp = bu->byte_to_cp[b];
    char tmp[4];
    size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
    if (wn == 0) return -1;
    if (strbuf_append_bytes(out, tmp, wn) != 0) return -1;
  }
  return 0;
}

/* BPE merge: input is a bytelevel unicode string; we treat it as a sequence of UTF-8 codepoints,
   but BPE symbols in GPT-2 are single "characters" (codepoints) initially. */
static int bpe_apply(const ie_tok_gptoss_t *tok, const char *in, strbuf_t *out_joined) {
  if (!tok || !in || !out_joined) return -1;

  /* Step 1: split into initial symbols = each codepoint encoded as UTF-8 string. */
  const size_t in_len = strlen(in);
  size_t i = 0;

  /* Store symbol strings in a temporary array (char* owned by a scratch arena). */
  typedef struct sym_s { const char *s; uint32_t id; } sym_t;
  sym_t *syms = NULL;
  size_t nsyms = 0, cap = 0;

  strbuf_t scratch = {0};

  while (i < in_len) {
    uint32_t cp = 0;
    size_t old = i;
    if (utf8_next_cp(in, in_len, &i, &cp) != 0) { strbuf_free(&scratch); free(syms); return -1; }

    /* Re-encode this codepoint as UTF-8 slice from input (fast). */
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
    return 0;
  }

  /* Step 2: repeatedly merge the best-ranked pair. */
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

    /* Merge syms[best_i] + syms[best_i+1] into a new symbol string. */
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

    /* Remove syms[best_i+1] by shifting left. */
    for (size_t k = best_i + 1; k + 1 < nsyms; ++k) syms[k] = syms[k + 1];
    nsyms--;
  }

  /* Step 3: join symbols with no separator (GPT-2 BPE output tokens). */
  for (size_t k = 0; k < nsyms; ++k) {
    if (strbuf_append_cstr(out_joined, syms[k].s) != 0) {
      strbuf_free(&scratch);
      free(syms);
      return -1;
    }
    if (k + 1 < nsyms) {
      /* Separator between final BPE tokens is implicit: we return concatenated string,
         and the caller will split using interned symbol boundaries by running vocab lookup
         over each final symbol. For simplicity, we will not emit spaces here. */
      /* No-op. */
    }
  }

  strbuf_free(&scratch);
  free(syms);
  return 0;
}

/* Simplified pretokenizer:
   - Preserves whitespace as separate segments.
   - For non-whitespace runs, if preceded by a space, we prefix U+0120 (Ġ) like GPT-2.
   This is not exact HF behavior, but good enough for interactive CLI sanity. */
static int pretokenize_basic(const char *text, char ***out_parts, size_t *out_n) {
  if (!text || !out_parts || !out_n) return -1;
  *out_parts = NULL;
  *out_n = 0;

  size_t cap = 0;
  size_t n = 0;
  char **parts = NULL;

  const char *p = text;
  int prev_was_space = 1;

  while (*p) {
    if (isspace((unsigned char)*p)) {
      const char *s = p;
      while (*p && isspace((unsigned char)*p)) ++p;
      size_t len = (size_t)(p - s);

      char *seg = xstrdup_n(s, len);
      if (!seg) { for (size_t i = 0; i < n; ++i) free(parts[i]); free(parts); return -1; }

      if (n == cap) {
        size_t nc = cap ? cap * 2 : 64;
        char **np = (char **)realloc(parts, nc * sizeof(*np));
        if (!np) { free(seg); for (size_t i = 0; i < n; ++i) free(parts[i]); free(parts); return -1; }
        parts = np;
        cap = nc;
      }
      parts[n++] = seg;
      prev_was_space = 1;
      continue;
    }

    const char *s = p;
    while (*p && !isspace((unsigned char)*p)) ++p;
    size_t len = (size_t)(p - s);

    strbuf_t sb = {0};
    if (prev_was_space) {
      const uint32_t cp = 0x0120; /* Ġ */
      char tmp[4];
      size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
      if (wn == 0 || strbuf_append_bytes(&sb, tmp, wn) != 0) { strbuf_free(&sb); goto fail; }
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
    prev_was_space = 0;
  }

  *out_parts = parts;
  *out_n = n;
  return 0;

fail:
  for (size_t i = 0; i < n; ++i) free(parts[i]);
  free(parts);
  return -1;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

int ie_tok_gptoss_open(const char *tokenizer_json_path, ie_tok_gptoss_t **out_tok) {
  if (!tokenizer_json_path || !out_tok) return IE_TOK_GPTOSS_ERR_ARGS;
  *out_tok = NULL;

  char *json = NULL;
  size_t json_len = 0;
  if (read_all_bytes(tokenizer_json_path, &json, &json_len) != 0) return IE_TOK_GPTOSS_ERR_IO;

  ie_tok_gptoss_t *tok = (ie_tok_gptoss_t *)xcalloc(1, sizeof(*tok));
  if (!tok) { free(json); return IE_TOK_GPTOSS_ERR_NOMEM; }

  byteunicode_init(&tok->bu);

  int rc = parse_tokenizer_json(json, tok);
  free(json);

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

  /* Decode pipeline:
     ids -> token strings (UTF-8 with byte-level unicode chars) -> codepoints -> bytes -> UTF-8 */
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
        /* Not in byte-level map: keep literal UTF-8 bytes of this codepoint. */
        char tmp[4];
        size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
        if (wn == 0) return IE_TOK_GPTOSS_ERR_INTERNAL;
        if (strbuf_append_bytes(&bytes, tmp, wn) != 0) return IE_TOK_GPTOSS_ERR_NOMEM;
      }
    }
  }

  /* Ensure NUL. */
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

    if (isspace((unsigned char)seg[0])) {
      /* Whitespace segment: encode literally (byte-level + BPE). */
      strbuf_t bl = {0};
      if (to_bytelevel_unicode(&tok->bu, seg, &bl) != 0) { strbuf_free(&bl); goto fail; }
      if (strbuf_reserve(&bl, 1) != 0) { strbuf_free(&bl); goto fail; }
      bl.v[bl.n] = '\0';

      /* Apply BPE. */
      strbuf_t bpe = {0};
      if (bpe_apply(tok, bl.v, &bpe) != 0) { strbuf_free(&bl); strbuf_free(&bpe); goto fail; }
      strbuf_free(&bl);

      /* Vocab lookup: we attempt the whole string; if missing, fallback to per-codepoint. */
      if (strbuf_reserve(&bpe, 1) != 0) { strbuf_free(&bpe); goto fail; }
      bpe.v[bpe.n] = '\0';

      uint32_t id = 0;
      if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, bpe.v, &id) == 0) {
        if (u32buf_push(&out_ids, id) != 0) { strbuf_free(&bpe); goto fail; }
      } else {
        /* Per-codepoint fallback. */
        size_t blen = bpe.n;
        size_t pos = 0;
        while (pos < blen) {
          size_t old = pos;
          uint32_t cp = 0;
          if (utf8_next_cp(bpe.v, blen, &pos, &cp) != 0) break;
          char tmp[4];
          size_t wn = utf8_put_cp(tmp, sizeof(tmp), cp);
          if (wn == 0) break;
          char *one = xstrdup_n(bpe.v + old, pos - old);
          if (!one) { strbuf_free(&bpe); goto fail; }
          if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, one, &id) == 0) {
            if (u32buf_push(&out_ids, id) != 0) { free(one); strbuf_free(&bpe); goto fail; }
          }
          free(one);
        }
      }
      strbuf_free(&bpe);
    } else {
      /* Non-whitespace segment. */
      strbuf_t bl = {0};
      if (to_bytelevel_unicode(&tok->bu, seg, &bl) != 0) { strbuf_free(&bl); goto fail; }
      if (strbuf_reserve(&bl, 1) != 0) { strbuf_free(&bl); goto fail; }
      bl.v[bl.n] = '\0';

      strbuf_t bpe = {0};
      if (bpe_apply(tok, bl.v, &bpe) != 0) { strbuf_free(&bl); strbuf_free(&bpe); goto fail; }
      strbuf_free(&bl);

      if (strbuf_reserve(&bpe, 1) != 0) { strbuf_free(&bpe); goto fail; }
      bpe.v[bpe.n] = '\0';

      uint32_t id = 0;
      if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, bpe.v, &id) == 0) {
        if (u32buf_push(&out_ids, id) != 0) { strbuf_free(&bpe); goto fail; }
      } else {
        /* Per-codepoint fallback. */
        size_t blen = bpe.n;
        size_t pos = 0;
        while (pos < blen) {
          size_t old = pos;
          uint32_t cp = 0;
          if (utf8_next_cp(bpe.v, blen, &pos, &cp) != 0) break;
          uint32_t tid = 0;
          char *one = xstrdup_n(bpe.v + old, pos - old);
          if (!one) { strbuf_free(&bpe); goto fail; }
          if (strid_map_get(tok->tok_to_id, tok->tok_to_id_cap, one, &tid) == 0) {
            if (u32buf_push(&out_ids, tid) != 0) { free(one); strbuf_free(&bpe); goto fail; }
          }
          free(one);
        }
      }
      strbuf_free(&bpe);
    }

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
