#define _POSIX_C_SOURCE 200809L

#include "tokenizer_hf.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Small helpers                                                             */
/* ========================================================================= */

static const char *skip_ws(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  return p;
}

static int read_file(const char *path, char **out, size_t *sz) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) { fclose(f); return -1; }
  char *buf = malloc((size_t)n + 1);
  if (!buf) { fclose(f); return -1; }
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
    free(buf);
    fclose(f);
    return -1;
  }
  fclose(f);
  buf[n] = 0;
  *out = buf;
  if (sz) *sz = (size_t)n;
  return 0;
}

/* ========================================================================= */
/* JSON string parsing (minimal, safe)                                       */
/* ========================================================================= */

static const char *parse_string(const char *p, char **out) {
  if (*p != '"') return NULL;
  ++p;

  size_t cap = 64, len = 0;
  char *buf = malloc(cap);
  if (!buf) return NULL;

  while (*p) {
    if (*p == '"') {
      buf[len] = 0;
      *out = buf;
      return p + 1;
    }
    if (*p == '\\') {
      ++p;
      if (!*p) break;
      char c = *p;
      if (c == 'n') c = '\n';
      else if (c == 't') c = '\t';
      else if (c == 'r') c = '\r';
      if (len + 1 >= cap) {
        cap *= 2;
        buf = realloc(buf, cap);
        if (!buf) return NULL;
      }
      buf[len++] = c;
      ++p;
      continue;
    }
    if (len + 1 >= cap) {
      cap *= 2;
      buf = realloc(buf, cap);
      if (!buf) return NULL;
    }
    buf[len++] = *p++;
  }

  free(buf);
  return NULL;
}

static const char *parse_u32(const char *p, uint32_t *out) {
  uint64_t v = 0;
  if (!isdigit((unsigned char)*p)) return NULL;
  while (isdigit((unsigned char)*p)) {
    v = v * 10 + (uint64_t)(*p - '0');
    if (v > 0xFFFFFFFFu) return NULL;
    ++p;
  }
  *out = (uint32_t)v;
  return p;
}

/* ========================================================================= */
/* GPT-2 byte decoder                                                        */
/* ========================================================================= */

typedef struct {
  uint32_t key;
  uint8_t  val;
} byte_map_entry_t;

static byte_map_entry_t gpt2_map[256];

static void build_gpt2_byte_decoder(void) {
  static int built = 0;
  if (built) return;
  built = 1;

  int used[256] = {0};
  int n = 0;

  for (int i = 33; i <= 126; ++i) {
    gpt2_map[n++] = (byte_map_entry_t){ (uint32_t)i, (uint8_t)i };
    used[i] = 1;
  }
  for (int i = 161; i <= 172; ++i) {
    gpt2_map[n++] = (byte_map_entry_t){ (uint32_t)i, (uint8_t)i };
    used[i] = 1;
  }
  for (int i = 174; i <= 255; ++i) {
    gpt2_map[n++] = (byte_map_entry_t){ (uint32_t)i, (uint8_t)i };
    used[i] = 1;
  }

  uint32_t extra = 0;
  for (int i = 0; i < 256; ++i) {
    if (!used[i]) {
      gpt2_map[n++] = (byte_map_entry_t){ 256u + extra, (uint8_t)i };
      ++extra;
    }
  }
}

static int gpt2_cp_to_byte(uint32_t cp, uint8_t *out) {
  for (int i = 0; i < 256; ++i) {
    if (gpt2_map[i].key == cp) {
      *out = gpt2_map[i].val;
      return 1;
    }
  }
  return 0;
}

/* ========================================================================= */
/* Public API                                                                */
/* ========================================================================= */

int tokenizer_hf_load(const char *path, tokenizer_hf_t *out) {
  if (!path || !out) return -1;
  memset(out, 0, sizeof(*out));

  char *json = NULL;
  if (read_file(path, &json, NULL) != 0) return -1;

  const char *p = strstr(json, "\"vocab\"");
  if (!p) { free(json); return -1; }

  p = strchr(p, '{');
  if (!p) { free(json); return -1; }
  ++p;

  uint32_t max_id = 0;

  const char *scan = p;
  while (*scan && *scan != '}') {
    char *tok = NULL;
    scan = skip_ws(scan);
    scan = parse_string(scan, &tok);
    if (!scan) break;
    scan = skip_ws(scan);
    if (*scan != ':') { free(tok); break; }
    ++scan;
    scan = skip_ws(scan);

    uint32_t id = 0;
    scan = parse_u32(scan, &id);
    if (!scan) { free(tok); break; }

    if (id > max_id) max_id = id;
    free(tok);

    scan = skip_ws(scan);
    if (*scan == ',') ++scan;
  }

  out->vocab_size = max_id + 1;
  out->id_to_token = calloc(out->vocab_size, sizeof(char*));
  if (!out->id_to_token) { free(json); return -1; }

  scan = p;
  while (*scan && *scan != '}') {
    char *tok = NULL;
    scan = skip_ws(scan);
    scan = parse_string(scan, &tok);
    if (!scan) break;
    scan = skip_ws(scan);
    if (*scan != ':') { free(tok); break; }
    ++scan;
    scan = skip_ws(scan);

    uint32_t id = 0;
    scan = parse_u32(scan, &id);
    if (!scan) { free(tok); break; }

    if (id < out->vocab_size) out->id_to_token[id] = tok;
    else free(tok);

    scan = skip_ws(scan);
    if (*scan == ',') ++scan;
  }

  free(json);
  build_gpt2_byte_decoder();
  out->loaded = 1;
  return 0;
}

void tokenizer_hf_free(tokenizer_hf_t *tok) {
  if (!tok) return;
  for (uint32_t i = 0; i < tok->vocab_size; ++i)
    free(tok->id_to_token[i]);
  free(tok->id_to_token);
  memset(tok, 0, sizeof(*tok));
}

int tokenizer_hf_decode(const tokenizer_hf_t *tok,
                         const int *ids,
                         size_t n_ids,
                         char *out,
                         size_t out_sz) {
  if (!tok || !tok->loaded || !out || out_sz == 0) return -1;
  size_t w = 0;
  out[0] = 0;

  for (size_t i = 0; i < n_ids; ++i) {
    int id = ids[i];
    if (id < 0 || (uint32_t)id >= tok->vocab_size) continue;
    const char *s = tok->id_to_token[id];
    if (!s) continue;

    while (*s) {
      uint32_t cp = (unsigned char)*s++;
      uint8_t b;
      if (gpt2_cp_to_byte(cp, &b)) {
        if (w + 1 >= out_sz) return -1;
        out[w++] = (char)b;
      } else {
        if (w + 1 >= out_sz) return -1;
        out[w++] = (char)cp;
      }
    }
  }

  out[w] = 0;
  return 0;
}
