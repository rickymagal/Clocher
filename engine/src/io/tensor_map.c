#define _POSIX_C_SOURCE 200809L

#include "tensor_map.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Helpers                                                                   */
/* ========================================================================= */

static const char *skip_ws(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  return p;
}

static int read_file(const char *path, char **out) {
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
  return 0;
}

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

static const char *parse_u64(const char *p, uint64_t *out) {
  uint64_t v = 0;
  if (!isdigit((unsigned char)*p)) return NULL;
  while (isdigit((unsigned char)*p)) {
    v = v * 10 + (uint64_t)(*p - '0');
    ++p;
  }
  *out = v;
  return p;
}

static tensor_dtype_t parse_dtype(const char *s) {
  if (!s) return TENSOR_DTYPE_UNKNOWN;
  if (strcmp(s, "float32") == 0) return TENSOR_DTYPE_F32;
  if (strcmp(s, "float16") == 0) return TENSOR_DTYPE_F16;
  if (strcmp(s, "int4") == 0)    return TENSOR_DTYPE_INT4;
  return TENSOR_DTYPE_UNKNOWN;
}

/* ========================================================================= */
/* Loader                                                                    */
/* ========================================================================= */

int tensor_map_load(const char *path, tensor_map_t *out) {
  if (!path || !out) return -1;
  memset(out, 0, sizeof(*out));

  char *json = NULL;
  if (read_file(path, &json) != 0) return -1;

  const char *p = skip_ws(json);
  if (*p != '{') { free(json); return -1; }
  ++p;

  /* First pass: count tensors */
  uint32_t count = 0;
  const char *scan = p;
  while (*scan) {
    scan = skip_ws(scan);
    if (*scan == '"') {
      char *tmp = NULL;
      scan = parse_string(scan, &tmp);
      free(tmp);
      if (!scan) break;
      scan = skip_ws(scan);
      if (*scan == ':') {
        ++count;
      }
    }
    ++scan;
  }

  if (count == 0) { free(json); return -1; }

  tensor_desc_t *arr = calloc(count, sizeof(tensor_desc_t));
  if (!arr) { free(json); return -1; }

  /* Second pass: parse tensors */
  uint32_t idx = 0;
  scan = p;

  while (*scan && idx < count) {
    scan = skip_ws(scan);
    if (*scan != '"') { ++scan; continue; }

    char *name = NULL;
    scan = parse_string(scan, &name);
    if (!scan) break;

    scan = skip_ws(scan);
    if (*scan != ':') { free(name); break; }
    ++scan;

    scan = skip_ws(scan);
    if (*scan != '{') { free(name); break; }
    ++scan;

    tensor_desc_t *td = &arr[idx];
    td->name = name;

    while (*scan && *scan != '}') {
      scan = skip_ws(scan);

      char *key = NULL;
      scan = parse_string(scan, &key);
      if (!scan) break;

      scan = skip_ws(scan);
      if (*scan != ':') { free(key); break; }
      ++scan;
      scan = skip_ws(scan);

      if (strcmp(key, "offset") == 0) {
        scan = parse_u64(scan, &td->offset);
      } else if (strcmp(key, "size_bytes") == 0) {
        scan = parse_u64(scan, &td->size_bytes);
      } else if (strcmp(key, "dtype") == 0) {
        char *ds = NULL;
        scan = parse_string(scan, &ds);
        td->dtype = parse_dtype(ds);
        free(ds);
      } else if (strcmp(key, "shape") == 0) {
        if (*scan != '[') { free(key); break; }
        ++scan;

        uint32_t dims[16];
        uint32_t nd = 0;

        while (*scan && *scan != ']') {
          scan = skip_ws(scan);
          uint64_t v = 0;
          scan = parse_u64(scan, &v);
          if (!scan) break;
          if (nd < 16) dims[nd++] = (uint32_t)v;
          scan = skip_ws(scan);
          if (*scan == ',') ++scan;
        }
        if (*scan == ']') ++scan;

        td->shape = malloc(sizeof(uint32_t) * nd);
        if (td->shape) {
          memcpy(td->shape, dims, sizeof(uint32_t) * nd);
          td->ndim = nd;
        }
      }

      free(key);
      scan = skip_ws(scan);
      if (*scan == ',') ++scan;
    }

    if (*scan == '}') ++scan;
    ++idx;
  }

  free(json);

  out->tensors = arr;
  out->count   = idx;
  out->loaded  = 1;
  return 0;
}

/* ========================================================================= */
/* Free                                                                      */
/* ========================================================================= */

void tensor_map_free(tensor_map_t *map) {
  if (!map || !map->tensors) return;
  for (uint32_t i = 0; i < map->count; ++i) {
    free(map->tensors[i].name);
    free(map->tensors[i].shape);
  }
  free(map->tensors);
  memset(map, 0, sizeof(*map));
}

/* ========================================================================= */
/* Lookup                                                                    */
/* ========================================================================= */

const tensor_desc_t *tensor_map_find(const tensor_map_t *map, const char *name) {
  if (!map || !map->loaded || !name) return NULL;
  for (uint32_t i = 0; i < map->count; ++i) {
    if (strcmp(map->tensors[i].name, name) == 0)
      return &map->tensors[i];
  }
  return NULL;
}
