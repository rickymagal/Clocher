#define _POSIX_C_SOURCE 200809L

#include "tensor_map.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Minimal JSON helpers (string/number/array/object, tolerant scanning)       */
/* ========================================================================= */

static const char *skip_ws(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  return p;
}

static int read_file(const char *path, char **out) {
  if (!path || !out) return -1;

  FILE *f = fopen(path, "rb");
  if (!f) return -1;

  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
  long n = ftell(f);
  if (n <= 0) { fclose(f); return -1; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }

  char *buf = (char *)malloc((size_t)n + 1u);
  if (!buf) { fclose(f); return -1; }

  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) {
    free(buf);
    return -1;
  }

  buf[n] = '\0';
  *out = buf;
  return 0;
}

static const char *parse_string(const char *p, char **out) {
  if (!p || !out) return NULL;
  if (*p != '"') return NULL;
  ++p;

  size_t cap = 64u, len = 0u;
  char *buf = (char *)malloc(cap);
  if (!buf) return NULL;

  while (*p) {
    if (*p == '"') {
      buf[len] = '\0';
      *out = buf;
      return p + 1;
    }

    if (*p == '\\') {
      ++p;
      if (!*p) break;
      /* Keep escaped char verbatim (sufficient for our JSON). */
    }

    if (len + 1u >= cap) {
      cap *= 2u;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) { free(buf); return NULL; }
      buf = nb;
    }

    buf[len++] = *p++;
  }

  free(buf);
  return NULL;
}

static const char *parse_u64(const char *p, uint64_t *out) {
  if (!p || !out) return NULL;
  p = skip_ws(p);

  uint64_t v = 0;
  if (!isdigit((unsigned char)*p)) return NULL;

  while (isdigit((unsigned char)*p)) {
    v = v * 10u + (uint64_t)(*p - '0');
    ++p;
  }

  *out = v;
  return p;
}

static const char *skip_value(const char *p);

static const char *skip_array(const char *p) {
  if (*p != '[') return NULL;
  ++p;
  for (;;) {
    p = skip_ws(p);
    if (!*p) return NULL;
    if (*p == ']') return p + 1;
    p = skip_value(p);
    if (!p) return NULL;
    p = skip_ws(p);
    if (*p == ',') { ++p; continue; }
    if (*p == ']') return p + 1;
    return NULL;
  }
}

static const char *skip_object(const char *p) {
  if (*p != '{') return NULL;
  ++p;
  for (;;) {
    p = skip_ws(p);
    if (!*p) return NULL;
    if (*p == '}') return p + 1;

    char *k = NULL;
    p = parse_string(p, &k);
    if (!p) return NULL;
    free(k);

    p = skip_ws(p);
    if (*p != ':') return NULL;
    ++p;

    p = skip_ws(p);
    p = skip_value(p);
    if (!p) return NULL;

    p = skip_ws(p);
    if (*p == ',') { ++p; continue; }
    if (*p == '}') return p + 1;
    return NULL;
  }
}

static const char *skip_value(const char *p) {
  p = skip_ws(p);
  if (!*p) return NULL;

  if (*p == '"') {
    char *tmp = NULL;
    const char *q = parse_string(p, &tmp);
    free(tmp);
    return q;
  }
  if (*p == '{') return skip_object(p);
  if (*p == '[') return skip_array(p);
  if (*p == '-' || isdigit((unsigned char)*p)) {
    uint64_t dummy = 0;
    const char *q = parse_u64(p, &dummy);
    return q ? q : NULL;
  }

  /* true/false/null */
  if (strncmp(p, "true", 4) == 0) return p + 4;
  if (strncmp(p, "false", 5) == 0) return p + 5;
  if (strncmp(p, "null", 4) == 0) return p + 4;

  return NULL;
}

/* ========================================================================= */
/* DType parsing                                                             */
/* ========================================================================= */

static tensor_dtype_t parse_dtype(const char *s) {
  if (!s) return TENSOR_DTYPE_UNKNOWN;

  if (strcmp(s, "float32") == 0 || strcmp(s, "f32") == 0 || strcmp(s, "fp32") == 0)
    return TENSOR_DTYPE_F32;

  if (strcmp(s, "float16") == 0 || strcmp(s, "f16") == 0 || strcmp(s, "fp16") == 0)
    return TENSOR_DTYPE_F16;

  if (strcmp(s, "bf16") == 0 || strcmp(s, "bfloat16") == 0)
    return TENSOR_DTYPE_BF16;

  if (strcmp(s, "u8") == 0 || strcmp(s, "uint8") == 0)
    return TENSOR_DTYPE_U8;

  if (strcmp(s, "int4") == 0)
    return TENSOR_DTYPE_INT4;

  return TENSOR_DTYPE_UNKNOWN;
}

/* ========================================================================= */
/* Tensor parsing                                                             */
/* ========================================================================= */

static void tensor_desc_zero(tensor_desc_t *td) {
  if (!td) return;
  td->name = NULL;
  td->offset = 0;
  td->size_bytes = 0;
  td->dtype = TENSOR_DTYPE_UNKNOWN;
  td->shape = NULL;
  td->ndim = 0;
}

static void tensor_desc_free(tensor_desc_t *td) {
  if (!td) return;
  free(td->name);
  free(td->shape);
  tensor_desc_zero(td);
}

static int push_tensor(tensor_desc_t **arr, uint32_t *count, uint32_t *cap, const tensor_desc_t *src) {
  if (!arr || !count || !cap || !src) return -1;

  if (*count == *cap) {
    uint32_t ncap = (*cap == 0u) ? 256u : (*cap * 2u);
    tensor_desc_t *na = (tensor_desc_t *)realloc(*arr, (size_t)ncap * sizeof(tensor_desc_t));
    if (!na) return -1;
    *arr = na;
    *cap = ncap;
  }

  (*arr)[*count] = *src;
  (*count)++;
  return 0;
}

static const char *parse_shape_array(const char *p, uint32_t **out_shape, uint32_t *out_ndim) {
  if (!p || !out_shape || !out_ndim) return NULL;
  *out_shape = NULL;
  *out_ndim = 0;

  p = skip_ws(p);
  if (*p != '[') return NULL;
  ++p;

  uint32_t tmp[32];
  uint32_t nd = 0;

  for (;;) {
    p = skip_ws(p);
    if (!*p) return NULL;
    if (*p == ']') { ++p; break; }

    uint64_t v = 0;
    const char *q = parse_u64(p, &v);
    if (!q) return NULL;
    p = q;

    if (nd < 32u) tmp[nd++] = (uint32_t)v;

    p = skip_ws(p);
    if (*p == ',') { ++p; continue; }
    if (*p == ']') { ++p; break; }
    return NULL;
  }

  if (nd > 0u) {
    uint32_t *shape = (uint32_t *)malloc((size_t)nd * sizeof(uint32_t));
    if (!shape) return NULL;
    memcpy(shape, tmp, (size_t)nd * sizeof(uint32_t));
    *out_shape = shape;
    *out_ndim = nd;
  }

  return p;
}

/* Parse object fields into td.
 * If map_style_name is non-NULL, td->name is taken from it and "name" field is optional.
 * If map_style_name is NULL, "name" field is required.
 */
static const char *parse_tensor_object(const char *p, const char *map_style_name, tensor_desc_t *td, int *out_ok) {
  if (!p || !td || !out_ok) return NULL;
  *out_ok = 0;

  p = skip_ws(p);
  if (*p != '{') return NULL;
  ++p;

  tensor_desc_zero(td);
  if (map_style_name) {
    td->name = (char *)malloc(strlen(map_style_name) + 1u);
    if (!td->name) return NULL;
    memcpy(td->name, map_style_name, strlen(map_style_name) + 1u);
  }

  for (;;) {
    p = skip_ws(p);
    if (!*p) { tensor_desc_free(td); return NULL; }
    if (*p == '}') { ++p; break; }

    char *key = NULL;
    p = parse_string(p, &key);
    if (!p) { tensor_desc_free(td); return NULL; }

    p = skip_ws(p);
    if (*p != ':') { free(key); tensor_desc_free(td); return NULL; }
    ++p;
    p = skip_ws(p);

    if (strcmp(key, "name") == 0) {
      char *nm = NULL;
      const char *q = parse_string(p, &nm);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
      if (!td->name) {
        td->name = nm;
      } else {
        free(nm);
      }
    } else if (strcmp(key, "offset") == 0) {
      const char *q = parse_u64(p, &td->offset);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
    } else if (strcmp(key, "size_bytes") == 0) {
      const char *q = parse_u64(p, &td->size_bytes);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
    } else if (strcmp(key, "dtype") == 0) {
      char *ds = NULL;
      const char *q = parse_string(p, &ds);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
      td->dtype = parse_dtype(ds);
      free(ds);
    } else if (strcmp(key, "shape") == 0) {
      uint32_t *shape = NULL;
      uint32_t ndim = 0;
      const char *q = parse_shape_array(p, &shape, &ndim);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
      td->shape = shape;
      td->ndim = ndim;
    } else {
      const char *q = skip_value(p);
      if (!q) { free(key); tensor_desc_free(td); return NULL; }
      p = q;
    }

    free(key);
    p = skip_ws(p);
    if (*p == ',') { ++p; continue; }
    if (*p == '}') { ++p; break; }
    /* tolerate missing comma by continuing if next token looks like string */
  }

  if (!td->name) {
    tensor_desc_free(td);
    return NULL;
  }

  *out_ok = 1;
  return p;
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
  if (!*p) { free(json); return -1; }

  tensor_desc_t *arr = NULL;
  uint32_t count = 0;
  uint32_t cap = 0;

  /* Case A: direct array: [ { ... }, ... ] */
  if (*p == '[') {
    ++p;
    for (;;) {
      p = skip_ws(p);
      if (!*p) break;
      if (*p == ']') { ++p; break; }

      if (*p != '{') { p = skip_value(p); if (!p) break; }
      else {
        tensor_desc_t td;
        int ok = 0;
        const char *q = parse_tensor_object(p, NULL, &td, &ok);
        if (!q || !ok) break;
        p = q;

        if (push_tensor(&arr, &count, &cap, &td) != 0) {
          tensor_desc_free(&td);
          break;
        }
      }

      p = skip_ws(p);
      if (*p == ',') { ++p; continue; }
      if (*p == ']') { ++p; break; }
    }
  }

  /* Case B/C: object (either { "tensors": [ ... ] } or map style { "name": {..}, ... }) */
  else if (*p == '{') {
    ++p;

    int found_tensors_array = 0;

    /* First, look for a "tensors" key with an array value. */
    const char *scan = p;
    while (*scan) {
      scan = skip_ws(scan);
      if (*scan == '}') break;
      if (*scan != '"') { ++scan; continue; }

      char *key = NULL;
      const char *qk = parse_string(scan, &key);
      if (!qk) { free(key); break; }
      scan = skip_ws(qk);
      if (*scan != ':') { free(key); break; }
      ++scan;
      scan = skip_ws(scan);

      if (key && strcmp(key, "tensors") == 0 && *scan == '[') {
        found_tensors_array = 1;

        ++scan;
        for (;;) {
          scan = skip_ws(scan);
          if (!*scan) break;
          if (*scan == ']') { ++scan; break; }

          if (*scan != '{') {
            scan = skip_value(scan);
            if (!scan) break;
          } else {
            tensor_desc_t td;
            int ok = 0;
            const char *qt = parse_tensor_object(scan, NULL, &td, &ok);
            if (!qt || !ok) break;
            scan = qt;

            if (push_tensor(&arr, &count, &cap, &td) != 0) {
              tensor_desc_free(&td);
              break;
            }
          }

          scan = skip_ws(scan);
          if (*scan == ',') { ++scan; continue; }
          if (*scan == ']') { ++scan; break; }
        }

        free(key);
        break;
      }

      /* Skip value */
      if (*scan == '{') scan = skip_object(scan);
      else if (*scan == '[') scan = skip_array(scan);
      else scan = skip_value(scan);

      free(key);
      if (!scan) break;

      scan = skip_ws(scan);
      if (*scan == ',') ++scan;
    }

    /* If we did not find "tensors": [ ... ], treat as map style:
     * { "tensor.name": { ... }, "tensor2": { ... }, ... }
     */
    if (!found_tensors_array) {
      while (*p) {
        p = skip_ws(p);
        if (!*p) break;
        if (*p == '}') { ++p; break; }
        if (*p == ',') { ++p; continue; }
        if (*p != '"') { p = skip_value(p); if (!p) break; continue; }

        char *name = NULL;
        p = parse_string(p, &name);
        if (!p) { free(name); break; }

        p = skip_ws(p);
        if (*p != ':') { free(name); break; }
        ++p;
        p = skip_ws(p);

        if (*p == '{') {
          tensor_desc_t td;
          int ok = 0;
          const char *q = parse_tensor_object(p, name, &td, &ok);
          if (!q || !ok) { free(name); break; }
          p = q;

          if (push_tensor(&arr, &count, &cap, &td) != 0) {
            tensor_desc_free(&td);
            free(name);
            break;
          }
        } else {
          p = skip_value(p);
          free(name);
          if (!p) break;
        }

        free(name);
        p = skip_ws(p);
        if (*p == ',') ++p;
      }
    }
  } else {
    free(json);
    return -1;
  }

  free(json);

  if (!arr || count == 0u) {
    free(arr);
    return -1;
  }

  /* Shrink to fit. */
  if (cap > count) {
    tensor_desc_t *na = (tensor_desc_t *)realloc(arr, (size_t)count * sizeof(tensor_desc_t));
    if (na) arr = na;
  }

  out->tensors = arr;
  out->count = count;
  out->loaded = 1;
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
    if (map->tensors[i].name && strcmp(map->tensors[i].name, name) == 0)
      return &map->tensors[i];
  }
  return NULL;
}
