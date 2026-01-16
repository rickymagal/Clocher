/* ============================================================================
 * File: engine/src/io/tensor_map.c
 * ============================================================================
 */
#define _POSIX_C_SOURCE 200809L

#include "tensor_map.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Internal helpers                                                           */
/* ========================================================================= */

/**
 * @brief Skip ASCII whitespace in a JSON string.
 * @param p Input pointer.
 * @return Pointer to first non-whitespace character (or NUL).
 */
static const char *skip_ws(const char *p) {
  while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  return p;
}

/**
 * @brief Read an entire file into a NUL-terminated buffer.
 * @param path Path to read.
 * @param out Receives malloc'd buffer (caller frees).
 * @return 0 on success, non-zero on failure.
 */
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

/**
 * @brief Parse a JSON string literal (minimal escape handling).
 *
 * @details
 * This parser preserves escape sequences by copying the escaped character verbatim,
 * which is sufficient for the tensor names used by this project.
 *
 * @param p Input pointer (must point at '"').
 * @param out Receives malloc'd C string (caller frees).
 * @return Pointer to first character after the closing quote, or NULL on failure.
 */
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

/**
 * @brief Parse an unsigned decimal integer.
 * @param p Input pointer.
 * @param out Receives parsed value.
 * @return Pointer after the number, or NULL on failure.
 */
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

/**
 * @brief Skip a JSON array value.
 * @param p Input pointer (must point at '[').
 * @return Pointer after the closing ']', or NULL on failure.
 */
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

/**
 * @brief Skip a JSON object value.
 * @param p Input pointer (must point at '{').
 * @return Pointer after the closing '}', or NULL on failure.
 */
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

/**
 * @brief Skip any JSON value (string/object/array/number/true/false/null).
 * @param p Input pointer.
 * @return Pointer after the value, or NULL on failure.
 */
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

  if (strncmp(p, "true", 4) == 0) return p + 4;
  if (strncmp(p, "false", 5) == 0) return p + 5;
  if (strncmp(p, "null", 4) == 0) return p + 4;

  return NULL;
}

/* ========================================================================= */
/* DType parsing                                                              */
/* ========================================================================= */

/**
 * @brief Lowercase ASCII character without locale effects.
 * @param c Input byte.
 * @return Lowercased byte.
 */
static unsigned char ie_tolower_ascii(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return (unsigned char)(c - 'A' + 'a');
  return c;
}

/**
 * @brief Case-insensitive ASCII equality for short identifiers.
 * @param a String A.
 * @param b String B.
 * @return 1 if equal, 0 otherwise.
 */
static int ie_streq_case_ascii(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    unsigned char ca = ie_tolower_ascii((unsigned char)*a++);
    unsigned char cb = ie_tolower_ascii((unsigned char)*b++);
    if (ca != cb) return 0;
  }
  return (*a == '\0' && *b == '\0') ? 1 : 0;
}

/**
 * @brief Parse dtype string into tensor_dtype_t.
 *
 * @details
 * Accepts common spellings and is ASCII case-insensitive:
 *  - "bf16", "BF16", "bfloat16"
 *  - "f16", "fp16", "float16"
 *  - "f32", "fp32", "float32"
 *  - "u8", "uint8"
 *  - "int4"
 *
 * @param s Input dtype string.
 * @return Parsed dtype tag.
 */
static tensor_dtype_t parse_dtype(const char *s) {
  if (!s) return TENSOR_DTYPE_UNKNOWN;

  if (ie_streq_case_ascii(s, "float32") || ie_streq_case_ascii(s, "f32") || ie_streq_case_ascii(s, "fp32"))
    return TENSOR_DTYPE_F32;

  if (ie_streq_case_ascii(s, "float16") || ie_streq_case_ascii(s, "f16") || ie_streq_case_ascii(s, "fp16"))
    return TENSOR_DTYPE_F16;

  if (ie_streq_case_ascii(s, "bf16") || ie_streq_case_ascii(s, "bfloat16"))
    return TENSOR_DTYPE_BF16;

  if (ie_streq_case_ascii(s, "u8") || ie_streq_case_ascii(s, "uint8"))
    return TENSOR_DTYPE_U8;

  if (ie_streq_case_ascii(s, "int4"))
    return TENSOR_DTYPE_INT4;

  return TENSOR_DTYPE_UNKNOWN;
}

/* ========================================================================= */
/* Tensor parsing                                                             */
/* ========================================================================= */

/**
 * @brief Zero-initialize a tensor descriptor.
 * @param td Descriptor to clear.
 */
static void tensor_desc_zero(tensor_desc_t *td) {
  if (!td) return;
  td->name = NULL;
  td->offset = 0;
  td->size_bytes = 0;
  td->dtype = TENSOR_DTYPE_UNKNOWN;
  td->shape = NULL;
  td->ndim = 0;
}

/**
 * @brief Free owned fields of a tensor descriptor and reset it.
 * @param td Descriptor to free/reset.
 */
static void tensor_desc_free(tensor_desc_t *td) {
  if (!td) return;
  free(td->name);
  free(td->shape);
  tensor_desc_zero(td);
}

/**
 * @brief Append a tensor descriptor to a growable array.
 * @param arr In/out array pointer.
 * @param count In/out count.
 * @param cap In/out capacity.
 * @param src Source descriptor (shallow-copied; ownership transfers to array).
 * @return 0 on success, non-zero on failure.
 */
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

/**
 * @brief Parse a JSON array of integers into a shape array.
 * @param p Input pointer.
 * @param out_shape Receives malloc'd shape array (caller frees).
 * @param out_ndim Receives dimension count.
 * @return Pointer after the closing ']', or NULL on failure.
 */
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

/**
 * @brief Parse a tensor descriptor object.
 *
 * @details
 * If @p map_style_name is non-NULL, td->name is taken from it and "name" is optional.
 * If @p map_style_name is NULL, the "name" field is required.
 *
 * Also accepts "nbytes" as an alias for "size_bytes" for compatibility with some emitters.
 *
 * @param p Input pointer (must point at '{').
 * @param map_style_name Optional name from map-style JSON key.
 * @param td Output descriptor (takes ownership of allocated fields).
 * @param out_ok Receives 1 if parsed and has a name, 0 otherwise.
 * @return Pointer after the closing '}', or NULL on failure.
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
    } else if (strcmp(key, "size_bytes") == 0 || strcmp(key, "nbytes") == 0) {
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
  }

  if (!td->name) {
    tensor_desc_free(td);
    return NULL;
  }

  *out_ok = 1;
  return p;
}

/* ========================================================================= */
/* Public API                                                                 */
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
  } else if (*p == '{') {
    ++p;

    int found_tensors_array = 0;

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

      if (*scan == '{') scan = skip_object(scan);
      else if (*scan == '[') scan = skip_array(scan);
      else scan = skip_value(scan);

      free(key);
      if (!scan) break;

      scan = skip_ws(scan);
      if (*scan == ',') ++scan;
    }

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

  if (cap > count) {
    tensor_desc_t *na = (tensor_desc_t *)realloc(arr, (size_t)count * sizeof(tensor_desc_t));
    if (na) arr = na;
  }

  out->tensors = arr;
  out->count = count;
  out->loaded = 1;
  return 0;
}

void tensor_map_free(tensor_map_t *map) {
  if (!map || !map->tensors) return;

  for (uint32_t i = 0; i < map->count; ++i) {
    free(map->tensors[i].name);
    free(map->tensors[i].shape);
  }

  free(map->tensors);
  memset(map, 0, sizeof(*map));
}

const tensor_desc_t *tensor_map_find(const tensor_map_t *map, const char *name) {
  if (!map || !map->loaded || !name) return NULL;
  for (uint32_t i = 0; i < map->count; ++i) {
    if (map->tensors[i].name && strcmp(map->tensors[i].name, name) == 0)
      return &map->tensors[i];
  }
  return NULL;
}

/* ========================================================================= */
/* Validation (hard tensor_map ↔ bin consistency check)                        */
/* ========================================================================= */

typedef struct tm_sort_ref_s {
  uint64_t off;
  uint64_t end;
  uint32_t idx;
} tm_sort_ref_t;

static int tm_sort_ref_cmp(const void *a, const void *b) {
  const tm_sort_ref_t *A = (const tm_sort_ref_t *)a;
  const tm_sort_ref_t *B = (const tm_sort_ref_t *)b;
  if (A->off < B->off) return -1;
  if (A->off > B->off) return 1;
  if (A->end < B->end) return -1;
  if (A->end > B->end) return 1;
  return 0;
}

static void tm_set_err(char *err, size_t errsz, const char *msg) {
  if (!err || errsz == 0u) return;
  (void)snprintf(err, errsz, "%s", msg ? msg : "unknown error");
}

int tensor_map_validate_against_bin(const tensor_map_t *map, size_t bin_size, char *err, size_t errsz) {
  if (!map || !map->loaded || !map->tensors || map->count == 0u) {
    tm_set_err(err, errsz, "tensor_map not loaded");
    return -1;
  }

  tm_sort_ref_t *refs = (tm_sort_ref_t *)malloc((size_t)map->count * sizeof(*refs));
  if (!refs) {
    tm_set_err(err, errsz, "OOM allocating validation refs");
    return -2;
  }

  for (uint32_t i = 0; i < map->count; ++i) {
    const tensor_desc_t *td = &map->tensors[i];

    if (!td->name || td->name[0] == '\0') {
      tm_set_err(err, errsz, "tensor_map contains entry with empty name");
      free(refs);
      return -3;
    }
    if (td->size_bytes == 0u) {
      char buf[256];
      if (tensor_desc_to_string(td, buf, sizeof(buf)) >= 0) {
        (void)snprintf(err, errsz, "tensor_map contains zero-sized tensor: %s", buf);
      } else {
        tm_set_err(err, errsz, "tensor_map contains zero-sized tensor");
      }
      free(refs);
      return -4;
    }

    const uint64_t end = td->offset + td->size_bytes;
    if (end < td->offset) {
      tm_set_err(err, errsz, "tensor_map offset+size overflow");
      free(refs);
      return -5;
    }
    if (end > (uint64_t)bin_size) {
      char buf[256];
      if (tensor_desc_to_string(td, buf, sizeof(buf)) >= 0) {
        (void)snprintf(err, errsz,
                       "tensor_map entry out of bounds: bin_size=%llu %s",
                       (unsigned long long)bin_size, buf);
      } else {
        tm_set_err(err, errsz, "tensor_map entry out of bounds");
      }
      free(refs);
      return -6;
    }

    refs[i].off = td->offset;
    refs[i].end = end;
    refs[i].idx = i;
  }

  qsort(refs, (size_t)map->count, sizeof(*refs), tm_sort_ref_cmp);

  uint64_t prev_end = refs[0].end;
  for (uint32_t k = 1; k < map->count; ++k) {
    if (refs[k].off < prev_end) {
      const tensor_desc_t *A = &map->tensors[refs[k - 1].idx];
      const tensor_desc_t *B = &map->tensors[refs[k].idx];

      char a_buf[256], b_buf[256];
      (void)tensor_desc_to_string(A, a_buf, sizeof(a_buf));
      (void)tensor_desc_to_string(B, b_buf, sizeof(b_buf));

      (void)snprintf(err, errsz,
                     "tensor_map overlap detected:\n  A: %s\n  B: %s",
                     a_buf, b_buf);

      free(refs);
      return -7;
    }
    prev_end = refs[k].end;
  }

  free(refs);
  if (err && errsz) err[0] = '\0';
  return 0;
}
