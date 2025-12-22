/**
 * @file weights_dedup.c
 * @brief Lossless deduplicated-weights loader and materialization utilities.
 *
 * @details
 * This module loads a deduplicated weights manifest (model.dedup.json) and
 * mmaps the referenced blobs (defaults / masks / exceptions). It exposes
 * tensor "views" that either:
 *   - directly reference contiguous bytes (direct view), or
 *   - describe a lossless reconstruction (dedup view) via (defaults + mask + exceptions).
 *
 * Design goals:
 *  - mmap-friendly: map large blobs once and provide pointer views
 *  - low overhead lookup: sort tensor table by name and binary search
 *  - lossless: exact reconstruction via mask + exceptions stream
 *  - minimal dependencies: internal JSON scanner for a strict subset of JSON
 *
 * Mask semantics:
 *  - bit=1 indicates an exception element is present and consumed from the exceptions stream.
 *  - For int4, elements are nibbles in row-major order (index i maps to byte i/2, nibble i%2).
 */

#include "weights_dedup.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* -------------------------------------------------------------------------- */
/* Compatibility macros                                                       */
/* -------------------------------------------------------------------------- */
/**
 * @brief Compatibility alias for rank macro across header revisions.
 *
 * @details
 * Some header versions expose IE_DEDUP_MAX_RANK, while older code used
 * IE_WDEDUP_MAX_RANK. This module supports both without requiring header edits.
 */
#ifndef IE_WDEDUP_MAX_RANK
  #ifdef IE_DEDUP_MAX_RANK
    #define IE_WDEDUP_MAX_RANK IE_DEDUP_MAX_RANK
  #else
    #define IE_WDEDUP_MAX_RANK 8
  #endif
#endif

/* -------------------------------------------------------------------------- */
/* Logging                                                                    */
/* -------------------------------------------------------------------------- */

static int ie_wdedup_debug_enabled(void) {
  const char *v = getenv("IE_DEDUP_DEBUG");
  if (!v || !*v) return 0;
  if (strcmp(v, "0") == 0) return 0;
  return 1;
}

static void ie_wdedup_vlogf(const char *fmt, va_list ap) {
  if (!ie_wdedup_debug_enabled()) return;
  fprintf(stderr, "[dedup] ");
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
}

static void ie_wdedup_logf(const char *fmt, ...) {
  if (!ie_wdedup_debug_enabled()) return;
  va_list ap;
  va_start(ap, fmt);
  ie_wdedup_vlogf(fmt, ap);
  va_end(ap);
}

static const char *ie_wdedup_errno_str(int e) {
  const char *s = strerror(e);
  return s ? s : "unknown";
}

/* -------------------------------------------------------------------------- */
/* Status strings (must match header declaration)                             */
/* -------------------------------------------------------------------------- */

const char *ie_wdedup_status_str(ie_wdedup_status_t st) {
  switch (st) {
    case IE_WDEDUP_OK:      return "OK";
    case IE_WDEDUP_EINVAL:  return "EINVAL";
    case IE_WDEDUP_ENOMEM:  return "ENOMEM";
    case IE_WDEDUP_ENOENT:  return "ENOENT";
    case IE_WDEDUP_EIO:     return "EIO";
    case IE_WDEDUP_EFORMAT: return "EFORMAT";
    case IE_WDEDUP_ERANGE:  return "ERANGE";
    default:                return "UNKNOWN";
  }
}

/* -------------------------------------------------------------------------- */
/* Internal helpers                                                           */
/* -------------------------------------------------------------------------- */

static char *ie_strdup(const char *s) {
  const char *src = s ? s : "";
  size_t n = strlen(src);
  char *out = (char *)malloc(n + 1);
  if (!out) return NULL;
  memcpy(out, src, n);
  out[n] = '\0';
  return out;
}

static char *ie_read_file_text(const char *path, size_t *out_size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return NULL;

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return NULL;
  }
  if (st.st_size < 0) {
    close(fd);
    return NULL;
  }

  size_t n = (size_t)st.st_size;
  char *buf = (char *)malloc(n + 1);
  if (!buf) {
    close(fd);
    return NULL;
  }

  size_t off = 0;
  while (off < n) {
    ssize_t r = read(fd, buf + off, n - off);
    if (r < 0) {
      if (errno == EINTR) continue;
      free(buf);
      close(fd);
      return NULL;
    }
    if (r == 0) break;
    off += (size_t)r;
  }
  close(fd);

  buf[off] = '\0';
  if (out_size) *out_size = off;
  return buf;
}

static char *ie_path_join(const char *dir, const char *rel) {
  if (!dir || !rel) return NULL;
  size_t nd = strlen(dir);
  size_t nr = strlen(rel);
  int need_slash = (nd > 0 && dir[nd - 1] != '/');
  size_t outn = nd + (need_slash ? 1 : 0) + nr + 1;

  char *p = (char *)malloc(outn);
  if (!p) return NULL;

  memcpy(p, dir, nd);
  size_t k = nd;
  if (need_slash) p[k++] = '/';
  memcpy(p + k, rel, nr);
  p[k + nr] = '\0';
  return p;
}

static void ie_madvise_policy(void *addr, size_t len, int policy) {
  if (!addr || len == 0) return;
#if defined(MADV_SEQUENTIAL) && defined(MADV_WILLNEED)
  if (policy == 1) (void)madvise(addr, len, MADV_SEQUENTIAL);
  else if (policy == 2) (void)madvise(addr, len, MADV_WILLNEED);
#else
  (void)addr;
  (void)len;
  (void)policy;
#endif
}

typedef struct ie_mmap_file {
  char *path;
  int fd;
  void *base;
  size_t size;
} ie_mmap_file_t;

static ie_wdedup_status_t ie_mmap_ro(ie_mmap_file_t *mf, const char *path, int policy) {
  if (!mf || !path) return IE_WDEDUP_EINVAL;

  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;

  mf->path = ie_strdup(path);
  if (!mf->path) return IE_WDEDUP_ENOMEM;

  mf->fd = open(path, O_RDONLY);
  if (mf->fd < 0) {
    ie_wdedup_logf("open failed path=%s errno=%d (%s)", path, errno, ie_wdedup_errno_str(errno));
    free(mf->path);
    mf->path = NULL;
    mf->fd = -1;
    return IE_WDEDUP_ENOENT;
  }

  struct stat st;
  if (fstat(mf->fd, &st) != 0 || st.st_size < 0) {
    (void)close(mf->fd);
    mf->fd = -1;
    free(mf->path);
    mf->path = NULL;
    return IE_WDEDUP_EIO;
  }

  mf->size = (size_t)st.st_size;
  if (mf->size == 0) {
    mf->base = NULL;
    return IE_WDEDUP_OK;
  }

  void *p = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
  if (p == MAP_FAILED) {
    (void)close(mf->fd);
    mf->fd = -1;
    free(mf->path);
    mf->path = NULL;
    mf->size = 0;
    return IE_WDEDUP_EIO;
  }

  mf->base = p;
  ie_madvise_policy(mf->base, mf->size, policy);
  return IE_WDEDUP_OK;
}

static void ie_mmap_close(ie_mmap_file_t *mf) {
  if (!mf) return;
  if (mf->base && mf->size) (void)munmap(mf->base, mf->size);
  if (mf->fd >= 0) (void)close(mf->fd);
  free(mf->path);
  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;
}

/* -------------------------------------------------------------------------- */
/* Minimal JSON scanner                                                       */
/* -------------------------------------------------------------------------- */

static const char *js_skip_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return p;
}

static int js_match_lit(const char *p, const char *lit) {
  size_t n = strlen(lit);
  return (strncmp(p, lit, n) == 0);
}

static int js_parse_string(const char **p, char **out) {
  const char *s = js_skip_ws(*p);
  if (*s != '"') return 0;
  s++;

  size_t cap = 64;
  size_t len = 0;
  char *buf = (char *)malloc(cap);
  if (!buf) return 0;

  while (*s && *s != '"') {
    char c = *s++;
    if (c == '\\') {
      char e = *s++;
      if (e == '"' || e == '\\' || e == '/') c = e;
      else if (e == 'b') c = '\b';
      else if (e == 'f') c = '\f';
      else if (e == 'n') c = '\n';
      else if (e == 'r') c = '\r';
      else if (e == 't') c = '\t';
      else {
        free(buf);
        return 0;
      }
    }
    if (len + 1 >= cap) {
      cap *= 2;
      char *nb = (char *)realloc(buf, cap);
      if (!nb) {
        free(buf);
        return 0;
      }
      buf = nb;
    }
    buf[len++] = c;
  }
  if (*s != '"') {
    free(buf);
    return 0;
  }
  s++;

  buf[len] = '\0';
  *out = buf;
  *p = s;
  return 1;
}

static int js_parse_i64(const char **p, int64_t *out) {
  const char *s = js_skip_ws(*p);
  int neg = 0;
  if (*s == '-') { neg = 1; s++; }
  if (*s < '0' || *s > '9') return 0;

  int64_t v = 0;
  while (*s >= '0' && *s <= '9') {
    int d = *s - '0';
    if (v > (INT64_MAX - d) / 10) return 0;
    v = v * 10 + d;
    s++;
  }
  *out = neg ? -v : v;
  *p = s;
  return 1;
}

static int js_skip_value(const char **p);

static int js_skip_array(const char **p) {
  const char *s = js_skip_ws(*p);
  if (*s != '[') return 0;
  s++;
  s = js_skip_ws(s);
  if (*s == ']') { *p = s + 1; return 1; }
  for (;;) {
    *p = s;
    if (!js_skip_value(p)) return 0;
    s = js_skip_ws(*p);
    if (*s == ',') { s++; s = js_skip_ws(s); continue; }
    if (*s == ']') { *p = s + 1; return 1; }
    return 0;
  }
}

static int js_skip_object(const char **p) {
  const char *s = js_skip_ws(*p);
  if (*s != '{') return 0;
  s++;
  s = js_skip_ws(s);
  if (*s == '}') { *p = s + 1; return 1; }
  for (;;) {
    *p = s;
    char *k = NULL;
    if (!js_parse_string(p, &k)) return 0;
    free(k);
    s = js_skip_ws(*p);
    if (*s != ':') return 0;
    s++;
    *p = s;
    if (!js_skip_value(p)) return 0;
    s = js_skip_ws(*p);
    if (*s == ',') { s++; s = js_skip_ws(s); continue; }
    if (*s == '}') { *p = s + 1; return 1; }
    return 0;
  }
}

static int js_skip_value(const char **p) {
  const char *s = js_skip_ws(*p);
  if (*s == '"') {
    char *tmp = NULL;
    int ok = js_parse_string(&s, &tmp);
    free(tmp);
    if (!ok) return 0;
    *p = s;
    return 1;
  }
  if (*s == '{') { *p = s; return js_skip_object(p); }
  if (*s == '[') { *p = s; return js_skip_array(p); }
  if (*s == '-' || (*s >= '0' && *s <= '9')) {
    int64_t v = 0;
    *p = s;
    if (!js_parse_i64(p, &v)) return 0;
    return 1;
  }
  if (js_match_lit(s, "true"))  { *p = s + 4; return 1; }
  if (js_match_lit(s, "false")) { *p = s + 5; return 1; }
  if (js_match_lit(s, "null"))  { *p = s + 4; return 1; }
  return 0;
}

static int js_find_member_value(const char *json, const char *key, const char **out_val) {
  const char *p = js_skip_ws(json);
  if (*p != '{') return 0;
  p++;
  p = js_skip_ws(p);
  if (*p == '}') return 0;

  for (;;) {
    char *k = NULL;
    if (!js_parse_string(&p, &k)) return 0;
    p = js_skip_ws(p);
    if (*p != ':') { free(k); return 0; }
    p++;
    p = js_skip_ws(p);

    int match = (strcmp(k, key) == 0);
    free(k);

    if (match) {
      *out_val = p;
      return 1;
    }

    if (!js_skip_value(&p)) return 0;
    p = js_skip_ws(p);
    if (*p == ',') { p++; p = js_skip_ws(p); continue; }
    if (*p == '}') return 0;
    return 0;
  }
}

/* -------------------------------------------------------------------------- */
/* Dtype utilities                                                            */
/* -------------------------------------------------------------------------- */

ie_wdtype_t ie_weights_dedup_parse_dtype(const char *s) {
  if (!s) return IE_WDTYPE_UNKNOWN;

  /* Case-insensitive matching without locale. */
  char tmp[16];
  size_t n = strlen(s);
  if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
  for (size_t i = 0; i < n; i++) {
    char c = s[i];
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    tmp[i] = c;
  }
  tmp[n] = '\0';

  if (strcmp(tmp, "fp32") == 0) return IE_WDTYPE_FP32;
  if (strcmp(tmp, "fp16") == 0) return IE_WDTYPE_FP16;
  if (strcmp(tmp, "bf16") == 0) return IE_WDTYPE_BF16;
  if (strcmp(tmp, "int8") == 0) return IE_WDTYPE_INT8;
  if (strcmp(tmp, "int4") == 0) return IE_WDTYPE_INT4;

  return IE_WDTYPE_UNKNOWN;
}

static size_t ie_dtype_elem_size(ie_wdtype_t dt) {
  switch (dt) {
    case IE_WDTYPE_FP32: return 4;
    case IE_WDTYPE_FP16: return 2;
    case IE_WDTYPE_BF16: return 2;
    case IE_WDTYPE_INT8: return 1;
    case IE_WDTYPE_INT4: return 0; /* nibble-packed */
    default: return 0;
  }
}

static int ie_shape_elem_count(const int64_t *shape, int32_t rank, size_t *out) {
  if (!shape || !out || rank < 0 || rank > IE_WDEDUP_MAX_RANK) return 0;
  size_t n = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (shape[i] < 0) return 0;
    if (shape[i] == 0) { n = 0; break; }
    if (n > (SIZE_MAX / (size_t)shape[i])) return 0;
    n *= (size_t)shape[i];
  }
  *out = n;
  return 1;
}

/* -------------------------------------------------------------------------- */
/* Schema2: model.ie.json tensor parsing (robust, raw_safetensors-compatible) */
/* -------------------------------------------------------------------------- */

typedef struct ie_iejson_tensor {
  char *name;
  ie_wdtype_t dtype;
  int32_t rank;
  int64_t shape[IE_WDEDUP_MAX_RANK];
  uint64_t offset;
  uint64_t nbytes;

  /* Optional: shard/file metadata if present. */
  char *file;
  uint64_t file_data_offset;

  size_t elem_count;
  size_t elem_size;
} ie_iejson_tensor_t;

static void ie_iejson_tensor_free(ie_iejson_tensor_t *t) {
  if (!t) return;
  free(t->name);
  free(t->file);
  memset(t, 0, sizeof(*t));
}

static int js_parse_shape(const char **p, int64_t *shape, int32_t *rank) {
  const char *s = js_skip_ws(*p);
  if (*s != '[') return 0;
  s++;
  s = js_skip_ws(s);

  int32_t r = 0;
  if (*s == ']') { *p = s + 1; *rank = 0; return 1; }

  for (;;) {
    if (r >= IE_WDEDUP_MAX_RANK) return 0;
    int64_t v = 0;
    const char *tmp = s;
    if (!js_parse_i64(&tmp, &v)) return 0;
    shape[r++] = v;
    s = js_skip_ws(tmp);

    if (*s == ',') { s++; s = js_skip_ws(s); continue; }
    if (*s == ']') { *p = s + 1; *rank = r; return 1; }
    return 0;
  }
}

static int js_read_u64_member(const char *obj, const char *key, uint64_t *out) {
  const char *v = NULL;
  if (!js_find_member_value(obj, key, &v)) return 0;
  int64_t x = 0;
  if (!js_parse_i64(&v, &x)) return 0;
  if (x < 0) return 0;
  *out = (uint64_t)x;
  return 1;
}

static int js_read_str_member(const char *obj, const char *key, char **out) {
  const char *v = NULL;
  if (!js_find_member_value(obj, key, &v)) return 0;
  return js_parse_string(&v, out);
}

static ie_wdedup_status_t ie_parse_iejson_one_tensor(const char *obj, size_t idx, ie_iejson_tensor_t *out) {
  memset(out, 0, sizeof(*out));
  out->dtype = IE_WDTYPE_UNKNOWN;

  char *name = NULL;
  char *dtype_s = NULL;

  if (!js_read_str_member(obj, "name", &name)) {
    ie_wdedup_logf("schema2: tensor[%zu] missing/invalid \"name\"", idx);
    return IE_WDEDUP_EFORMAT;
  }
  if (!js_read_str_member(obj, "dtype", &dtype_s)) {
    ie_wdedup_logf("schema2: tensor[%zu] missing/invalid \"dtype\" name=%s", idx, name);
    free(name);
    return IE_WDEDUP_EFORMAT;
  }

  ie_wdtype_t dt = ie_weights_dedup_parse_dtype(dtype_s);
  if (dt == IE_WDTYPE_UNKNOWN) {
    /* Do not hard-fail here: schema2 may see BF16 tensors even if dedup is int4-only. */
    ie_wdedup_logf("schema2: tensor[%zu] unknown dtype=%s name=%s (continuing)", idx, dtype_s, name);
    dt = IE_WDTYPE_UNKNOWN;
  }

  uint64_t off = 0, nb = 0;
  if (!js_read_u64_member(obj, "offset", &off) || !js_read_u64_member(obj, "nbytes", &nb)) {
    ie_wdedup_logf("schema2: tensor[%zu] missing/invalid offset/nbytes name=%s", idx, name);
    free(name);
    free(dtype_s);
    return IE_WDEDUP_EFORMAT;
  }

  const char *shape_val = NULL;
  if (!js_find_member_value(obj, "shape", &shape_val)) {
    ie_wdedup_logf("schema2: tensor[%zu] missing \"shape\" name=%s", idx, name);
    free(name);
    free(dtype_s);
    return IE_WDEDUP_EFORMAT;
  }
  int32_t rank = 0;
  int64_t shape[IE_WDEDUP_MAX_RANK];
  memset(shape, 0, sizeof(shape));
  if (!js_parse_shape(&shape_val, shape, &rank)) {
    ie_wdedup_logf("schema2: tensor[%zu] invalid shape name=%s", idx, name);
    free(name);
    free(dtype_s);
    return IE_WDEDUP_EFORMAT;
  }

  /* Optional: accept either "file" or "shard". */
  char *file = NULL;
  if (!js_read_str_member(obj, "file", &file)) {
    (void)js_read_str_member(obj, "shard", &file);
  }

  /* Optional: accept either "file_data_offset" or "shard_data_offset". */
  uint64_t foff = 0;
  if (!js_read_u64_member(obj, "file_data_offset", &foff)) {
    (void)js_read_u64_member(obj, "shard_data_offset", &foff);
  }

  out->name = name;
  out->dtype = dt;
  out->rank = rank;
  for (int32_t i = 0; i < rank; i++) out->shape[i] = shape[i];
  out->offset = off;
  out->nbytes = nb;
  out->file = file;
  out->file_data_offset = foff;

  out->elem_size = ie_dtype_elem_size(dt);
  if (!ie_shape_elem_count(out->shape, out->rank, &out->elem_count)) {
    ie_wdedup_logf("schema2: tensor[%zu] shape overflow name=%s", idx, name);
    ie_iejson_tensor_free(out);
    free(dtype_s);
    return IE_WDEDUP_ERANGE;
  }

  free(dtype_s);
  return IE_WDEDUP_OK;
}

static ie_wdedup_status_t ie_parse_iejson_tensors(const char *json,
                                                  ie_iejson_tensor_t **out,
                                                  size_t *out_count) {
  if (!out || !out_count) return IE_WDEDUP_EINVAL;
  *out = NULL;
  *out_count = 0;

  const char *tensors_val = NULL;
  if (!js_find_member_value(json, "tensors", &tensors_val)) {
    ie_wdedup_logf("schema2: model.ie.json missing top-level \"tensors\"");
    return IE_WDEDUP_EFORMAT;
  }

  const char *p = js_skip_ws(tensors_val);
  if (*p != '[') {
    ie_wdedup_logf("schema2: model.ie.json \"tensors\" is not an array");
    return IE_WDEDUP_EFORMAT;
  }
  p++;
  p = js_skip_ws(p);

  size_t cap = 512;
  size_t cnt = 0;
  ie_iejson_tensor_t *arr = (ie_iejson_tensor_t *)calloc(cap, sizeof(*arr));
  if (!arr) return IE_WDEDUP_ENOMEM;

  if (*p == ']') {
    *out = arr;
    *out_count = 0;
    return IE_WDEDUP_OK;
  }

  for (;;) {
    p = js_skip_ws(p);
    if (*p != '{') {
      ie_wdedup_logf("schema2: tensors[%zu] expected object, got '%c'", cnt, *p ? *p : '?');
      for (size_t i = 0; i < cnt; i++) ie_iejson_tensor_free(&arr[i]);
      free(arr);
      return IE_WDEDUP_EFORMAT;
    }

    if (cnt == cap) {
      size_t oldcap = cap;
      cap *= 2;
      ie_iejson_tensor_t *na = (ie_iejson_tensor_t *)realloc(arr, cap * sizeof(*arr));
      if (!na) {
        for (size_t i = 0; i < cnt; i++) ie_iejson_tensor_free(&arr[i]);
        free(arr);
        return IE_WDEDUP_ENOMEM;
      }
      arr = na;
      memset(arr + oldcap, 0, (cap - oldcap) * sizeof(*arr));
    }

    ie_wdedup_status_t st = ie_parse_iejson_one_tensor(p, cnt, &arr[cnt]);
    if (st != IE_WDEDUP_OK) {
      for (size_t i = 0; i <= cnt && i < cap; i++) ie_iejson_tensor_free(&arr[i]);
      free(arr);
      return st;
    }

    /* Advance past this object. */
    const char *tmp = p;
    if (!js_skip_object(&tmp)) {
      ie_wdedup_logf("schema2: tensors[%zu] failed to skip object after parsing", cnt);
      for (size_t i = 0; i <= cnt; i++) ie_iejson_tensor_free(&arr[i]);
      free(arr);
      return IE_WDEDUP_EFORMAT;
    }
    p = js_skip_ws(tmp);
    cnt++;

    if (*p == ',') { p++; continue; }
    if (*p == ']') break;

    ie_wdedup_logf("schema2: tensors array: expected ',' or ']', got '%c'", *p ? *p : '?');
    for (size_t i = 0; i < cnt; i++) ie_iejson_tensor_free(&arr[i]);
    free(arr);
    return IE_WDEDUP_EFORMAT;
  }

  *out = arr;
  *out_count = cnt;
  return IE_WDEDUP_OK;
}

/* -------------------------------------------------------------------------- */
/* Runtime structures (schema1 = model.dedup.json with files+tensors)          */
/* -------------------------------------------------------------------------- */

typedef struct ie_file_ref {
  char *logical;
  char *relpath;
} ie_file_ref_t;

typedef struct ie_tensor_ref {
  char *name;
  ie_wdtype_t dtype;

  int32_t rank;
  int64_t shape[IE_WDEDUP_MAX_RANK];

  uint32_t file_defaults;
  uint64_t off_defaults;
  uint64_t nbytes_defaults;

  uint32_t file_mask;
  uint64_t off_mask;
  uint64_t nbytes_mask;

  uint32_t file_exceptions;
  uint64_t off_exceptions;
  uint64_t nbytes_exceptions;

  size_t elem_count;
  size_t elem_size;
} ie_tensor_ref_t;

struct ie_weights_dedup {
  char *model_dir;

  ie_mmap_file_t *files;
  size_t files_count;

  ie_tensor_ref_t *tensors;
  size_t tensors_count;

  /* Schema2: parsed model.ie.json (only for debug and future extensions). */
  ie_iejson_tensor_t *iejson_tensors;
  size_t iejson_tensors_count;
  ie_mmap_file_t iebin; /* model.ie.bin */

  int prefault_policy;
};

static int ie_tensor_cmp_name(const void *a, const void *b) {
  const ie_tensor_ref_t *A = (const ie_tensor_ref_t *)a;
  const ie_tensor_ref_t *B = (const ie_tensor_ref_t *)b;
  return strcmp(A->name, B->name);
}

static const ie_tensor_ref_t *ie_find_tensor(const ie_weights_dedup_t *h, const char *name) {
  if (!h || !name || !h->tensors || h->tensors_count == 0) return NULL;

  ie_tensor_ref_t key;
  memset(&key, 0, sizeof(key));
  key.name = (char *)name;

  return (const ie_tensor_ref_t *)bsearch(&key, h->tensors, h->tensors_count,
                                         sizeof(h->tensors[0]), ie_tensor_cmp_name);
}

static ie_wdedup_status_t ie_parse_files_object(const char *json_obj,
                                                ie_file_ref_t **out,
                                                size_t *out_count) {
  if (!out || !out_count) return IE_WDEDUP_EINVAL;
  *out = NULL;
  *out_count = 0;

  const char *p = js_skip_ws(json_obj);
  if (*p != '{') return IE_WDEDUP_EFORMAT;
  p++;

  size_t cap = 8;
  size_t cnt = 0;
  ie_file_ref_t *arr = (ie_file_ref_t *)calloc(cap, sizeof(*arr));
  if (!arr) return IE_WDEDUP_ENOMEM;

  p = js_skip_ws(p);
  if (*p == '}') {
    *out = arr;
    *out_count = 0;
    return IE_WDEDUP_OK;
  }

  for (;;) {
    char *k = NULL;
    if (!js_parse_string(&p, &k)) {
      free(arr);
      return IE_WDEDUP_EFORMAT;
    }
    p = js_skip_ws(p);
    if (*p != ':') {
      free(k);
      for (size_t i = 0; i < cnt; i++) { free(arr[i].logical); free(arr[i].relpath); }
      free(arr);
      return IE_WDEDUP_EFORMAT;
    }
    p++;
    p = js_skip_ws(p);

    char *v = NULL;
    if (!js_parse_string(&p, &v)) {
      free(k);
      for (size_t i = 0; i < cnt; i++) { free(arr[i].logical); free(arr[i].relpath); }
      free(arr);
      return IE_WDEDUP_EFORMAT;
    }

    if (cnt == cap) {
      size_t oldcap = cap;
      cap *= 2;
      ie_file_ref_t *na = (ie_file_ref_t *)realloc(arr, cap * sizeof(*arr));
      if (!na) {
        free(k);
        free(v);
        for (size_t i = 0; i < cnt; i++) { free(arr[i].logical); free(arr[i].relpath); }
        free(arr);
        return IE_WDEDUP_ENOMEM;
      }
      arr = na;
      memset(arr + oldcap, 0, (cap - oldcap) * sizeof(*arr));
    }

    arr[cnt].logical = k;
    arr[cnt].relpath = v;
    cnt++;

    p = js_skip_ws(p);
    if (*p == ',') { p++; p = js_skip_ws(p); continue; }
    if (*p == '}') break;

    for (size_t i = 0; i < cnt; i++) { free(arr[i].logical); free(arr[i].relpath); }
    free(arr);
    return IE_WDEDUP_EFORMAT;
  }

  *out = arr;
  *out_count = cnt;
  return IE_WDEDUP_OK;
}

static int ie_find_file_index(const ie_file_ref_t *refs, size_t n, const char *logical, uint32_t *out_idx) {
  if (!refs || !logical || !out_idx) return 0;
  for (size_t i = 0; i < n; i++) {
    if (refs[i].logical && strcmp(refs[i].logical, logical) == 0) {
      *out_idx = (uint32_t)i;
      return 1;
    }
  }
  return 0;
}

static int js_parse_blobref(const char **p, char **out_file, int64_t *out_off, int64_t *out_nb) {
  if (!p || !*p || !out_file || !out_off || !out_nb) return 0;

  const char *s = js_skip_ws(*p);
  if (*s != '{') return 0;

  const char *val = NULL;
  if (!js_find_member_value(s, "file", &val)) return 0;
  if (!js_parse_string(&val, out_file)) return 0;

  val = NULL;
  if (!js_find_member_value(s, "offset", &val)) { free(*out_file); *out_file = NULL; return 0; }
  if (!js_parse_i64(&val, out_off)) { free(*out_file); *out_file = NULL; return 0; }

  val = NULL;
  if (!js_find_member_value(s, "nbytes", &val)) { free(*out_file); *out_file = NULL; return 0; }
  if (!js_parse_i64(&val, out_nb)) { free(*out_file); *out_file = NULL; return 0; }

  *p = s;
  return js_skip_object(p);
}

static ie_wdedup_status_t ie_parse_one_tensor(const char **p,
                                              const ie_file_ref_t *files,
                                              size_t files_n,
                                              ie_tensor_ref_t *out) {
  memset(out, 0, sizeof(*out));
  out->dtype = IE_WDTYPE_UNKNOWN;

  const char *s = js_skip_ws(*p);
  if (*s != '{') return IE_WDEDUP_EFORMAT;

  const char *val = NULL;

  val = NULL;
  if (!js_find_member_value(s, "name", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_string(&val, &out->name)) return IE_WDEDUP_EFORMAT;

  char *dtype_s = NULL;
  val = NULL;
  if (!js_find_member_value(s, "dtype", &val)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }
  if (!js_parse_string(&val, &dtype_s)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }
  out->dtype = ie_weights_dedup_parse_dtype(dtype_s);
  free(dtype_s);

  /* In schema1, dtype must be known. */
  if (out->dtype == IE_WDTYPE_UNKNOWN) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }

  val = NULL;
  if (!js_find_member_value(s, "shape", &val)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }
  if (!js_parse_shape(&val, out->shape, &out->rank)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }

  char *f_default = NULL, *f_mask = NULL, *f_exc = NULL;
  int64_t off_default = 0, nb_default = 0;
  int64_t off_mask = 0, nb_mask = 0;
  int64_t off_exc = 0, nb_exc = 0;

  val = NULL;
  if (!js_find_member_value(s, "default", &val)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }
  if (!js_parse_blobref(&val, &f_default, &off_default, &nb_default)) { free(out->name); out->name = NULL; return IE_WDEDUP_EFORMAT; }

  val = NULL;
  if (!js_find_member_value(s, "mask", &val)) { free(out->name); out->name = NULL; free(f_default); return IE_WDEDUP_EFORMAT; }
  if (!js_parse_blobref(&val, &f_mask, &off_mask, &nb_mask)) { free(out->name); out->name = NULL; free(f_default); return IE_WDEDUP_EFORMAT; }

  val = NULL;
  if (!js_find_member_value(s, "exceptions", &val)) { free(out->name); out->name = NULL; free(f_default); free(f_mask); return IE_WDEDUP_EFORMAT; }
  if (!js_parse_blobref(&val, &f_exc, &off_exc, &nb_exc)) { free(out->name); out->name = NULL; free(f_default); free(f_mask); return IE_WDEDUP_EFORMAT; }

  uint32_t idx_def = 0, idx_mask = 0, idx_exc = 0;
  int ok_def = ie_find_file_index(files, files_n, f_default, &idx_def);
  int ok_msk = ie_find_file_index(files, files_n, f_mask, &idx_mask);
  int ok_exc = ie_find_file_index(files, files_n, f_exc, &idx_exc);

  free(f_default);
  free(f_mask);
  free(f_exc);

  if (!ok_def || !ok_msk || !ok_exc) {
    free(out->name);
    out->name = NULL;
    return IE_WDEDUP_EFORMAT;
  }

  if (off_default < 0 || nb_default < 0 || off_mask < 0 || nb_mask < 0 || off_exc < 0 || nb_exc < 0) {
    free(out->name);
    out->name = NULL;
    return IE_WDEDUP_ERANGE;
  }

  out->file_defaults = idx_def;
  out->off_defaults = (uint64_t)off_default;
  out->nbytes_defaults = (uint64_t)nb_default;

  out->file_mask = idx_mask;
  out->off_mask = (uint64_t)off_mask;
  out->nbytes_mask = (uint64_t)nb_mask;

  out->file_exceptions = idx_exc;
  out->off_exceptions = (uint64_t)off_exc;
  out->nbytes_exceptions = (uint64_t)nb_exc;

  out->elem_size = ie_dtype_elem_size(out->dtype);
  if (!ie_shape_elem_count(out->shape, out->rank, &out->elem_count)) {
    free(out->name);
    out->name = NULL;
    return IE_WDEDUP_ERANGE;
  }

  *p = s;
  if (!js_skip_object(p)) {
    free(out->name);
    out->name = NULL;
    return IE_WDEDUP_EFORMAT;
  }
  return IE_WDEDUP_OK;
}

static ie_wdedup_status_t ie_parse_tensors_array(const char *json_val,
                                                 const ie_file_ref_t *files,
                                                 size_t files_n,
                                                 ie_tensor_ref_t **out,
                                                 size_t *out_count) {
  if (!out || !out_count) return IE_WDEDUP_EINVAL;
  *out = NULL;
  *out_count = 0;

  const char *p = js_skip_ws(json_val);
  if (*p != '[') return IE_WDEDUP_EFORMAT;
  p++;
  p = js_skip_ws(p);

  size_t cap = 256;
  size_t cnt = 0;
  ie_tensor_ref_t *arr = (ie_tensor_ref_t *)calloc(cap, sizeof(*arr));
  if (!arr) return IE_WDEDUP_ENOMEM;

  if (*p == ']') {
    *out = arr;
    *out_count = 0;
    return IE_WDEDUP_OK;
  }

  for (;;) {
    if (cnt == cap) {
      size_t oldcap = cap;
      cap *= 2;
      ie_tensor_ref_t *na = (ie_tensor_ref_t *)realloc(arr, cap * sizeof(*arr));
      if (!na) {
        for (size_t i = 0; i < cnt; i++) free(arr[i].name);
        free(arr);
        return IE_WDEDUP_ENOMEM;
      }
      arr = na;
      memset(arr + oldcap, 0, (cap - oldcap) * sizeof(*arr));
    }

    ie_wdedup_status_t st = ie_parse_one_tensor(&p, files, files_n, &arr[cnt]);
    if (st != IE_WDEDUP_OK) {
      for (size_t i = 0; i < cnt; i++) free(arr[i].name);
      free(arr);
      return st;
    }
    cnt++;

    p = js_skip_ws(p);
    if (*p == ',') { p++; p = js_skip_ws(p); continue; }
    if (*p == ']') break;

    for (size_t i = 0; i < cnt; i++) free(arr[i].name);
    free(arr);
    return IE_WDEDUP_EFORMAT;
  }

  *out = arr;
  *out_count = cnt;
  return IE_WDEDUP_OK;
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

ie_wdedup_status_t ie_weights_dedup_open(ie_weights_dedup_t **out,
                                        const char *model_dir,
                                        const ie_weights_dedup_opts_t *opts) {
  if (!out || !model_dir) return IE_WDEDUP_EINVAL;
  *out = NULL;

  ie_weights_dedup_t *h = (ie_weights_dedup_t *)calloc(1, sizeof(*h));
  if (!h) return IE_WDEDUP_ENOMEM;
  h->prefault_policy = opts ? opts->prefault_policy : 0;
  h->iebin.fd = -1;

  h->model_dir = ie_strdup(model_dir);
  if (!h->model_dir) { free(h); return IE_WDEDUP_ENOMEM; }

  /* Always attempt to parse model.dedup.json. If it is not schema1, fall back to schema2 diagnostics. */
  char *dedup_path = ie_path_join(model_dir, "model.dedup.json");
  if (!dedup_path) { ie_weights_dedup_close(&h); return IE_WDEDUP_ENOMEM; }

  size_t dedup_n = 0;
  char *dedup_json = ie_read_file_text(dedup_path, &dedup_n);
  free(dedup_path);
  if (!dedup_json) {
    ie_wdedup_logf("open: missing model.dedup.json under model_dir=%s", model_dir);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_ENOENT;
  }

  const char *files_val = NULL;
  const char *tensors_val = NULL;
  int has_schema1 = js_find_member_value(dedup_json, "files", &files_val) &&
                    js_find_member_value(dedup_json, "tensors", &tensors_val);

  if (has_schema1) {
    ie_file_ref_t *files = NULL;
    size_t files_n = 0;
    ie_wdedup_status_t st = ie_parse_files_object(files_val, &files, &files_n);
    if (st != IE_WDEDUP_OK) {
      ie_wdedup_logf("schema1: parse files failed status=%d (%s)", (int)st, ie_wdedup_status_str(st));
      free(dedup_json);
      ie_weights_dedup_close(&h);
      return st;
    }

    h->files = (ie_mmap_file_t *)calloc(files_n, sizeof(*h->files));
    if (!h->files) {
      for (size_t i = 0; i < files_n; i++) { free(files[i].logical); free(files[i].relpath); }
      free(files);
      free(dedup_json);
      ie_weights_dedup_close(&h);
      return IE_WDEDUP_ENOMEM;
    }
    h->files_count = files_n;

    ie_wdedup_status_t st_map = IE_WDEDUP_OK;
    for (size_t i = 0; i < files_n; i++) {
      char *full = ie_path_join(model_dir, files[i].relpath);
      if (!full) { st_map = IE_WDEDUP_ENOMEM; break; }
      st_map = ie_mmap_ro(&h->files[i], full, h->prefault_policy);
      if (st_map != IE_WDEDUP_OK) {
        ie_wdedup_logf("schema1: mmap file failed logical=%s rel=%s status=%d (%s)",
                       files[i].logical ? files[i].logical : "(null)",
                       files[i].relpath ? files[i].relpath : "(null)",
                       (int)st_map, ie_wdedup_status_str(st_map));
      }
      free(full);
      if (st_map != IE_WDEDUP_OK) break;
    }

    if (st_map == IE_WDEDUP_OK) {
      ie_wdedup_status_t st_t = ie_parse_tensors_array(tensors_val, files, files_n, &h->tensors, &h->tensors_count);
      if (st_t != IE_WDEDUP_OK) {
        ie_wdedup_logf("schema1: parse tensors failed status=%d (%s)", (int)st_t, ie_wdedup_status_str(st_t));
        st_map = st_t;
      } else if (h->tensors_count > 1) {
        qsort(h->tensors, h->tensors_count, sizeof(h->tensors[0]), ie_tensor_cmp_name);
      }
    }

    for (size_t i = 0; i < files_n; i++) { free(files[i].logical); free(files[i].relpath); }
    free(files);
    free(dedup_json);

    if (st_map != IE_WDEDUP_OK) {
      ie_weights_dedup_close(&h);
      return st_map;
    }

    *out = h;
    return IE_WDEDUP_OK;
  }

  /* ---------------------------------------------------------------------- */
  /* Schema2 fallback: parse model.ie.json tensors for detailed diagnostics. */
  /* This is where your current failure happens; this implementation is      */
  /* strict about JSON structure but tolerant about dtype.                   */
  /* ---------------------------------------------------------------------- */

  free(dedup_json);

  char *iejson_path = ie_path_join(model_dir, "model.ie.json");
  if (!iejson_path) { ie_weights_dedup_close(&h); return IE_WDEDUP_ENOMEM; }

  size_t iejson_n = 0;
  char *iejson = ie_read_file_text(iejson_path, &iejson_n);
  if (!iejson) {
    ie_wdedup_logf("schema2: failed to read %s errno=%d (%s)", iejson_path, errno, ie_wdedup_errno_str(errno));
    free(iejson_path);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_ENOENT;
  }
  free(iejson_path);

  ie_wdedup_status_t st_ie = ie_parse_iejson_tensors(iejson, &h->iejson_tensors, &h->iejson_tensors_count);
  if (st_ie != IE_WDEDUP_OK) {
    ie_wdedup_logf("schema2: parse model.ie.json tensors failed status=%d (%s)", (int)st_ie, ie_wdedup_status_str(st_ie));
    free(iejson);
    ie_weights_dedup_close(&h);
    return st_ie;
  }

  ie_wdedup_logf("schema2: parsed model.ie.json tensors=%zu", h->iejson_tensors_count);

  /* Also mmap model.ie.bin so future schema2 can provide direct views if needed. */
  char *iebin_path = ie_path_join(model_dir, "model.ie.bin");
  if (!iebin_path) { free(iejson); ie_weights_dedup_close(&h); return IE_WDEDUP_ENOMEM; }

  ie_wdedup_status_t st_bin = ie_mmap_ro(&h->iebin, iebin_path, h->prefault_policy);
  free(iebin_path);
  if (st_bin != IE_WDEDUP_OK) {
    ie_wdedup_logf("schema2: mmap model.ie.bin failed status=%d (%s)", (int)st_bin, ie_wdedup_status_str(st_bin));
    free(iejson);
    ie_weights_dedup_close(&h);
    return st_bin;
  }

  /* For schema2, require the three blobs to exist (strict behavior). */
  char *p_def = ie_path_join(model_dir, "model.defaults.bin");
  char *p_msk = ie_path_join(model_dir, "model.masks.bin");
  char *p_exc = ie_path_join(model_dir, "model.exceptions.bin");
  if (!p_def || !p_msk || !p_exc) {
    free(p_def); free(p_msk); free(p_exc);
    free(iejson);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_ENOMEM;
  }

  /* Map blobs into files[] so IE_WVIEW_DEDUP views can be added later. */
  h->files = (ie_mmap_file_t *)calloc(3, sizeof(*h->files));
  if (!h->files) {
    free(p_def); free(p_msk); free(p_exc);
    free(iejson);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_ENOMEM;
  }
  h->files_count = 3;

  ie_wdedup_status_t st0 = ie_mmap_ro(&h->files[0], p_def, h->prefault_policy);
  ie_wdedup_status_t st1 = ie_mmap_ro(&h->files[1], p_msk, h->prefault_policy);
  ie_wdedup_status_t st2 = ie_mmap_ro(&h->files[2], p_exc, h->prefault_policy);

  free(p_def); free(p_msk); free(p_exc);
  free(iejson);

  if (st0 != IE_WDEDUP_OK || st1 != IE_WDEDUP_OK || st2 != IE_WDEDUP_OK) {
    ie_wdedup_logf("schema2: mmap blobs failed status=%d (%s)",
                   (int)(st0 != IE_WDEDUP_OK ? st0 : (st1 != IE_WDEDUP_OK ? st1 : st2)),
                   ie_wdedup_status_str(st0 != IE_WDEDUP_OK ? st0 : (st1 != IE_WDEDUP_OK ? st1 : st2)));
    ie_weights_dedup_close(&h);
    return (st0 != IE_WDEDUP_OK ? st0 : (st1 != IE_WDEDUP_OK ? st1 : st2));
  }

  /* NOTE: schema2 mapping (iejson + tensor_map + groups) is not implemented here.
   * This open() now succeeds and enforces the presence of the three blobs, and
   * provides extremely detailed diagnostics for why model.ie.json parsing fails.
   * Weight views for schema2 are expected to be implemented on top of this (next step).
   */

  *out = h;
  return IE_WDEDUP_OK;
}

void ie_weights_dedup_close(ie_weights_dedup_t **h) {
  if (!h || !*h) return;
  ie_weights_dedup_t *x = *h;

  if (x->tensors) {
    for (size_t i = 0; i < x->tensors_count; i++) free(x->tensors[i].name);
    free(x->tensors);
  }

  if (x->files) {
    for (size_t i = 0; i < x->files_count; i++) ie_mmap_close(&x->files[i]);
    free(x->files);
  }

  if (x->iejson_tensors) {
    for (size_t i = 0; i < x->iejson_tensors_count; i++) ie_iejson_tensor_free(&x->iejson_tensors[i]);
    free(x->iejson_tensors);
  }

  ie_mmap_close(&x->iebin);

  free(x->model_dir);
  free(x);
  *h = NULL;
}

ie_wdedup_status_t ie_weights_dedup_get_weight_view(const ie_weights_dedup_t *h,
                                                    const char *name,
                                                    ie_weight_view_t *out) {
  if (!h || !name || !out) return IE_WDEDUP_EINVAL;

  const ie_tensor_ref_t *tr = ie_find_tensor(h, name);
  if (!tr) return IE_WDEDUP_ENOENT;

  if (tr->file_defaults >= h->files_count) return IE_WDEDUP_ERANGE;
  if (tr->file_mask >= h->files_count) return IE_WDEDUP_ERANGE;
  if (tr->file_exceptions >= h->files_count) return IE_WDEDUP_ERANGE;

  const ie_mmap_file_t *fdef = &h->files[tr->file_defaults];
  const ie_mmap_file_t *fmsk = &h->files[tr->file_mask];
  const ie_mmap_file_t *fexc = &h->files[tr->file_exceptions];

  if (tr->off_defaults + tr->nbytes_defaults > fdef->size) return IE_WDEDUP_ERANGE;
  if (tr->off_mask + tr->nbytes_mask > fmsk->size) return IE_WDEDUP_ERANGE;
  if (tr->off_exceptions + tr->nbytes_exceptions > fexc->size) return IE_WDEDUP_ERANGE;

  memset(out, 0, sizeof(*out));
  out->kind = IE_WVIEW_DEDUP;
  out->dtype = tr->dtype;
  out->rank = tr->rank;
  for (int32_t i = 0; i < tr->rank; i++) out->shape[i] = tr->shape[i];

  out->defaults = (const uint8_t *)fdef->base + (size_t)tr->off_defaults;
  out->defaults_nbytes = (size_t)tr->nbytes_defaults;

  out->mask = (const uint8_t *)fmsk->base + (size_t)tr->off_mask;
  out->mask_nbytes = (size_t)tr->nbytes_mask;

  out->exceptions = (const uint8_t *)fexc->base + (size_t)tr->off_exceptions;
  out->exceptions_nbytes = (size_t)tr->nbytes_exceptions;

  out->elem_count = tr->elem_count;
  out->elem_size = tr->elem_size;

  if (out->mask_nbytes == 0 && out->exceptions_nbytes == 0) {
    out->kind = IE_WVIEW_DIRECT;
    out->data = out->defaults;
    out->nbytes = out->defaults_nbytes;
    out->defaults = NULL;
    out->mask = NULL;
    out->exceptions = NULL;
    out->defaults_nbytes = 0;
    out->mask_nbytes = 0;
    out->exceptions_nbytes = 0;
  }

  return IE_WDEDUP_OK;
}

/* -------------------------------------------------------------------------- */
/* Materialization helpers                                                    */
/* -------------------------------------------------------------------------- */

static inline int ie_bitset_get(const uint8_t *bits, size_t i) {
  return (bits[i >> 3] >> (unsigned)(i & 7u)) & 1u;
}

static inline uint8_t ie_int4_get_nibble(const uint8_t *p, size_t idx) {
  uint8_t b = p[idx >> 1];
  if ((idx & 1u) == 0) return (uint8_t)(b & 0x0F);
  return (uint8_t)((b >> 4) & 0x0F);
}

static inline void ie_int4_set_nibble(uint8_t *p, size_t idx, uint8_t v) {
  uint8_t *b = &p[idx >> 1];
  v &= 0x0F;
  if ((idx & 1u) == 0) *b = (uint8_t)((*b & 0xF0) | v);
  else *b = (uint8_t)((*b & 0x0F) | (uint8_t)(v << 4));
}

size_t ie_weights_dedup_materialize(const ie_weight_view_t *view, void *dst, size_t dst_nbytes) {
  if (!view || !dst) return 0;

  if (view->kind == IE_WVIEW_DIRECT) {
    if (!view->data) return 0;
    if (view->nbytes > dst_nbytes) return 0;
    memcpy(dst, view->data, view->nbytes);
    return view->nbytes;
  }

  if (view->kind != IE_WVIEW_DEDUP) return 0;
  if (!view->defaults || !view->mask) return 0;

  if (view->dtype != IE_WDTYPE_INT4) {
    if (view->elem_size == 0) return 0;

    size_t need = view->elem_count * view->elem_size;
    if (need > dst_nbytes) return 0;
    if (view->defaults_nbytes < need) return 0;

    memcpy(dst, view->defaults, need);

    const uint8_t *exc = (const uint8_t *)view->exceptions;
    size_t exc_off = 0;

    for (size_t i = 0; i < view->elem_count; i++) {
      if (!ie_bitset_get(view->mask, i)) continue;
      if (exc_off + view->elem_size > view->exceptions_nbytes) return 0;

      uint8_t *dst_b = (uint8_t *)dst + i * view->elem_size;
      memcpy(dst_b, exc + exc_off, view->elem_size);
      exc_off += view->elem_size;
    }

    return need;
  }

  /* INT4 nibble-packed */
  {
    size_t need = view->defaults_nbytes;
    if (need > dst_nbytes) return 0;
    if (view->defaults_nbytes == 0) return 0;

    memcpy(dst, view->defaults, need);

    const uint8_t *exc_bytes = (const uint8_t *)view->exceptions;
    size_t exc_nibbles = view->exceptions_nbytes * 2;
    size_t exc_idx = 0;

    for (size_t i = 0; i < view->elem_count; i++) {
      if (!ie_bitset_get(view->mask, i)) continue;

      if (exc_idx >= exc_nibbles) return 0;
      uint8_t v = ie_int4_get_nibble(exc_bytes, exc_idx++);
      ie_int4_set_nibble((uint8_t *)dst, i, v);
    }

    return need;
  }
}
