/* engine/src/io/weights_dedup.c */
/**
 * @file weights_dedup.c
 * @brief Lossless deduplicated-weights loader implementation.
 *
 * Design goals:
 *  - mmap-friendly: map large blobs once, return pointer views
 *  - low overhead lookup: sorted tensor table + binary search
 *  - lossless: exact reconstruction via mask + exceptions stream
 *
 * This module intentionally uses a minimal, self-contained JSON scanner that supports
 * the subset of JSON needed for `model.dedup.json`. This avoids hard dependencies on
 * external JSON libraries and keeps build integration simple.
 *
 * Expected JSON shape (subset):
 *
 * {
 *   "files": {
 *     "defaults": "model.defaults.bin",
 *     "exceptions": "model.exceptions.bin",
 *     "masks": "model.masks.bin"
 *   },
 *   "tensors": [
 *     {
 *       "name": "layers.0.attn.q_proj.weight",
 *       "dtype": "fp16",
 *       "shape": [4096, 4096],
 *       "default":    {"file":"defaults",   "offset":123, "nbytes":456},
 *       "mask":       {"file":"masks",      "offset":789, "nbytes":1011},
 *       "exceptions": {"file":"exceptions", "offset":1213,"nbytes":1415}
 *     }
 *   ]
 * }
 *
 * Mask semantics:
 *  - bit=1 indicates an exception element is present and consumed from exceptions stream.
 *  - For int4, elements are nibbles in row-major order (index i maps to byte i/2, nibble i%2).
 */

#include "weights_dedup.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ----------------------------- Internal helpers ----------------------------- */

/**
 * @brief ISO C replacement for strdup(3).
 *
 * Many toolchains require feature macros (or non-ISO extensions) for `strdup`.
 * This project builds with `-std=c11 -Werror -pedantic`, so we provide a tiny,
 * portable duplicate helper.
 *
 * @param[in] s Input string (may be NULL).
 * @return Newly allocated duplicate (must be free'd), or NULL on OOM.
 */
static char *ie_strdup(const char *s) {
  const char *src = s ? s : "";
  size_t n = strlen(src);
  char *out = (char *)malloc(n + 1);
  if (!out) return NULL;
  memcpy(out, src, n);
  out[n] = '\0';
  return out;
}

/**
 * @brief Read an entire file into memory (NUL-terminated).
 *
 * @param[in]  path Path to file.
 * @param[out] out_size Returned size in bytes (excluding NUL).
 * @return Newly allocated buffer (must be free'd) or NULL on error.
 */
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

/**
 * @brief Join directory + relative filename into a newly allocated path.
 *
 * @param[in] dir Directory path (no trailing slash required).
 * @param[in] rel Relative filename.
 * @return Newly allocated string or NULL.
 */
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

/**
 * @brief Best-effort madvise to improve paging behavior.
 *
 * @param[in] addr Mapped base.
 * @param[in] len  Mapped size.
 * @param[in] policy 0=none, 1=sequential, 2=willneed.
 */
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

/**
 * @brief A single mmapped file.
 */
typedef struct ie_mmap_file {
  char *path;
  int fd;
  void *base;
  size_t size;
} ie_mmap_file_t;

/**
 * @brief Open and mmap a file read-only.
 *
 * @param[out] mf     File mapping to fill.
 * @param[in]  path   Absolute path to file.
 * @param[in]  policy Prefault/advice policy.
 * @return IE_WDEDUP_OK on success.
 */
static ie_wdedup_status_t ie_mmap_ro(ie_mmap_file_t *mf, const char *path, int policy) {
  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;

  mf->path = ie_strdup(path ? path : "");
  if (!mf->path) return IE_WDEDUP_ENOMEM;

  mf->fd = open(path, O_RDONLY);
  if (mf->fd < 0) return IE_WDEDUP_ENOENT;

  struct stat st;
  if (fstat(mf->fd, &st) != 0) return IE_WDEDUP_EIO;
  if (st.st_size < 0) return IE_WDEDUP_EIO;
  mf->size = (size_t)st.st_size;

  if (mf->size == 0) {
    mf->base = NULL;
    return IE_WDEDUP_OK;
  }

  void *p = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
  if (p == MAP_FAILED) return IE_WDEDUP_EIO;

  mf->base = p;
  ie_madvise_policy(mf->base, mf->size, policy);
  return IE_WDEDUP_OK;
}

/**
 * @brief Unmap and close a mapping.
 *
 * @param[in,out] mf Mapping to close.
 */
static void ie_mmap_close(ie_mmap_file_t *mf) {
  if (!mf) return;
  if (mf->base && mf->size) (void)munmap(mf->base, mf->size);
  if (mf->fd >= 0) (void)close(mf->fd);
  free(mf->path);
  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;
}

/* ----------------------------- Minimal JSON scan ---------------------------- */

/**
 * @brief Skip JSON whitespace.
 *
 * @param[in] p Cursor.
 * @return Cursor advanced past whitespace.
 */
static const char *js_skip_ws(const char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  return p;
}

/**
 * @brief Match a JSON literal.
 *
 * @param[in] p Cursor.
 * @param[in] lit Literal to match (e.g., "true", "null").
 * @return Non-zero if matched; 0 otherwise.
 */
static int js_match_lit(const char *p, const char *lit) {
  size_t n = strlen(lit);
  return (strncmp(p, lit, n) == 0);
}

/**
 * @brief Parse a JSON string (no full unicode support; supports common escapes).
 *
 * @param[in,out] p Cursor.
 * @param[out] out Newly allocated C string.
 * @return 1 on success, 0 on failure.
 */
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

/**
 * @brief Parse a JSON integer (int64).
 *
 * @param[in,out] p Cursor.
 * @param[out] out Parsed value.
 * @return 1 on success, 0 on failure.
 */
static int js_parse_i64(const char **p, int64_t *out) {
  const char *s = js_skip_ws(*p);
  int neg = 0;
  if (*s == '-') {
    neg = 1;
    s++;
  }
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

/**
 * @brief Skip a generic JSON value (object/array/string/number/literal).
 *
 * @param[in,out] p Cursor.
 * @return 1 on success, 0 on failure.
 */
static int js_skip_value(const char **p);

/**
 * @brief Skip a JSON array.
 *
 * @param[in,out] p Cursor.
 * @return 1 on success, 0 on failure.
 */
static int js_skip_array(const char **p) {
  const char *s = js_skip_ws(*p);
  if (*s != '[') return 0;
  s++;
  s = js_skip_ws(s);
  if (*s == ']') {
    *p = s + 1;
    return 1;
  }
  for (;;) {
    *p = s;
    if (!js_skip_value(p)) return 0;
    s = js_skip_ws(*p);
    if (*s == ',') {
      s++;
      s = js_skip_ws(s);
      continue;
    }
    if (*s == ']') {
      *p = s + 1;
      return 1;
    }
    return 0;
  }
}

/**
 * @brief Skip a JSON object.
 *
 * @param[in,out] p Cursor.
 * @return 1 on success, 0 on failure.
 */
static int js_skip_object(const char **p) {
  const char *s = js_skip_ws(*p);
  if (*s != '{') return 0;
  s++;
  s = js_skip_ws(s);
  if (*s == '}') {
    *p = s + 1;
    return 1;
  }
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
    if (*s == ',') {
      s++;
      s = js_skip_ws(s);
      continue;
    }
    if (*s == '}') {
      *p = s + 1;
      return 1;
    }
    return 0;
  }
}

/**
 * @brief Skip a JSON value dispatch (string/object/array/number/literal).
 *
 * @param[in,out] p Cursor.
 * @return 1 on success, 0 on failure.
 */
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
  if (*s == '{') {
    *p = s;
    return js_skip_object(p);
  }
  if (*s == '[') {
    *p = s;
    return js_skip_array(p);
  }
  if (*s == '-' || (*s >= '0' && *s <= '9')) {
    int64_t v = 0;
    *p = s;
    if (!js_parse_i64(p, &v)) return 0;
    return 1;
  }
  if (js_match_lit(s, "true")) {
    *p = s + 4;
    return 1;
  }
  if (js_match_lit(s, "false")) {
    *p = s + 5;
    return 1;
  }
  if (js_match_lit(s, "null")) {
    *p = s + 4;
    return 1;
  }
  return 0;
}

/**
 * @brief Find a top-level object member by key and return cursor at its value.
 *
 * @param[in]  json Cursor at start of object ('{').
 * @param[in]  key  Member name to find.
 * @param[out] out_val Cursor set to start of member value on success.
 * @return 1 if found, 0 if not found or invalid.
 */
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
    if (*p != ':') {
      free(k);
      return 0;
    }
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
    if (*p == ',') {
      p++;
      p = js_skip_ws(p);
      continue;
    }
    if (*p == '}') return 0;
    return 0;
  }
}

/* --------------------------- Dedup data structures -------------------------- */

/**
 * @brief Logical file reference parsed from the top-level `files` object.
 */
typedef struct ie_file_ref {
  char *logical;  /* "defaults" | "exceptions" | "masks" */
  char *relpath;  /* relative path in model dir */
} ie_file_ref_t;

/**
 * @brief Per-tensor metadata referencing defaults/masks/exceptions blobs.
 */
typedef struct ie_tensor_ref {
  char *name;
  ie_wdtype_t dtype;

  int32_t rank;
  int64_t shape[IE_WDEDUP_MAX_RANK];

  /* Each blob ref points into one of the mapped files */
  uint32_t file_defaults;
  uint64_t off_defaults;
  uint64_t nbytes_defaults;

  uint32_t file_mask;
  uint64_t off_mask;
  uint64_t nbytes_mask;

  uint32_t file_exceptions;
  uint64_t off_exceptions;
  uint64_t nbytes_exceptions;

  /* Derived */
  size_t elem_count;
  size_t elem_size;
} ie_tensor_ref_t;

/**
 * @brief Opaque handle for the deduplicated weights container.
 */
struct ie_weights_dedup {
  char *model_dir;

  ie_mmap_file_t *files;
  size_t files_count;

  ie_tensor_ref_t *tensors;
  size_t tensors_count;

  int prefault_policy;
};

/* ----------------------------- Dtype utilities ------------------------------ */

/**
 * @brief Parse a dtype string to an `ie_wdtype_t`.
 *
 * @param[in] s Dtype string (e.g., "fp16", "int4").
 * @return Parsed dtype enum value (or IE_WDTYPE_UNKNOWN).
 */
ie_wdtype_t ie_weights_dedup_parse_dtype(const char *s) {
  if (!s) return IE_WDTYPE_UNKNOWN;
  if (strcmp(s, "fp32") == 0) return IE_WDTYPE_FP32;
  if (strcmp(s, "fp16") == 0) return IE_WDTYPE_FP16;
  if (strcmp(s, "bf16") == 0) return IE_WDTYPE_BF16;
  if (strcmp(s, "int8") == 0) return IE_WDTYPE_INT8;
  if (strcmp(s, "int4") == 0) return IE_WDTYPE_INT4;
  return IE_WDTYPE_UNKNOWN;
}

/**
 * @brief Element size in bytes for a dtype (int4 returns 0 because nibble-based).
 *
 * @param[in] dt Data type enum.
 * @return Size in bytes for one element (0 for INT4 / unknown).
 */
static size_t ie_dtype_elem_size(ie_wdtype_t dt) {
  switch (dt) {
    case IE_WDTYPE_FP32: return 4;
    case IE_WDTYPE_FP16: return 2;
    case IE_WDTYPE_BF16: return 2;
    case IE_WDTYPE_INT8: return 1;
    case IE_WDTYPE_INT4: return 0;
    default: return 0;
  }
}

/**
 * @brief Compute element count from shape (product of dims).
 *
 * @param[in]  shape Dimension array.
 * @param[in]  rank  Number of dimensions in @p shape.
 * @param[out] out   Element count on success.
 * @return 1 on success, 0 on overflow/invalid.
 */
static int ie_shape_elem_count(const int64_t *shape, int32_t rank, size_t *out) {
  if (!shape || !out || rank < 0 || rank > IE_WDEDUP_MAX_RANK) return 0;
  size_t n = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (shape[i] < 0) return 0;
    if (shape[i] == 0) {
      n = 0;
      break;
    }
    if (n > (SIZE_MAX / (size_t)shape[i])) return 0;
    n *= (size_t)shape[i];
  }
  *out = n;
  return 1;
}

/* ------------------------- Tensor table / lookup ---------------------------- */

/**
 * @brief Compare tensor refs by name (qsort/bsearch).
 *
 * @param[in] a Pointer to ie_tensor_ref_t.
 * @param[in] b Pointer to ie_tensor_ref_t.
 * @return strcmp-style ordering value.
 */
static int ie_tensor_cmp_name(const void *a, const void *b) {
  const ie_tensor_ref_t *A = (const ie_tensor_ref_t *)a;
  const ie_tensor_ref_t *B = (const ie_tensor_ref_t *)b;
  return strcmp(A->name, B->name);
}

/**
 * @brief Binary-search tensor table.
 *
 * @param[in] h Handle.
 * @param[in] name Tensor name.
 * @return Pointer to tensor ref on success, NULL if not found/invalid.
 */
static const ie_tensor_ref_t *ie_find_tensor(const ie_weights_dedup_t *h, const char *name) {
  if (!h || !name || !h->tensors || h->tensors_count == 0) return NULL;

  ie_tensor_ref_t key;
  memset(&key, 0, sizeof(key));
  key.name = (char *)name;

  return (const ie_tensor_ref_t *)bsearch(&key, h->tensors, h->tensors_count,
                                          sizeof(h->tensors[0]), ie_tensor_cmp_name);
}

/* ------------------------------- JSON parsing ------------------------------- */

/**
 * @brief Parse a shape array: [d0, d1, ...]
 *
 * @param[in,out] p Cursor (points at '[' on entry).
 * @param[out] shape Output dimension array.
 * @param[out] rank Output rank.
 * @return 1 on success, 0 on failure.
 */
static int js_parse_shape(const char **p, int64_t *shape, int32_t *rank) {
  const char *s = js_skip_ws(*p);
  if (*s != '[') return 0;
  s++;
  s = js_skip_ws(s);

  int32_t r = 0;
  if (*s == ']') {
    *p = s + 1;
    *rank = 0;
    return 1;
  }

  for (;;) {
    if (r >= IE_WDEDUP_MAX_RANK) return 0;
    int64_t v = 0;
    const char *tmp = s;
    if (!js_parse_i64(&tmp, &v)) return 0;
    shape[r++] = v;
    s = js_skip_ws(tmp);

    if (*s == ',') {
      s++;
      s = js_skip_ws(s);
      continue;
    }
    if (*s == ']') {
      *p = s + 1;
      *rank = r;
      return 1;
    }
    return 0;
  }
}

/**
 * @brief Parse a blob ref object: {"file":"defaults","offset":N,"nbytes":M}
 *
 * @param[in,out] p Cursor (points at '{' on entry).
 * @param[out] out_file Allocated logical file name (must be free'd).
 * @param[out] out_off Offset (int64).
 * @param[out] out_nb  Size in bytes (int64).
 * @return 1 on success, 0 on failure.
 */
static int js_parse_blobref(const char **p, char **out_file, int64_t *out_off, int64_t *out_nb) {
  const char *s = js_skip_ws(*p);
  if (*s != '{') return 0;

  const char *val = NULL;
  if (!js_find_member_value(s, "file", &val)) return 0;
  if (!js_parse_string(&val, out_file)) return 0;

  val = NULL;
  if (!js_find_member_value(s, "offset", &val)) return 0;
  if (!js_parse_i64(&val, out_off)) return 0;

  val = NULL;
  if (!js_find_member_value(s, "nbytes", &val)) return 0;
  if (!js_parse_i64(&val, out_nb)) return 0;

  /* advance cursor past object */
  *p = s;
  return js_skip_object(p);
}

/**
 * @brief Parse the top-level "files" object into logical->relpath table.
 *
 * Supports only string values.
 *
 * @param[in]  json_obj Cursor at '{' for files object.
 * @param[out] out Allocated array of file refs (caller frees elements + array).
 * @param[out] out_count Number of entries.
 * @return Status code.
 */
static ie_wdedup_status_t ie_parse_files_object(const char *json_obj,
                                                ie_file_ref_t **out,
                                                size_t *out_count) {
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
    if (!js_parse_string(&p, &k)) { free(arr); return IE_WDEDUP_EFORMAT; }
    p = js_skip_ws(p);
    if (*p != ':') { free(k); free(arr); return IE_WDEDUP_EFORMAT; }
    p++;
    p = js_skip_ws(p);

    char *v = NULL;
    if (!js_parse_string(&p, &v)) { free(k); free(arr); return IE_WDEDUP_EFORMAT; }

    if (cnt == cap) {
      cap *= 2;
      ie_file_ref_t *na = (ie_file_ref_t *)realloc(arr, cap * sizeof(*arr));
      if (!na) { free(k); free(v); free(arr); return IE_WDEDUP_ENOMEM; }
      arr = na;
      memset(arr + cnt, 0, (cap - cnt) * sizeof(*arr));
    }

    arr[cnt].logical = k;
    arr[cnt].relpath = v;
    cnt++;

    p = js_skip_ws(p);
    if (*p == ',') {
      p++;
      p = js_skip_ws(p);
      continue;
    }
    if (*p == '}') {
      break;
    }
    for (size_t i = 0; i < cnt; i++) { free(arr[i].logical); free(arr[i].relpath); }
    free(arr);
    return IE_WDEDUP_EFORMAT;
  }

  *out = arr;
  *out_count = cnt;
  return IE_WDEDUP_OK;
}

/**
 * @brief Find index of a logical file name in file refs.
 *
 * @param[in]  refs File ref array.
 * @param[in]  n Number of entries in @p refs.
 * @param[in]  logical Logical name to find.
 * @param[out] out_idx Index on success.
 * @return 1 if found, 0 otherwise.
 */
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

/**
 * @brief Parse one tensor object within the "tensors" array.
 *
 * @param[in,out] p Cursor (points at '{' on entry; advanced past object on success).
 * @param[in] files Parsed files table.
 * @param[in] files_n Number of entries in @p files.
 * @param[out] out Parsed tensor reference.
 * @return Status code.
 */
static ie_wdedup_status_t ie_parse_one_tensor(const char **p,
                                              const ie_file_ref_t *files,
                                              size_t files_n,
                                              ie_tensor_ref_t *out) {
  memset(out, 0, sizeof(*out));
  out->dtype = IE_WDTYPE_UNKNOWN;

  const char *s = js_skip_ws(*p);
  if (*s != '{') return IE_WDEDUP_EFORMAT;

  const char *val = NULL;

  /* name */
  val = NULL;
  if (!js_find_member_value(s, "name", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_string(&val, &out->name)) return IE_WDEDUP_EFORMAT;

  /* dtype */
  char *dtype_s = NULL;
  val = NULL;
  if (!js_find_member_value(s, "dtype", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_string(&val, &dtype_s)) return IE_WDEDUP_EFORMAT;
  out->dtype = ie_weights_dedup_parse_dtype(dtype_s);
  free(dtype_s);
  if (out->dtype == IE_WDTYPE_UNKNOWN) return IE_WDEDUP_EFORMAT;

  /* shape */
  val = NULL;
  if (!js_find_member_value(s, "shape", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_shape(&val, out->shape, &out->rank)) return IE_WDEDUP_EFORMAT;

  /* default/mask/exceptions blob refs */
  char *f_default = NULL, *f_mask = NULL, *f_exc = NULL;
  int64_t off_default = 0, nb_default = 0;
  int64_t off_mask = 0, nb_mask = 0;
  int64_t off_exc = 0, nb_exc = 0;

  val = NULL;
  if (!js_find_member_value(s, "default", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_blobref(&val, &f_default, &off_default, &nb_default)) return IE_WDEDUP_EFORMAT;

  val = NULL;
  if (!js_find_member_value(s, "mask", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_blobref(&val, &f_mask, &off_mask, &nb_mask)) return IE_WDEDUP_EFORMAT;

  val = NULL;
  if (!js_find_member_value(s, "exceptions", &val)) return IE_WDEDUP_EFORMAT;
  if (!js_parse_blobref(&val, &f_exc, &off_exc, &nb_exc)) return IE_WDEDUP_EFORMAT;

  uint32_t idx_def = 0, idx_mask = 0, idx_exc = 0;
  if (!ie_find_file_index(files, files_n, f_default, &idx_def)) { free(f_default); free(f_mask); free(f_exc); return IE_WDEDUP_EFORMAT; }
  if (!ie_find_file_index(files, files_n, f_mask, &idx_mask))   { free(f_default); free(f_mask); free(f_exc); return IE_WDEDUP_EFORMAT; }
  if (!ie_find_file_index(files, files_n, f_exc, &idx_exc))     { free(f_default); free(f_mask); free(f_exc); return IE_WDEDUP_EFORMAT; }

  free(f_default);
  free(f_mask);
  free(f_exc);

  if (off_default < 0 || nb_default < 0 || off_mask < 0 || nb_mask < 0 || off_exc < 0 || nb_exc < 0) {
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

  /* Derived element info */
  out->elem_size = ie_dtype_elem_size(out->dtype);
  if (!ie_shape_elem_count(out->shape, out->rank, &out->elem_count)) return IE_WDEDUP_ERANGE;

  /* advance cursor past object */
  *p = s;
  if (!js_skip_object(p)) return IE_WDEDUP_EFORMAT;
  return IE_WDEDUP_OK;
}

/**
 * @brief Parse the "tensors" array into an allocated table.
 *
 * @param[in]  json_val Cursor at '[' for the tensors array.
 * @param[in]  files Parsed files table.
 * @param[in]  files_n Number of entries in @p files.
 * @param[out] out Allocated tensor array (caller frees names + array).
 * @param[out] out_count Number of parsed tensors.
 * @return Status code.
 */
static ie_wdedup_status_t ie_parse_tensors_array(const char *json_val,
                                                 const ie_file_ref_t *files,
                                                 size_t files_n,
                                                 ie_tensor_ref_t **out,
                                                 size_t *out_count) {
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
      cap *= 2;
      ie_tensor_ref_t *na = (ie_tensor_ref_t *)realloc(arr, cap * sizeof(*arr));
      if (!na) {
        for (size_t i = 0; i < cnt; i++) free(arr[i].name);
        free(arr);
        return IE_WDEDUP_ENOMEM;
      }
      arr = na;
      memset(arr + cnt, 0, (cap - cnt) * sizeof(*arr));
    }

    ie_wdedup_status_t st = ie_parse_one_tensor(&p, files, files_n, &arr[cnt]);
    if (st != IE_WDEDUP_OK) {
      for (size_t i = 0; i <= cnt; i++) free(arr[i].name);
      free(arr);
      return st;
    }
    cnt++;

    p = js_skip_ws(p);
    if (*p == ',') {
      p++;
      p = js_skip_ws(p);
      continue;
    }
    if (*p == ']') break;

    for (size_t i = 0; i < cnt; i++) free(arr[i].name);
    free(arr);
    return IE_WDEDUP_EFORMAT;
  }

  *out = arr;
  *out_count = cnt;
  return IE_WDEDUP_OK;
}

/* ------------------------------- Public API -------------------------------- */

/**
 * @brief Open a deduplicated weights handle from a model directory.
 *
 * Reads `model.dedup.json`, mmaps the referenced backing files, parses the tensor
 * table, and sorts it by tensor name for fast binary-search lookups.
 *
 * @param[out] out Output handle.
 * @param[in]  model_dir Directory containing `model.dedup.json` and backing files.
 * @param[in]  opts Options (may be NULL).
 * @return Status code.
 */
ie_wdedup_status_t ie_weights_dedup_open(ie_weights_dedup_t **out,
                                        const char *model_dir,
                                        const ie_weights_dedup_opts_t *opts) {
  if (!out || !model_dir) return IE_WDEDUP_EINVAL;
  *out = NULL;

  ie_weights_dedup_t *h = (ie_weights_dedup_t *)calloc(1, sizeof(*h));
  if (!h) return IE_WDEDUP_ENOMEM;
  h->prefault_policy = opts ? opts->prefault_policy : 0;

  h->model_dir = ie_strdup(model_dir);
  if (!h->model_dir) { free(h); return IE_WDEDUP_ENOMEM; }

  char *json_path = ie_path_join(model_dir, "model.dedup.json");
  if (!json_path) { ie_weights_dedup_close(&h); return IE_WDEDUP_ENOMEM; }

  size_t json_n = 0;
  char *json = ie_read_file_text(json_path, &json_n);
  free(json_path);
  if (!json) { ie_weights_dedup_close(&h); return IE_WDEDUP_ENOENT; }

  /* parse top-level files */
  const char *files_val = NULL;
  if (!js_find_member_value(json, "files", &files_val)) {
    free(json);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_EFORMAT;
  }

  ie_file_ref_t *files = NULL;
  size_t files_n = 0;
  ie_wdedup_status_t st = ie_parse_files_object(files_val, &files, &files_n);
  if (st != IE_WDEDUP_OK) {
    free(json);
    ie_weights_dedup_close(&h);
    return st;
  }

  /* mmap each file */
  h->files = (ie_mmap_file_t *)calloc(files_n, sizeof(*h->files));
  if (!h->files) {
    for (size_t i = 0; i < files_n; i++) { free(files[i].logical); free(files[i].relpath); }
    free(files);
    free(json);
    ie_weights_dedup_close(&h);
    return IE_WDEDUP_ENOMEM;
  }
  h->files_count = files_n;

  for (size_t i = 0; i < files_n; i++) {
    char *full = ie_path_join(model_dir, files[i].relpath);
    if (!full) { st = IE_WDEDUP_ENOMEM; break; }

    st = ie_mmap_ro(&h->files[i], full, h->prefault_policy);
    free(full);
    if (st != IE_WDEDUP_OK) break;
  }

  /* parse tensors */
  if (st == IE_WDEDUP_OK) {
    const char *tensors_val = NULL;
    if (!js_find_member_value(json, "tensors", &tensors_val)) {
      st = IE_WDEDUP_EFORMAT;
    } else {
      st = ie_parse_tensors_array(tensors_val, files, files_n, &h->tensors, &h->tensors_count);
      if (st == IE_WDEDUP_OK && h->tensors_count > 1) {
        qsort(h->tensors, h->tensors_count, sizeof(h->tensors[0]), ie_tensor_cmp_name);
      }
    }
  }

  /* cleanup file-ref table + json buffer */
  for (size_t i = 0; i < files_n; i++) { free(files[i].logical); free(files[i].relpath); }
  free(files);
  free(json);

  if (st != IE_WDEDUP_OK) {
    ie_weights_dedup_close(&h);
    return st;
  }

  *out = h;
  return IE_WDEDUP_OK;
}

/**
 * @brief Close a deduplicated weights handle and release resources.
 *
 * This unmaps all backing files, frees tensor metadata, and frees the handle.
 *
 * @param[in,out] h Handle pointer (set to NULL on return).
 */
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

  free(x->model_dir);
  free(x);
  *h = NULL;
}

/**
 * @brief Look up a tensor by name and return a weight view describing how to read it.
 *
 * For deduplicated tensors, the returned view points at:
 *  - a defaults byte-range (base tensor data),
 *  - a bitmask indicating which elements are overridden,
 *  - an exceptions stream providing the replacement values.
 *
 * If a tensor has no mask/exceptions, the view is returned as IE_WVIEW_DIRECT.
 *
 * @param[in]  h Handle.
 * @param[in]  name Tensor name.
 * @param[out] out Output view descriptor.
 * @return Status code.
 */
ie_wdedup_status_t ie_weights_dedup_get_weight_view(const ie_weights_dedup_t *h,
                                                    const char *name,
                                                    ie_weight_view_t *out) {
  if (!h || !name || !out) return IE_WDEDUP_EINVAL;

  const ie_tensor_ref_t *tr = ie_find_tensor(h, name);
  if (!tr) return IE_WDEDUP_ENOENT;

  /* validate blob ranges */
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

  /* If there are no exceptions and no mask, allow this to be treated as direct. */
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

/* -------------------------- Materialization helpers ------------------------- */

/**
 * @brief Read a bit from a bitset.
 *
 * @param[in] bits Bitset bytes.
 * @param[in] i    Bit index.
 * @return 0 or 1.
 */
static inline int ie_bitset_get(const uint8_t *bits, size_t i) {
  return (bits[i >> 3] >> (unsigned)(i & 7u)) & 1u;
}

/**
 * @brief Get an int4 nibble from a packed byte array (lo/hi nibbles).
 *
 * Nibble order:
 *  - index even  => low nibble
 *  - index odd   => high nibble
 *
 * @param[in] p Packed int4 byte array.
 * @param[in] idx Nibble index.
 * @return Nibble value in [0, 15].
 */
static inline uint8_t ie_int4_get_nibble(const uint8_t *p, size_t idx) {
  uint8_t b = p[idx >> 1];
  if ((idx & 1u) == 0) return (uint8_t)(b & 0x0F);
  return (uint8_t)((b >> 4) & 0x0F);
}

/**
 * @brief Set an int4 nibble into a packed byte array (lo/hi nibbles).
 *
 * @param[in,out] p Packed int4 byte array.
 * @param[in] idx Nibble index.
 * @param[in] v Nibble value (low 4 bits are used).
 */
static inline void ie_int4_set_nibble(uint8_t *p, size_t idx, uint8_t v) {
  uint8_t *b = &p[idx >> 1];
  v &= 0x0F;
  if ((idx & 1u) == 0) {
    *b = (uint8_t)((*b & 0xF0) | v);
  } else {
    *b = (uint8_t)((*b & 0x0F) | (uint8_t)(v << 4));
  }
}

/**
 * @brief Materialize a weight view into a contiguous destination buffer.
 *
 * Behavior:
 * - If @p view is IE_WVIEW_DIRECT, this copies the referenced bytes verbatim.
 * - If @p view is IE_WVIEW_DEDUP:
 *   - For non-int4 dtypes: copy defaults, then apply per-element overrides from
 *     the exceptions stream where the mask bit is set.
 *   - For int4: destination is a packed nibble array; defaults are copied, then
 *     overridden nibble-by-nibble from the exceptions stream.
 *
 * The function returns 0 on any validation/size error.
 *
 * @param[in]  view Parsed weight view.
 * @param[out] dst Destination buffer.
 * @param[in]  dst_nbytes Capacity of @p dst in bytes.
 * @return Number of bytes written to @p dst, or 0 on failure.
 */
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

  /* fp/int8 path: copy defaults then override elementwise */
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

    /* It is valid for exceptions_nbytes to have padding; we only require not to underflow. */
    return need;
  }

  /* int4 path: dst is packed int4 bytes. */
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
