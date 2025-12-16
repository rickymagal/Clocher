/* ============================================================================
 * File: engine/src/io/weights.c
 * ============================================================================
 */
/**
 * @file weights.c
 * @brief Implementation of the IEBIN v1 loader with relaxed JSON parsing.
 *
 * ## Design Goals
 * - Zero third-party dependencies (plain C, no JSON libraries).
 * - Resilient scanning (tolerate extra whitespace/fields).
 * - Safety with -Werror (no unsafe formatting, careful bounds checking).
 * - Clarity (straight-line code with explicit pre/post-conditions).
 *
 * This module extracts minimal header info from `model.ie.json` and resolves
 * the path to `model.ie.bin`. It also exposes a small @ref ie_weights_touch()
 * routine to validate OS-level readability and optionally warm caches.
 *
 * ### Deduplicated weights support (lossless)
 * If a sibling `model.dedup.json` exists in the same directory as `model.ie.json`,
 * this loader will prefer it and treat the deduplicated artifacts as the active
 * weights source. This allows drop-in replacement of the model directory without
 * changing engine call sites.
 *
 * The dedup loader itself is implemented in:
 *   - `engine/include/io/weights_dedup.h`
 *   - `engine/src/io/weights_dedup.c`
 *
 * This file only decides which representation is available and records enough
 * information for downstream code to route tensor reads accordingly.
 *
 * ### INT4 weight-only helpers (exploratory)
 * Additional helpers are provided to discover and decode INT4 weight-only
 * metadata from the JSON header and to dequantize packed INT4 blobs into
 * float32 matrices. These helpers are not part of the public `ie_io.h` API
 * yet; they are provided here to enable incremental integration without
 * changing headers:
 *   - ::ie_weights_read_int4_meta
 *   - ::ie_weights_decode_int4
 *
 * The decoding helpers depend on `ie_quant_int4.h` for packing rules and
 * dequantization routines.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_io.h"            /* ie_weights_t, IE_IO_* status codes */
#include "ie_quant_int4.h"    /* INT4 quant packing/dequant helpers */
#include "weights_dedup.h" /* dedup loader (mmap-friendly) */

#include <ctype.h>    /* tolower */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>   /* uint16_t, uint32_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* -------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @internal
 * @brief Copy C string @p src into @p dst (NUL-terminated), truncating if needed.
 * Ensures @p dst is always NUL-terminated if @p dstsz > 0.
 *
 * @param dst   Destination buffer.
 * @param dstsz Destination buffer size in bytes.
 * @param src   Source C string (may be NULL).
 */
static void cpyz(char *dst, size_t dstsz, const char *src) {
  if (!dst || dstsz == 0) return;
  if (!src) { dst[0] = '\0'; return; }
  size_t n = strlen(src);
  if (n >= dstsz) n = dstsz - 1;
  memcpy(dst, src, n);
  dst[n] = '\0';
}

/**
 * @internal
 * @brief Return pointer to the last slash '/' in @p p (or NULL if none).
 *
 * @param p Input path string.
 * @return Pointer to last '/' inside @p p, or NULL.
 */
static const char *last_slash(const char *p) {
  const char *s = NULL;
  if (!p) return NULL;
  for (const char *q = p; *q; ++q) if (*q == '/') s = q;
  return s;
}

/**
 * @internal
 * @brief Safe path join: `out = dir [/] file` with truncation and no `snprintf`.
 *
 * @param out   Output buffer.
 * @param outsz Output buffer size in bytes.
 * @param dir   Directory path (may be NULL/empty).
 * @param file  File name (must not be NULL when used).
 */
static void join_path(char *out, size_t outsz, const char *dir, const char *file) {
  if (!out || outsz == 0) return;
  out[0] = '\0';
  if (!file || !*file) return;

  if (!dir || !*dir) { cpyz(out, outsz, file); return; }

  const size_t ld = strlen(dir);
  const size_t lf = strlen(file);
  const int need_slash = (ld > 0 && dir[ld - 1] != '/');

  size_t pos = 0;

  size_t copyd = ld;
  if (copyd >= outsz) copyd = outsz - 1;
  if (copyd > 0) { memcpy(out + pos, dir, copyd); pos += copyd; }

  if (pos < outsz - 1 && need_slash) { out[pos++] = '/'; }

  if (pos < outsz - 1) {
    size_t room = (outsz - 1) - pos;
    size_t copyf = (lf > room) ? room : lf;
    if (copyf > 0) { memcpy(out + pos, file, copyf); pos += copyf; }
  }

  out[pos] = '\0';
}

/**
 * @internal
 * @brief Read entire text file into memory (NUL-terminated). Strips UTF-8 BOM.
 *
 * @param path     Path to the text file.
 * @param buf      Output pointer to allocated buffer with contents (+NUL).
 * @param len_out  Optional output length (excluding NUL).
 * @return 0 on success; negative on error.
 */
static int read_all_text(const char *path, char **buf, size_t *len_out) {
  if (!path || !buf) return -1;
  *buf = NULL; if (len_out) *len_out = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -2;

  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -3; }
  long n = ftell(f);
  if (n < 0) { fclose(f); return -3; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -3; }

  size_t size = (size_t)n;
  char *mem = (char*)malloc(size + 1);
  if (!mem) { fclose(f); return -4; }

  size_t rd = fread(mem, 1, size, f);
  fclose(f);
  if (rd != size) { free(mem); return -5; }

  /* Strip UTF-8 BOM if present */
  if (size >= 3 &&
      (unsigned char)mem[0] == 0xEF &&
      (unsigned char)mem[1] == 0xBB &&
      (unsigned char)mem[2] == 0xBF) {
    size -= 3;
    memmove(mem, mem + 3, size);
  }

  mem[size] = '\0';
  *buf = mem;
  if (len_out) *len_out = size;
  return 0;
}

/**
 * @internal
 * @brief Find integer value of `"key": <int>` in JSON text via relaxed scan.
 *
 * @param json    JSON buffer (NUL-terminated).
 * @param key     Key name without quotes.
 * @param out_val Output integer.
 * @return 1 if found, 0 if not found, negative on bad args.
 */
static int scan_json_key_int(const char *json, const char *key, int *out_val) {
  if (!json || !key || !out_val) return -1;
  const char *p = json;
  const size_t klen = strlen(key);
  while ((p = strstr(p, "\""))) {
    ++p;
    if (strncmp(p, key, klen) == 0 && p[klen] == '"') {
      const char *c = p + klen + 1;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != ':') continue;
      ++c;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      char *end = NULL;
      long v = strtol(c, &end, 10);
      if (end && end != c) { *out_val = (int)v; return 1; }
    }
  }
  return 0;
}

/**
 * @internal
 * @brief Find string value of `"key": "<value>"` via relaxed scan.
 *
 * @param json  JSON buffer (NUL-terminated).
 * @param key   Key name without quotes.
 * @param dst   Output buffer for value.
 * @param dstsz Output buffer size.
 * @return 1 if found, 0 if not found, negative on bad args.
 */
static int scan_json_key_string(const char *json, const char *key, char *dst, size_t dstsz) {
  if (!json || !key || !dst || dstsz == 0) return -1;
  dst[0] = '\0';
  const char *p = json;
  const size_t klen = strlen(key);
  while ((p = strstr(p, "\""))) {
    ++p;
    if (strncmp(p, key, klen) == 0 && p[klen] == '"') {
      const char *c = p + klen + 1;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != ':') continue;
      ++c;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != '"') continue;
      ++c;
      const char *start = c;
      while (*c && *c != '"') ++c;
      size_t n = (size_t)(c - start);
      if (n >= dstsz) n = dstsz - 1;
      memcpy(dst, start, n);
      dst[n] = '\0';
      return 1;
    }
  }
  return 0;
}

/**
 * @internal
 * @brief Locate the JSON object section for a given key (e.g., `"quant": { ... }`).
 * Returns pointers to the body range [`*out_begin`, `*out_end`).
 *
 * @param json      JSON buffer (NUL-terminated).
 * @param key       Object key to search (without quotes).
 * @param out_begin Output: pointer to first byte after '{'.
 * @param out_end   Output: pointer to the matching '}'.
 * @return 1 if found, 0 if not found, negative on error.
 */
static int scan_json_object_range(const char *json,
                                  const char *key,
                                  const char **out_begin,
                                  const char **out_end) {
  if (!json || !key || !out_begin || !out_end) return -1;
  *out_begin = *out_end = NULL;

  const size_t klen = strlen(key);
  const char *p = json;
  while ((p = strstr(p, "\""))) {
    ++p;
    if (strncmp(p, key, klen) == 0 && p[klen] == '"') {
      const char *c = p + klen + 1;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != ':') continue;
      ++c;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != '{') return 0;
      /* scan balanced braces */
      int depth = 0;
      const char *start = c + 1;
      const char *q = c;
      do {
        if (*q == '{') ++depth;
        else if (*q == '}') --depth;
        ++q;
      } while (*q && depth > 0);
      if (depth == 0) {
        *out_begin = start;
        *out_end   = q - 1; /* position of '}' */
        return 1;
      }
      return 0;
    }
  }
  return 0;
}

/**
 * @internal
 * @brief Find string value inside a known object range: `"key": "<value>"`.
 *
 * @param begin Start pointer of object body (first byte after '{').
 * @param end   End pointer (position of matching '}').
 * @param key   Key to search (without quotes).
 * @param dst   Output buffer for the string value.
 * @param dstsz Output buffer size.
 * @return 1 if found, 0 if not found, negative on bad args.
 */
static int scan_json_key_string_in_range(const char *begin,
                                         const char *end,
                                         const char *key,
                                         char *dst,
                                         size_t dstsz) {
  if (!begin || !end || !key || !dst || dstsz == 0) return -1;
  dst[0] = '\0';
  const size_t klen = strlen(key);
  const char *p = begin;
  while (p < end) {
    const char *q = memchr(p, '"', (size_t)(end - p));
    if (!q) break;
    ++q;
    if ((size_t)(end - q) >= klen && memcmp(q, key, klen) == 0 && q[klen] == '"') {
      const char *c = q + klen + 1;
      while (c < end && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n')) ++c;
      if (c >= end || *c != ':') { p = q + 1; continue; }
      ++c;
      while (c < end && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n')) ++c;
      if (c >= end || *c != '"') { p = q + 1; continue; }
      ++c;
      const char *start = c;
      while (c < end && *c != '"') ++c;
      size_t n = (size_t)(c - start);
      if (n >= dstsz) n = dstsz - 1;
      memcpy(dst, start, n);
      dst[n] = '\0';
      return 1;
    }
    p = q + 1;
  }
  return 0;
}

/**
 * @internal
 * @brief Portable positional read helper. Uses pread(2) when available.
 *
 * @param fd     File descriptor.
 * @param buf    Output buffer.
 * @param count  Number of bytes to read.
 * @param offset Offset from file start.
 * @return Bytes read on success, -1 on error.
 */
static ssize_t ie_pread(int fd, void *buf, size_t count, off_t offset) {
#if defined(_XOPEN_SOURCE) || defined(_POSIX_C_SOURCE)
  return pread(fd, buf, count, offset);
#else
  off_t cur = lseek(fd, 0, SEEK_CUR);
  if (cur == (off_t)-1) return -1;
  if (lseek(fd, offset, SEEK_SET) == (off_t)-1) return -1;
  ssize_t r = read(fd, buf, count);
  (void)lseek(fd, cur, SEEK_SET);
  return r;
#endif
}

/**
 * @internal
 * @brief Case-insensitive ASCII equality (NULL-safe).
 *
 * @param a First string.
 * @param b Second string.
 * @return 1 if equal (case-insensitive), 0 otherwise.
 */
static int ascii_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    if ((unsigned char)tolower(*a) != (unsigned char)tolower(*b)) return 0;
    ++a; ++b;
  }
  return *a == '\0' && *b == '\0';
}

/**
 * @internal
 * @brief Check whether a file exists and is readable.
 *
 * @param path Path to test.
 * @return 1 if accessible for reading, 0 otherwise.
 */
static int file_exists_readable(const char *path) {
  if (!path || !*path) return 0;
  return access(path, R_OK) == 0;
}


/**
 * @internal
 * @brief Parse a boolean environment flag.
 *
 * Accepts the following case-insensitive values as true: "1", "true", "yes", "on".
 * Accepts the following case-insensitive values as false: "0", "false", "no", "off".
 * Any other non-empty value is treated as true (conservative).
 *
 * @param name Environment variable name.
 * @param default_value Value returned when the variable is not set.
 * @return 1 for true, 0 for false.
 */
static int env_flag_get(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;
  if (ascii_ieq(v, "0") || ascii_ieq(v, "false") || ascii_ieq(v, "no") || ascii_ieq(v, "off")) return 0;
  if (ascii_ieq(v, "1") || ascii_ieq(v, "true") || ascii_ieq(v, "yes") || ascii_ieq(v, "on")) return 1;
  return 1;
}

/* -------------------------------------------------------------------------- */
/* Public IEBIN v1 loader                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @copydoc ie_weights_open
 *
 * Behavior:
 * - Loads `model.ie.json` exactly as before.
 * - Additionally checks whether a sibling `model.dedup.json` exists in the same
 *   directory. If present, it opens the dedup loader and records that this
 *   weights handle is backed by deduplicated artifacts.
 *
 * Notes:
 * - This function does not change the public ABI in `ie_io.h`. It relies on
 *   existing spare/opaque fields in `ie_weights_t` to store the dedup handle.
 * - If your `ie_weights_t` does not have such a field yet, add:
 *     `void *dedup_handle; int is_dedup;`
 *   to `ie_io.h` (or a private extension struct) and recompile.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out) {
  if (!out) return IE_IO_ERR_ARGS;
  memset(out, 0, sizeof(*out));

  if (!json_path || !*json_path) return IE_IO_ERR_ARGS;
  cpyz(out->json_path, sizeof(out->json_path), json_path);

  /* Read JSON text */
  char *jbuf = NULL; size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    return IE_IO_ERR_JSON;
  }

  /* Parse relaxed header keys */
  int ver = 0; (void)scan_json_key_int(jbuf, "version", &ver);
  if (ver <= 0) ver = 1;
  out->version = ver;

  char dtype[16]; dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  if (dtype[0] == '\0') strcpy(dtype, "float32");
  cpyz(out->dtype, sizeof(out->dtype), dtype);

  /* Determine weights path (absolute or relative to JSON dir) */
  char resolved_bin[512]; resolved_bin[0] = '\0';
  if (bin_path && *bin_path) {
    cpyz(resolved_bin, sizeof(resolved_bin), bin_path);
  } else {
    char bin_key[256]; bin_key[0] = '\0';
    (void)scan_json_key_string(jbuf, "bin", bin_key, sizeof(bin_key));
    if (bin_key[0] == '\0') { free(jbuf); return IE_IO_ERR_BIN_UNSPEC; }

    const char *slash = last_slash(json_path);
    if (slash) {
      char dir[512]; size_t dn = (size_t)(slash - json_path);
      if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
      memcpy(dir, json_path, dn); dir[dn] = '\0';
      join_path(resolved_bin, sizeof(resolved_bin), dir, bin_key);
    } else {
      cpyz(resolved_bin, sizeof(resolved_bin), bin_key);
    }
  }
  cpyz(out->weights_path, sizeof(out->weights_path), resolved_bin);

  /* Stat the weights file to get its size */
  struct stat st;
  if (stat(out->weights_path, &st) != 0 || st.st_size < 0) {
    free(jbuf);
    return IE_IO_ERR_STAT;
  }
  out->bin_size_bytes = (size_t)st.st_size;
  out->loaded = 1;

  /* Dedup branch: optionally open model.dedup.json if present next to model.ie.json. */
  {
    /* Default policy:
     *  - If IE_DEDUP is unset: auto-enable when model.dedup.json is present.
     *  - If IE_DEDUP=0: never enable dedup (even if artifacts are present).
     *  - If IE_DEDUP=1: attempt dedup; on failure, either fall back (default) or hard-fail
     *    when IE_DEDUP_STRICT=1.
     */
    const int dedup_enabled = env_flag_get("IE_DEDUP", 1);
    const int dedup_strict  = env_flag_get("IE_DEDUP_STRICT", 0);

    if (dedup_enabled) {
      const char *slash = last_slash(json_path);
      char dir[512]; dir[0] = '\0';
      if (slash) {
        size_t dn = (size_t)(slash - json_path);
        if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
        memcpy(dir, json_path, dn);
        dir[dn] = '\0';
      } else {
        /* If json_path has no slash, treat current working directory as the dir. */
        cpyz(dir, sizeof(dir), ".");
      }

      char dedup_json[512]; dedup_json[0] = '\0';
      join_path(dedup_json, sizeof(dedup_json), dir, "model.dedup.json");

      if (file_exists_readable(dedup_json)) {
        ie_weights_dedup_opts_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.prefault_policy = 0;

        ie_weights_dedup_t *dh = NULL;
        ie_wdedup_status_t dst = ie_weights_dedup_open(&dh, dir, &opts);
        if (dst == IE_WDEDUP_OK && dh) {
          /* Store into the weights handle (requires fields to exist in ie_weights_t). */
          out->is_dedup = 1;
          out->dedup_handle = (void *)dh;
        } else {
          /* Do not break baseline loading if dedup artifacts are incomplete.
           * Strict mode exists for CI or debugging.
           */
          if (dedup_strict) {
            free(jbuf);
            return IE_IO_ERR_JSON;
          }
          /* Best-effort fallback */
          out->is_dedup = 0;
          out->dedup_handle = NULL;
        }
      }
    }
  }

  free(jbuf);
  return IE_IO_OK;
}

/**
 * @copydoc ie_weights_touch
 *
 * For deduplicated weights, this function touches the primary backing file
 * (the regular `model.ie.bin`) exactly as before. The dedup path warm-up is
 * expected to be handled by the dedup loader and/or the mmap layer.
 */
int ie_weights_touch(const ie_weights_t *w) {
  if (!w || !w->weights_path[0]) return IE_IO_ERR_ARGS;
  int fd = open(w->weights_path, O_RDONLY);
  if (fd < 0) return IE_IO_ERR_OPEN;

  unsigned char buf[4096];
  ssize_t r1 = ie_pread(fd, buf, sizeof(buf), 0);
  ssize_t r2 = 0;
  if (w->bin_size_bytes > sizeof(buf)) {
    off_t off = (off_t)(w->bin_size_bytes - sizeof(buf));
    r2 = ie_pread(fd, buf, sizeof(buf), off);
  }
  close(fd);

  if (r1 < 0 || r2 < 0) return IE_IO_ERR_READ;
  return IE_IO_OK;
}

/**
 * @copydoc ie_weights_close
 *
 * This now closes the dedup handle if one was opened.
 */
void ie_weights_close(ie_weights_t *w) {
  if (!w) return;
  if (w->is_dedup && w->dedup_handle) {
    ie_weights_dedup_t *dh = (ie_weights_dedup_t *)w->dedup_handle;
    ie_weights_dedup_close(&dh);
    w->dedup_handle = NULL;
    w->is_dedup = 0;
  }
}

/* -------------------------------------------------------------------------- */
/* INT4 weight-only helpers (exploratory; not yet in public header)           */
/* -------------------------------------------------------------------------- */

/**
 * @struct ie_int4_quant_meta
 * @brief Minimal metadata extracted from `"quant": { ... }` for INT4 weights.
 *
 * This struct is defined here for incremental integration only. When promoting
 * to the public API, move the definition to a dedicated header.
 */
struct ie_int4_quant_meta {
  int   present;                       /**< 1 if INT4 metadata found. */
  int   per_row;                       /**< 1 for per-row, 0 for per-tensor. */
  char  scale_bin[512];                /**< Path to scales binary. */
  char  pack[32];                      /**< Expected "nibble_lohi". */
  int   zero_point;                    /**< Expected 0. */
  int   symmetric;                     /**< Expected 1 (true). */
};

/**
 * @brief Read INT4 quantization metadata from an IEBIN v1 JSON header.
 *
 * The function scans a relaxed JSON header, locates the "quant" object and
 * extracts fields relevant to INT4 decoding. It resolves @c scale_bin relative
 * to @p json_path if needed.
 *
 * @param json_path  Path to `model.ie.json`.
 * @param out_meta   Output metadata (must be non-NULL). On success,
 *                   `out_meta->present` is 1 when INT4 metadata is found.
 * @return ::IE_IO_OK on success (even if no INT4 meta is present),
 *         or an ::IE_IO_ERR_* code on I/O/parse errors.
 */
int ie_weights_read_int4_meta(const char *json_path, struct ie_int4_quant_meta *out_meta) {
  if (!json_path || !out_meta) return IE_IO_ERR_ARGS;
  memset(out_meta, 0, sizeof(*out_meta));

  /* Read JSON */
  char *jbuf = NULL; size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    return IE_IO_ERR_JSON;
  }

  /* dtype must indicate INT4/Q4 (root-level) to proceed */
  char dtype[16]; dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  if (!(ascii_ieq(dtype, "int4") || ascii_ieq(dtype, "q4") || ascii_ieq(dtype, "mixed"))) {
    free(jbuf);
    out_meta->present = 0;
    return IE_IO_OK;
  }

  /* Find "quant": { ... } section */
  const char *qb = NULL, *qe = NULL;
  if (scan_json_object_range(jbuf, "quant", &qb, &qe) != 1 || !qb || !qe || qb >= qe) {
    free(jbuf);
    out_meta->present = 0;
    return IE_IO_OK; /* treat as missing metadata */
  }

  /* Extract fields from the quant object */
  char per[32]; per[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "per", per, sizeof(per));
  out_meta->per_row = ascii_ieq(per, "row") ? 1 : 0;

  char scale_bin_rel[512]; scale_bin_rel[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "scale_bin", scale_bin_rel, sizeof(scale_bin_rel));

  char pack[32]; pack[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "pack", pack, sizeof(pack));
  cpyz(out_meta->pack, sizeof(out_meta->pack), pack);

  char zp_str[32]; zp_str[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "zp", zp_str, sizeof(zp_str));
  if (zp_str[0]) out_meta->zero_point = atoi(zp_str);

  char sym_str[32]; sym_str[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "symmetric", sym_str, sizeof(sym_str));
  if (sym_str[0]) out_meta->symmetric = (ascii_ieq(sym_str, "true") || strcmp(sym_str, "1") == 0) ? 1 : 0;

  /* Resolve scale_bin relative to JSON directory if needed */
  if (scale_bin_rel[0]) {
    const char *slash = last_slash(json_path);
    if (slash) {
      char dir[512]; size_t dn = (size_t)(slash - json_path);
      if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
      memcpy(dir, json_path, dn); dir[dn] = '\0';
      join_path(out_meta->scale_bin, sizeof(out_meta->scale_bin), dir, scale_bin_rel);
    } else {
      cpyz(out_meta->scale_bin, sizeof(out_meta->scale_bin), scale_bin_rel);
    }
  } else {
    out_meta->scale_bin[0] = '\0';
  }

  out_meta->present = 1;
  free(jbuf);
  return IE_IO_OK;
}

/**
 * @internal
 * @brief Convert IEEE754 half-precision (fp16) to single-precision float (fp32).
 *
 * This conversion handles zeros, subnormals, normals, infinities, and NaNs.
 *
 * @param h 16-bit half-precision bit pattern.
 * @return Converted 32-bit float value.
 */
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;  /* zero */
    } else {
      /* subnormal: normalize mantissa */
      int e = -1;
      do { e++; mant <<= 1; } while ((mant & 0x0400u) == 0);
      mant &= 0x03FFu;
      f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13); /* Inf/NaN */
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

/**
 * @brief Dequantize a packed INT4 weight blob into a float32 matrix.
 *
 * Reads the packed INT4 blob and the associated scales from disk and produces a
 * row-major float32 matrix into @p dst (caller-allocated).
 *
 * The function auto-detects scale dtype between fp16 and fp32 by comparing the
 * file size with the expected number of elements.
 *
 * @param packed_path  Path to packed INT4 weights (`*.int4.bin`).
 * @param scales_path  Path to scales file. For per-tensor it must contain
 *                     a single float (fp16 or fp32); for per-row it must
 *                     contain `rows` floats.
 * @param rows         Number of matrix rows.
 * @param cols         Number of matrix columns.
 * @param per_row      Non-zero for per-row scales; 0 for per-tensor.
 * @param dst          Output buffer (length = rows*cols floats).
 * @return ::IE_IO_OK on success, or an ::IE_IO_ERR_* code on failure.
 */
int ie_weights_decode_int4(const char *packed_path,
                           const char *scales_path,
                           size_t rows,
                           size_t cols,
                           int per_row,
                           float *dst) {
  if (!packed_path || !scales_path || !dst || rows == 0 || cols == 0) return IE_IO_ERR_ARGS;

  /* Read packed blob */
  int fdw = open(packed_path, O_RDONLY);
  if (fdw < 0) return IE_IO_ERR_OPEN;

  struct stat stw;
  if (fstat(fdw, &stw) != 0) { close(fdw); return IE_IO_ERR_STAT; }

  const size_t need_packed = rows * ie_int4_rowbytes(cols);
  if ((size_t)stw.st_size < need_packed) { close(fdw); return IE_IO_ERR_READ; }

  uint8_t *buf_packed = (uint8_t*)malloc(need_packed);
  if (!buf_packed) { close(fdw); return IE_IO_ERR_ALLOC; }

  ssize_t rd = ie_pread(fdw, buf_packed, need_packed, 0);
  close(fdw);
  if (rd < 0 || (size_t)rd != need_packed) { free(buf_packed); return IE_IO_ERR_READ; }

  /* Read scales (auto-detect fp16 or fp32) */
  int fds = open(scales_path, O_RDONLY);
  if (fds < 0) { free(buf_packed); return IE_IO_ERR_OPEN; }

  struct stat sts;
  if (fstat(fds, &sts) != 0) { close(fds); free(buf_packed); return IE_IO_ERR_STAT; }

  size_t nsc = per_row ? rows : 1;
  size_t need16 = nsc * sizeof(uint16_t);
  size_t need32 = nsc * sizeof(float);

  int is16 = ((size_t)sts.st_size == need16);
  int is32 = ((size_t)sts.st_size == need32);
  if (!(is16 || is32)) { close(fds); free(buf_packed); return IE_IO_ERR_READ; }

  float *buf_scales = (float*)malloc(need32);
  if (!buf_scales) { close(fds); free(buf_packed); return IE_IO_ERR_ALLOC; }

  if (is32) {
    ssize_t rds = ie_pread(fds, buf_scales, need32, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need32) { free(buf_scales); free(buf_packed); return IE_IO_ERR_READ; }
  } else {
    uint16_t *tmp = (uint16_t*)malloc(need16);
    if (!tmp) { close(fds); free(buf_scales); free(buf_packed); return IE_IO_ERR_ALLOC; }
    ssize_t rds = ie_pread(fds, tmp, need16, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need16) { free(tmp); free(buf_scales); free(buf_packed); return IE_IO_ERR_READ; }
    for (size_t i = 0; i < nsc; ++i) buf_scales[i] = fp16_to_fp32(tmp[i]);
    free(tmp);
  }

  /* Dequantize */
  int qst = 0;
  if (per_row) {
    qst = ie_int4_dequantize_per_row(buf_packed, rows, cols, buf_scales, dst);
  } else {
    qst = ie_int4_dequantize_per_tensor(buf_packed, rows, cols, buf_scales[0], dst);
  }

  free(buf_scales);
  free(buf_packed);

  if (qst != IE_INT4_STATUS_OK) return IE_IO_ERR_DECODE;
  return IE_IO_OK;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */

