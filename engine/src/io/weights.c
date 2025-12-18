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
 * this loader can prefer it and treat the deduplicated artifacts as the active
 * weights source.
 *
 * IMPORTANT:
 * - This file does NOT assume your ie_weights_t has dedup fields.
 * - If your ie_weights_t DOES have them, define IE_WEIGHTS_HAS_DEDUP_FIELDS=1
 *   in your build (or in ie_io.h) and provide:
 *     - int  is_dedup;
 *     - void *dedup_handle;
 *
 * If those fields do not exist, dedup is simply not activated here (baseline
 * path remains unchanged and the TU still compiles cleanly under -Werror).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_io.h"         /* ie_weights_t, IE_IO_* status codes */
#include "ie_quant_int4.h" /* INT4 quant packing/dequant helpers */
#include "weights_dedup.h" /* NOTE: include path has NO subfolders */

#include <ctype.h> /* tolower */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h> /* uint16_t, uint32_t */
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
 * @brief Copy string into bounded buffer and always NUL-terminate.
 *
 * @param dst   Destination buffer (may be NULL).
 * @param dstsz Capacity in bytes.
 * @param src   Source string (may be NULL).
 */
static void cpyz(char *dst, size_t dstsz, const char *src) {
  if (!dst || dstsz == 0) return;
  if (!src) {
    dst[0] = '\0';
    return;
  }
  size_t n = strlen(src);
  if (n >= dstsz) n = dstsz - 1;
  memcpy(dst, src, n);
  dst[n] = '\0';
}

/**
 * @brief Return pointer to the last '/' in a path, or NULL if none.
 *
 * @param p NUL-terminated path.
 * @return Pointer to last '/', or NULL.
 */
static const char *last_slash(const char *p) {
  const char *s = NULL;
  if (!p) return NULL;
  for (const char *q = p; *q; ++q) {
    if (*q == '/') s = q;
  }
  return s;
}

/**
 * @brief Join directory and filename into an output buffer.
 *
 * If @p dir is empty, the result is just @p file. Ensures NUL termination.
 *
 * @param out   Output buffer.
 * @param outsz Output capacity.
 * @param dir   Directory path (may be NULL/empty).
 * @param file  Filename (must be non-empty).
 */
static void join_path(char *out, size_t outsz, const char *dir, const char *file) {
  if (!out || outsz == 0) return;
  out[0] = '\0';
  if (!file || !*file) return;

  if (!dir || !*dir) {
    cpyz(out, outsz, file);
    return;
  }

  const size_t ld = strlen(dir);
  const size_t lf = strlen(file);
  const int need_slash = (ld > 0 && dir[ld - 1] != '/');

  size_t pos = 0;

  size_t copyd = ld;
  if (copyd >= outsz) copyd = outsz - 1;
  if (copyd > 0) {
    memcpy(out + pos, dir, copyd);
    pos += copyd;
  }

  if (pos < outsz - 1 && need_slash) out[pos++] = '/';

  if (pos < outsz - 1) {
    size_t room = (outsz - 1) - pos;
    size_t copyf = (lf > room) ? room : lf;
    if (copyf > 0) {
      memcpy(out + pos, file, copyf);
      pos += copyf;
    }
  }

  out[pos] = '\0';
}

/**
 * @brief Read an entire file into a newly allocated NUL-terminated buffer.
 *
 * Also strips a UTF-8 BOM if present.
 *
 * @param path    File path.
 * @param buf     Output pointer to malloc'd buffer (caller frees).
 * @param len_out Optional output length (bytes, excluding NUL).
 * @return 0 on success, negative on failure.
 */
static int read_all_text(const char *path, char **buf, size_t *len_out) {
  if (!path || !buf) return -1;
  *buf = NULL;
  if (len_out) *len_out = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return -2;

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return -3;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return -3;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return -3;
  }

  size_t size = (size_t)n;
  char *mem = (char *)malloc(size + 1);
  if (!mem) {
    fclose(f);
    return -4;
  }

  size_t rd = fread(mem, 1, size, f);
  fclose(f);
  if (rd != size) {
    free(mem);
    return -5;
  }

  if (size >= 3 && (unsigned char)mem[0] == 0xEF && (unsigned char)mem[1] == 0xBB &&
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
 * @brief Scan a relaxed JSON blob for an integer value at key "key".
 *
 * This is a simple best-effort scanner. It does not implement full JSON.
 *
 * @param json     NUL-terminated text.
 * @param key      Key name without quotes (e.g., "version").
 * @param out_val  Output integer.
 * @return 1 if found, 0 if not found, negative on invalid args.
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
      if (end && end != c) {
        *out_val = (int)v;
        return 1;
      }
    }
  }
  return 0;
}

/**
 * @brief Scan a relaxed JSON blob for a string value at key "key".
 *
 * Copies up to dstsz-1 bytes and NUL-terminates.
 *
 * @param json   NUL-terminated JSON-ish text.
 * @param key    Key name without quotes.
 * @param dst    Output buffer.
 * @param dstsz  Output capacity.
 * @return 1 if found, 0 if not found, negative on invalid args.
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
 * @brief Locate the byte range of a JSON object value "{ ... }" for a key.
 *
 * Returns pointers to the interior (after '{') and the closing brace position
 * (pointer to the '}' character). The range is [begin, end).
 *
 * @param json       JSON-ish text.
 * @param key        Object key to find.
 * @param out_begin  Output begin pointer (inside the braces).
 * @param out_end    Output end pointer (points to closing brace).
 * @return 1 if found, 0 if not found/malformed, negative on invalid args.
 */
static int scan_json_object_range(const char *json, const char *key, const char **out_begin,
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
      int depth = 0;
      const char *start = c + 1;
      const char *q = c;
      do {
        if (*q == '{')
          ++depth;
        else if (*q == '}')
          --depth;
        ++q;
      } while (*q && depth > 0);
      if (depth == 0) {
        *out_begin = start;
        *out_end = q - 1;
        return 1;
      }
      return 0;
    }
  }
  return 0;
}

/**
 * @brief Scan for a string key within a bounded [begin,end) range.
 *
 * This is used to parse nested objects without allocating substrings.
 *
 * @param begin  Range begin.
 * @param end    Range end (must be >= begin).
 * @param key    Key name.
 * @param dst    Output buffer.
 * @param dstsz  Output capacity.
 * @return 1 if found, 0 if not found, negative on invalid args.
 */
static int scan_json_key_string_in_range(const char *begin, const char *end, const char *key,
                                         char *dst, size_t dstsz) {
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
      if (c >= end || *c != ':') {
        p = q + 1;
        continue;
      }
      ++c;
      while (c < end && (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n')) ++c;
      if (c >= end || *c != '"') {
        p = q + 1;
        continue;
      }
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
 * @brief Portable pread wrapper.
 *
 * Uses pread() when available; otherwise emulates with lseek+read and restores
 * the file offset.
 *
 * @param fd      Open file descriptor.
 * @param buf     Destination buffer.
 * @param count   Number of bytes to read.
 * @param offset  Offset from start.
 * @return Bytes read, or -1 on error.
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
 * @brief ASCII-only case-insensitive equality.
 *
 * @param a NUL-terminated.
 * @param b NUL-terminated.
 * @return 1 if equal (case-insensitive ASCII), else 0.
 */
static int ascii_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    if ((unsigned char)tolower((unsigned char)*a) != (unsigned char)tolower((unsigned char)*b)) {
      return 0;
    }
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

#if defined(IE_WEIGHTS_HAS_DEDUP_FIELDS) && (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
/**
 * @brief Check if a path exists and is readable.
 *
 * @param path NUL-terminated path.
 * @return 1 if readable, else 0.
 */
static int file_exists_readable(const char *path) {
  if (!path || !*path) return 0;
  return access(path, R_OK) == 0;
}

/**
 * @brief Parse environment flag-like strings into a boolean.
 *
 * Accepts: 0/1, false/true, no/yes, off/on (ASCII case-insensitive).
 * Unknown values default to 1 (enabled) to avoid silent disable.
 *
 * @param name          Environment variable name.
 * @param default_value Default value if not set/empty.
 * @return 0 or 1.
 */
static int env_flag_get(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;
  if (ascii_ieq(v, "0") || ascii_ieq(v, "false") || ascii_ieq(v, "no") || ascii_ieq(v, "off"))
    return 0;
  if (ascii_ieq(v, "1") || ascii_ieq(v, "true") || ascii_ieq(v, "yes") || ascii_ieq(v, "on"))
    return 1;
  return 1;
}

/**
 * @brief Create a best-effort symlink model.dedup.json -> dedup_manifest.json.
 *
 * Some pipelines may output `dedup_manifest.json`. This helper creates a
 * sibling symlink named `model.dedup.json` if it is missing. Failures are
 * non-fatal; the function simply returns whether the expected link is readable.
 *
 * @param dir Directory containing the artifacts.
 * @return 1 if model.dedup.json ends up readable, else 0.
 */
static int ensure_dedup_manifest_link(const char *dir) {
  if (!dir || !*dir) return 0;

  char target[512];
  char linkpath[512];
  char manifest[512];

  target[0] = linkpath[0] = manifest[0] = '\0';

  join_path(linkpath, sizeof(linkpath), dir, "model.dedup.json");
  if (file_exists_readable(linkpath)) return 1;

  join_path(manifest, sizeof(manifest), dir, "dedup_manifest.json");
  if (!file_exists_readable(manifest)) return 0;

  cpyz(target, sizeof(target), "dedup_manifest.json");
  (void)symlink(target, linkpath);

  return file_exists_readable(linkpath) ? 1 : 0;
}
#endif /* IE_WEIGHTS_HAS_DEDUP_FIELDS */

/* -------------------------------------------------------------------------- */
/* Public IEBIN v1 loader                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Open and parse IEBIN v1 metadata and resolve the weights path.
 *
 * See ie_io.h for the public contract.
 *
 * @param json_path Path to model.ie.json.
 * @param bin_path  Optional override to model.ie.bin.
 * @param out       Output descriptor.
 * @return IE_IO_OK on success, IE_IO_ERR_* on failure.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out) {
  if (!out) return IE_IO_ERR_ARGS;
  memset(out, 0, sizeof(*out));

  if (!json_path || !*json_path) return IE_IO_ERR_ARGS;
  cpyz(out->json_path, sizeof(out->json_path), json_path);

  char *jbuf = NULL;
  size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    return IE_IO_ERR_JSON;
  }

  int ver = 0;
  (void)scan_json_key_int(jbuf, "version", &ver);
  if (ver <= 0) ver = 1;
  out->version = ver;

  char dtype[16];
  dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  if (dtype[0] == '\0') cpyz(dtype, sizeof(dtype), "float32");
  cpyz(out->dtype, sizeof(out->dtype), dtype);

  char resolved_bin[512];
  resolved_bin[0] = '\0';
  if (bin_path && *bin_path) {
    cpyz(resolved_bin, sizeof(resolved_bin), bin_path);
  } else {
    char bin_key[256];
    bin_key[0] = '\0';
    (void)scan_json_key_string(jbuf, "bin", bin_key, sizeof(bin_key));
    if (bin_key[0] == '\0') {
      free(jbuf);
      return IE_IO_ERR_BIN_UNSPEC;
    }

    const char *slash = last_slash(json_path);
    if (slash) {
      char dir[512];
      size_t dn = (size_t)(slash - json_path);
      if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
      memcpy(dir, json_path, dn);
      dir[dn] = '\0';
      join_path(resolved_bin, sizeof(resolved_bin), dir, bin_key);
    } else {
      cpyz(resolved_bin, sizeof(resolved_bin), bin_key);
    }
  }
  cpyz(out->weights_path, sizeof(out->weights_path), resolved_bin);

  struct stat st;
  if (stat(out->weights_path, &st) != 0 || st.st_size < 0) {
    free(jbuf);
    return IE_IO_ERR_STAT;
  }
  out->bin_size_bytes = (size_t)st.st_size;
  out->loaded = 1;

#if defined(IE_WEIGHTS_HAS_DEDUP_FIELDS) && (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
  {
    const int dedup_enabled = env_flag_get("IE_DEDUP", 1);
    const int dedup_strict = env_flag_get("IE_DEDUP_STRICT", 0);

    if (dedup_enabled) {
      const char *slash = last_slash(json_path);
      char dir[512];
      dir[0] = '\0';
      if (slash) {
        size_t dn = (size_t)(slash - json_path);
        if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
        memcpy(dir, json_path, dn);
        dir[dn] = '\0';
      } else {
        cpyz(dir, sizeof(dir), ".");
      }

      char dedup_json[512];
      dedup_json[0] = '\0';
      join_path(dedup_json, sizeof(dedup_json), dir, "model.dedup.json");

      if (!file_exists_readable(dedup_json)) (void)ensure_dedup_manifest_link(dir);

      if (file_exists_readable(dedup_json)) {
        ie_weights_dedup_opts_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.prefault_policy = 0;

        ie_weights_dedup_t *dh = NULL;
        ie_wdedup_status_t dst = ie_weights_dedup_open(&dh, dir, &opts);
        if (dst == IE_WDEDUP_OK && dh) {
          out->is_dedup = 1;
          out->dedup_handle = (void *)dh;
        } else {
          if (dedup_strict) {
            free(jbuf);
            return IE_IO_ERR_JSON;
          }
          out->is_dedup = 0;
          out->dedup_handle = NULL;
        }
      }
    }
  }
#endif

  free(jbuf);
  return IE_IO_OK;
}

/**
 * @brief Touch the weights binary to verify readability.
 *
 * Performs small positional reads near the start and end (if file is large).
 *
 * @param w Opened weights descriptor.
 * @return IE_IO_OK on success, IE_IO_ERR_* on error.
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
 * @brief Close weights descriptor and release optional dedup handle.
 *
 * @param w Descriptor to close (may be NULL).
 */
void ie_weights_close(ie_weights_t *w) {
  if (!w) return;

#if defined(IE_WEIGHTS_HAS_DEDUP_FIELDS) && (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
  if (w->is_dedup && w->dedup_handle) {
    ie_weights_dedup_t *dh = (ie_weights_dedup_t *)w->dedup_handle;
    ie_weights_dedup_close(&dh);
    w->dedup_handle = NULL;
    w->is_dedup = 0;
  }
#endif
}

/* -------------------------------------------------------------------------- */
/* INT4 weight-only helpers (exploratory; not yet in public header)           */
/* -------------------------------------------------------------------------- */

/**
 * @brief Minimal parsed view of INT4 quantization metadata.
 */
struct ie_int4_quant_meta {
  int present;
  int per_row;
  char scale_bin[512];
  char pack[32];
  int zero_point;
  int symmetric;
};

/**
 * @brief Parse INT4-related metadata from model.ie.json (best-effort).
 *
 * The function looks for dtype in {"int4","q4","mixed"} and a "quant" object
 * containing fields like:
 * - per: "row" or other
 * - scale_bin: relative path to scales
 * - pack: packing identifier
 * - zp: zero point
 * - symmetric: boolean-ish
 *
 * @param json_path Path to model.ie.json.
 * @param out_meta  Output struct.
 * @return IE_IO_OK on success or when metadata not present; IE_IO_ERR_* on error.
 */
int ie_weights_read_int4_meta(const char *json_path, struct ie_int4_quant_meta *out_meta) {
  if (!json_path || !out_meta) return IE_IO_ERR_ARGS;
  memset(out_meta, 0, sizeof(*out_meta));

  char *jbuf = NULL;
  size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    return IE_IO_ERR_JSON;
  }

  char dtype[16];
  dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  if (!(ascii_ieq(dtype, "int4") || ascii_ieq(dtype, "q4") || ascii_ieq(dtype, "mixed"))) {
    free(jbuf);
    out_meta->present = 0;
    return IE_IO_OK;
  }

  const char *qb = NULL, *qe = NULL;
  if (scan_json_object_range(jbuf, "quant", &qb, &qe) != 1 || !qb || !qe || qb >= qe) {
    free(jbuf);
    out_meta->present = 0;
    return IE_IO_OK;
  }

  char per[32];
  per[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "per", per, sizeof(per));
  out_meta->per_row = ascii_ieq(per, "row") ? 1 : 0;

  char scale_bin_rel[512];
  scale_bin_rel[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "scale_bin", scale_bin_rel, sizeof(scale_bin_rel));

  char pack[32];
  pack[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "pack", pack, sizeof(pack));
  cpyz(out_meta->pack, sizeof(out_meta->pack), pack);

  char zp_str[32];
  zp_str[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "zp", zp_str, sizeof(zp_str));
  if (zp_str[0]) out_meta->zero_point = atoi(zp_str);

  char sym_str[32];
  sym_str[0] = '\0';
  (void)scan_json_key_string_in_range(qb, qe, "symmetric", sym_str, sizeof(sym_str));
  if (sym_str[0]) {
    out_meta->symmetric =
        (ascii_ieq(sym_str, "true") || strcmp(sym_str, "1") == 0) ? 1 : 0;
  }

  if (scale_bin_rel[0]) {
    const char *slash = last_slash(json_path);
    if (slash) {
      char dir[512];
      size_t dn = (size_t)(slash - json_path);
      if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
      memcpy(dir, json_path, dn);
      dir[dn] = '\0';
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
 * @brief Convert IEEE-754 half (fp16) bit-pattern to float (fp32).
 *
 * @param h 16-bit half bits.
 * @return float value.
 */
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h & 0x7C00u) >> 10;
  uint32_t mant = (h & 0x03FFu);
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      int e = -1;
      do {
        e++;
        mant <<= 1;
      } while ((mant & 0x0400u) == 0);
      mant &= 0x03FFu;
      f = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &f, sizeof(out));
  return out;
}

/**
 * @brief Decode packed INT4 weights + scales into float weights.
 *
 * @param packed_path Path to packed INT4 data.
 * @param scales_path Path to scales (fp16 or fp32).
 * @param rows        Rows.
 * @param cols        Cols.
 * @param per_row     Non-zero for per-row scales; otherwise per-tensor.
 * @param dst         Output buffer (rows*cols floats).
 * @return IE_IO_OK on success, IE_IO_ERR_* on failure.
 */
int ie_weights_decode_int4(const char *packed_path, const char *scales_path, size_t rows, size_t cols,
                           int per_row, float *dst) {
  if (!packed_path || !scales_path || !dst || rows == 0 || cols == 0) return IE_IO_ERR_ARGS;

  int fdw = open(packed_path, O_RDONLY);
  if (fdw < 0) return IE_IO_ERR_OPEN;

  struct stat stw;
  if (fstat(fdw, &stw) != 0) {
    close(fdw);
    return IE_IO_ERR_STAT;
  }

  const size_t need_packed = rows * ie_int4_rowbytes(cols);
  if ((size_t)stw.st_size < need_packed) {
    close(fdw);
    return IE_IO_ERR_READ;
  }

  uint8_t *buf_packed = (uint8_t *)malloc(need_packed);
  if (!buf_packed) {
    close(fdw);
    return IE_IO_ERR_ALLOC;
  }

  ssize_t rd = ie_pread(fdw, buf_packed, need_packed, 0);
  close(fdw);
  if (rd < 0 || (size_t)rd != need_packed) {
    free(buf_packed);
    return IE_IO_ERR_READ;
  }

  int fds = open(scales_path, O_RDONLY);
  if (fds < 0) {
    free(buf_packed);
    return IE_IO_ERR_OPEN;
  }

  struct stat sts;
  if (fstat(fds, &sts) != 0) {
    close(fds);
    free(buf_packed);
    return IE_IO_ERR_STAT;
  }

  size_t nsc = per_row ? rows : 1;
  size_t need16 = nsc * sizeof(uint16_t);
  size_t need32 = nsc * sizeof(float);

  int is16 = ((size_t)sts.st_size == need16);
  int is32 = ((size_t)sts.st_size == need32);
  if (!(is16 || is32)) {
    close(fds);
    free(buf_packed);
    return IE_IO_ERR_READ;
  }

  float *buf_scales = (float *)malloc(need32);
  if (!buf_scales) {
    close(fds);
    free(buf_packed);
    return IE_IO_ERR_ALLOC;
  }

  if (is32) {
    ssize_t rds = ie_pread(fds, buf_scales, need32, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need32) {
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_READ;
    }
  } else {
    uint16_t *tmp = (uint16_t *)malloc(need16);
    if (!tmp) {
      close(fds);
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_ALLOC;
    }
    ssize_t rds = ie_pread(fds, tmp, need16, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need16) {
      free(tmp);
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_READ;
    }
    for (size_t i = 0; i < nsc; ++i) buf_scales[i] = fp16_to_fp32(tmp[i]);
    free(tmp);
  }

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
