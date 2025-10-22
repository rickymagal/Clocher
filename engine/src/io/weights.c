/**
 * @file weights.c
 * @brief Implementation of the IEBIN v1 loader with relaxed JSON parsing.
 *
 * ## Design Goals
 * - **Zero third-party dependencies**: plain C, no JSON libraries.
 * - **Resilient scanning**: tolerate extra whitespace and fields.
 * - **Safety with -Werror**: no unsafe formatting, careful bounds checking.
 * - **Clarity**: straight-line code with explicit pre/post-conditions.
 *
 * The loader extracts a minimal header from `model.ie.json` and resolves the
 * path to `model.ie.bin`. It also exposes a small @ref ie_weights_touch()
 * routine to validate OS-level readability and optionally warm caches.
 */

/* Ensure POSIX functions (e.g., pread) are declared when available. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_io.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ========================================================================== */
/* Internal helpers (documented; @internal)                                    */
/* ========================================================================== */

/**
 * @internal
 * @brief Copy C string @p src into @p dst (NUL-terminated), truncating if needed.
 * @param dst   Destination buffer.
 * @param dstsz Destination size in bytes.
 * @param src   Source string (may be NULL).
 *
 * Ensures @p dst is always NUL-terminated if @p dstsz > 0.
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
 * @param out   Output buffer.
 * @param outsz Output buffer size in bytes.
 * @param dir   Directory path (may be NULL/empty).
 * @param file  File name or path (must not be NULL when used).
 *
 * Avoids format-string warnings and guarantees NUL termination.
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

  /* copy dir */
  size_t copyd = ld;
  if (copyd >= outsz) copyd = outsz - 1;
  if (copyd > 0) { memcpy(out + pos, dir, copyd); pos += copyd; }

  if (pos < outsz - 1 && need_slash) { out[pos++] = '/'; }

  /* copy file */
  if (pos < outsz - 1) {
    size_t room = (outsz - 1) - pos;
    size_t copyf = (lf > room) ? room : lf;
    if (copyf > 0) { memcpy(out + pos, file, copyf); pos += copyf; }
  }

  out[pos] = '\0';
}

/**
 * @internal
 * @brief Read entire text file into memory (NUL-terminated).
 * @param path    Input file path.
 * @param buf     Output pointer to heap buffer (caller must free on success).
 * @param len_out Optional output size (bytes) excluding NUL.
 * @return 0 on success; negative on error.
 *
 * Strips a UTF-8 BOM if present.
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
 * @param json    NUL-terminated JSON text.
 * @param key     Key name including quotes (e.g., `"version"`).
 * @param out_val Output integer.
 * @return 1 if found, 0 if not found, negative on bad args.
 */
static int scan_json_key_int(const char *json, const char *key, int *out_val) {
  if (!json || !key || !out_val) return -1;
  const char *p = json;
  const size_t klen = strlen(key);
  while ((p = strstr(p, key))) {
    if (p > json && p[-1] == '"' && p[klen] == '"') {
      const char *c = p + klen + 1;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != ':') { ++p; continue; }
      ++c;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      char *end = NULL;
      long v = strtol(c, &end, 10);
      if (end && end != c) { *out_val = (int)v; return 1; }
    }
    ++p;
  }
  return 0;
}

/**
 * @internal
 * @brief Find string value of `"key": "<value>"` via relaxed scan.
 * @param json  NUL-terminated JSON text.
 * @param key   Key name including quotes (e.g., `"dtype"` or `"bin"`).
 * @param dst   Output buffer.
 * @param dstsz Output buffer size in bytes.
 * @return 1 if found, 0 if not found, negative on bad args.
 */
static int scan_json_key_string(const char *json, const char *key, char *dst, size_t dstsz) {
  if (!json || !key || !dst || dstsz == 0) return -1;
  dst[0] = '\0';
  const char *p = json;
  const size_t klen = strlen(key);
  while ((p = strstr(p, key))) {
    if (p > json && p[-1] == '"' && p[klen] == '"') {
      const char *c = p + klen + 1;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != ':') { ++p; continue; }
      ++c;
      while (*c == ' ' || *c == '\t' || *c == '\r' || *c == '\n') ++c;
      if (*c != '"') { ++p; continue; }
      ++c;
      const char *start = c;
      while (*c && *c != '"') ++c;
      size_t n = (size_t)(c - start);
      if (n >= dstsz) n = dstsz - 1;
      memcpy(dst, start, n);
      dst[n] = '\0';
      return 1;
    }
    ++p;
  }
  return 0;
}

/**
 * @internal
 * @brief Portable positional read helper.
 *
 * Uses `pread(2)` when available (declared under POSIX feature macros). If not
 * declared by the system headers, falls back to `lseek+read` while preserving
 * the original file offset.
 *
 * @param fd     File descriptor opened for reading.
 * @param buf    Destination buffer.
 * @param count  Maximum bytes to read.
 * @param offset Absolute byte offset in the file to read from.
 * @return Number of bytes read (>=0) on success, or -1 on error with errno set.
 */
static ssize_t ie_pread(int fd, void *buf, size_t count, off_t offset) {
  /* If pread is declared, use it directly. */
  #if defined(_XOPEN_SOURCE) || defined(_POSIX_C_SOURCE)
  /* Some libcs still require the symbol even if declared, so try and use it. */
  #ifdef __GLIBC__
  return pread(fd, buf, count, offset);
  #else
  /* Attempt direct call; if not linked, fallback below at compile-time. */
  return pread(fd, buf, count, offset);
  #endif
  #else
  /* Fallback: save/restore file position around a plain read. */
  off_t cur = lseek(fd, 0, SEEK_CUR);
  if (cur == (off_t)-1) return -1;
  if (lseek(fd, offset, SEEK_SET) == (off_t)-1) return -1;
  ssize_t r = read(fd, buf, count);
  /* Best-effort restore; ignore restore error to preserve original read errno. */
  (void)lseek(fd, cur, SEEK_SET);
  return r;
  #endif
}

/* ========================================================================== */
/* Public API                                                                 */
/* ========================================================================== */

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
  if (dtype[0] == '\0') cpyz(dtype, sizeof(dtype), "float32");
  cpyz(out->dtype, sizeof(out->dtype), dtype);

  /* Determine weights path */
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

  free(jbuf);
  return IE_IO_OK;
}

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

void ie_weights_close(ie_weights_t *w) {
  (void)w;
  /* Currently a no-op; reserved for future resource ownership. */
}
