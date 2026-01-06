/* ============================================================================
 * File: engine/src/io/weights.c
 * ============================================================================
 */
/**
 * @file weights.c
 * @brief IEBIN v1 loader with relaxed JSON scanning and detailed diagnostics.
 *
 * @details
 * This module implements a minimal dependency, plain-C loader for the IEBIN v1
 * artifact pair:
 *
 *  - model.ie.json (metadata and tensor descriptors)
 *  - model.ie.bin  (raw tensor bytes, typically mmap'd by higher layers)
 *
 * The loader is intentionally tolerant to extra fields and formatting changes
 * in the JSON file. It does not use a third-party JSON parser. Instead, it
 * performs targeted scans for a small set of required keys.
 *
 * ## Logging
 * All logs in this module go to stderr via util_logging and will not interfere
 * with stdout JSON outputs from benchmarks or CLI tools.
 *
 * Logging verbosity is controlled via environment variable IE_LOG_LEVEL:
 *  - 0: silent
 *  - 1: errors only
 *  - 2: errors + warnings + info (default)
 *  - 3: errors + warnings + info + debug
 *
 * The log messages are designed to make failures actionable by printing:
 *  - Which file path was attempted (json/bin/dedup artifacts)
 *  - Which schema keys were found or missing
 *  - File sizes and short-read conditions
 *  - Dedup enable/strict flags and decision outcomes
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ie_io.h"         /* ie_weights_t, IE_IO_* status codes */
#include "ie_quant_int4.h" /* INT4 quant packing/dequant helpers */
#include "util_logging.h"  /* ie_log_info/warn/error (stderr) */
#include "weights_dedup.h" /* NOTE: include path has NO subfolders */

#include <ctype.h> /* tolower */
#include <errno.h>
#include <fcntl.h>
#include <stdint.h> /* uint16_t, uint32_t */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> /* strcasecmp */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* Force-enable the optional dedup fields unless the build explicitly disables them. */
#ifndef IE_WEIGHTS_HAS_DEDUP_FIELDS
#define IE_WEIGHTS_HAS_DEDUP_FIELDS 1
#endif

/* -------------------------------------------------------------------------- */
/* Logging helpers                                                            */
/* -------------------------------------------------------------------------- */

/**
 * @brief Read IE_LOG_LEVEL once and cache it for the process lifetime.
 *
 * @details
 * This module uses a local log level cache to avoid repeated getenv calls.
 * The log level defaults to 2 (info) to aid diagnostics during development.
 *
 * @return Cached log level in range [0,3].
 */
static int ie_weights_log_level(void) {
  static int init = 0;
  static int lvl = 2; /* Default: info. */

  if (init) return lvl;
  init = 1;

  const char *v = getenv("IE_LOG_LEVEL");
  if (!v || !*v) return lvl;

  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (end && end != v) {
    if (x < 0) x = 0;
    if (x > 3) x = 3;
    lvl = (int)x;
    return lvl;
  }

  /* Best-effort string forms. */
  if (strcasecmp(v, "silent") == 0 || strcasecmp(v, "quiet") == 0) {
    lvl = 0;
  } else if (strcasecmp(v, "error") == 0 || strcasecmp(v, "errors") == 0) {
    lvl = 1;
  } else if (strcasecmp(v, "info") == 0) {
    lvl = 2;
  } else if (strcasecmp(v, "debug") == 0) {
    lvl = 3;
  }
  return lvl;
}

/**
 * @brief Return non-zero if the requested logging level is enabled.
 *
 * @param level 1=error, 2=info, 3=debug.
 * @return 1 if enabled, else 0.
 */
static int ie_weights_log_enabled(int level) {
  return ie_weights_log_level() >= level;
}

/** @brief Emit an info log if info logging is enabled. */
#define WLOGI(...)                 \
  do {                             \
    if (ie_weights_log_enabled(2)) \
      ie_log_info(__VA_ARGS__);    \
  } while (0)

/** @brief Emit a debug log if debug logging is enabled. */
#define WLOGD(...)                 \
  do {                             \
    if (ie_weights_log_enabled(3)) \
      ie_log_info(__VA_ARGS__);    \
  } while (0)

/** @brief Emit a warning log if info logging is enabled. */
#define WLOGW(...)                 \
  do {                             \
    if (ie_weights_log_enabled(2)) \
      ie_log_warn(__VA_ARGS__);    \
  } while (0)

/** @brief Emit an error log if error logging is enabled. */
#define WLOGE(...)                 \
  do {                             \
    if (ie_weights_log_enabled(1)) \
      ie_log_error(__VA_ARGS__);   \
  } while (0)

/**
 * @brief Log an errno-based failure with contextual path information.
 *
 * @param what Short operation label (e.g., "fopen", "stat").
 * @param path Associated file path (may be NULL).
 */
static void wlog_errno_(const char *what, const char *path) {
  if (!ie_weights_log_enabled(1)) return;
  if (path && *path) {
    WLOGE("%s failed: path='%s' errno=%d (%s)", what, path, errno, strerror(errno));
  } else {
    WLOGE("%s failed: errno=%d (%s)", what, errno, strerror(errno));
  }
}

/* -------------------------------------------------------------------------- */
/* Small helpers                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @brief Skip ASCII whitespace characters in a string.
 *
 * @param p Input pointer (may be NULL).
 * @return Pointer advanced past whitespace, or NULL if @p p is NULL.
 */
static const char *skip_ws(const char *p) {
  if (!p) return NULL;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  return p;
}

/**
 * @brief Copy a C-string into a bounded buffer and always NUL-terminate.
 *
 * @param dst Destination buffer (may be NULL).
 * @param dstsz Capacity in bytes.
 * @param src Source string (may be NULL).
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
 * @brief Return a pointer to the last '/' in a path, or NULL if none.
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
 * @brief Join directory and filename into an output buffer (NUL-terminated).
 *
 * @details
 * This function performs a bounded concatenation:
 *  - If @p dir ends with '/', it is not duplicated.
 *  - If @p dir is empty, @p file is copied verbatim.
 *
 * @param out Output buffer.
 * @param outsz Output capacity.
 * @param dir Directory path (may be NULL/empty).
 * @param file Filename (must be non-empty).
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
 * @details
 * - Reads in binary mode.
 * - Strips a UTF-8 BOM if present.
 * - Always NUL-terminates the returned buffer.
 *
 * @param path File path.
 * @param buf Output pointer to malloc'd buffer (caller frees).
 * @param len_out Optional output length (bytes, excluding NUL).
 * @return 0 on success, negative on failure.
 */
static int read_all_text(const char *path, char **buf, size_t *len_out) {
  if (!path || !buf) return -1;
  *buf = NULL;
  if (len_out) *len_out = 0;

  WLOGD("read_all_text: path='%s'", path);

  FILE *f = fopen(path, "rb");
  if (!f) {
    wlog_errno_("fopen", path);
    return -2;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    wlog_errno_("fseek", path);
    fclose(f);
    return -3;
  }
  long n = ftell(f);
  if (n < 0) {
    wlog_errno_("ftell", path);
    fclose(f);
    return -3;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    wlog_errno_("fseek", path);
    fclose(f);
    return -3;
  }

  size_t size = (size_t)n;
  char *mem = (char *)malloc(size + 1);
  if (!mem) {
    WLOGE("read_all_text: malloc failed (bytes=%zu) path='%s'", size + 1, path);
    fclose(f);
    return -4;
  }

  size_t rd = fread(mem, 1, size, f);
  fclose(f);
  if (rd != size) {
    WLOGE("read_all_text: short read (got=%zu want=%zu) path='%s'", rd, size, path);
    free(mem);
    return -5;
  }

  if (size >= 3 && (unsigned char)mem[0] == 0xEF && (unsigned char)mem[1] == 0xBB &&
      (unsigned char)mem[2] == 0xBF) {
    WLOGD("read_all_text: stripping UTF-8 BOM path='%s'", path);
    size -= 3;
    memmove(mem, mem + 3, size);
  }

  mem[size] = '\0';
  *buf = mem;
  if (len_out) *len_out = size;

  WLOGD("read_all_text: ok (bytes=%zu) path='%s'", size, path);
  return 0;
}

/**
 * @brief Scan a relaxed JSON blob for an integer value at a given key.
 *
 * @details
 * This is a best-effort scan that:
 *  - searches for `"key"` occurrences,
 *  - expects a ':' after the key,
 *  - parses a base-10 integer with strtol().
 *
 * It does not validate full JSON structure.
 *
 * @param json NUL-terminated text.
 * @param key Key name without quotes (e.g., "version").
 * @param out_val Output integer.
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
      c = skip_ws(c);
      if (!c || *c != ':') continue;
      ++c;
      c = skip_ws(c);
      if (!c) continue;
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
 * @brief Unescape a JSON string into a destination buffer (best-effort).
 *
 * @details
 * Supported escapes:
 *  - \" \\ \/
 *  - \n \r \t
 *
 * Unknown escapes are handled by dropping the backslash and copying the escaped
 * character verbatim.
 *
 * @param dst Destination buffer.
 * @param dstsz Destination capacity.
 * @param src Source buffer (not NUL-terminated; length provided).
 * @param srclen Source length in bytes.
 * @return Number of bytes written (excluding NUL terminator).
 */
static size_t json_unescape_into(char *dst, size_t dstsz, const char *src, size_t srclen) {
  if (!dst || dstsz == 0) return 0;
  size_t di = 0;
  for (size_t si = 0; si < srclen && di + 1 < dstsz; ++si) {
    char ch = src[si];
    if (ch == '\\' && si + 1 < srclen) {
      char n = src[si + 1];
      if (n == '"' || n == '\\' || n == '/') {
        dst[di++] = n;
        ++si;
        continue;
      }
      if (n == 'n') {
        dst[di++] = '\n';
        ++si;
        continue;
      }
      if (n == 'r') {
        dst[di++] = '\r';
        ++si;
        continue;
      }
      if (n == 't') {
        dst[di++] = '\t';
        ++si;
        continue;
      }
      /* Unknown escape: drop backslash, keep char. */
      dst[di++] = n;
      ++si;
      continue;
    }
    dst[di++] = ch;
  }
  dst[di] = '\0';
  return di;
}

/**
 * @brief Scan a relaxed JSON blob for a string value at a given key.
 *
 * @details
 * Finds `"key": "value"` and copies a minimally unescaped `value` into @p dst.
 * This scan does not validate full JSON structure; it is tolerant to extra
 * whitespace and unrelated fields.
 *
 * @param json NUL-terminated JSON-ish text.
 * @param key Key name without quotes.
 * @param dst Output buffer.
 * @param dstsz Output capacity.
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
      c = skip_ws(c);
      if (!c || *c != ':') continue;
      ++c;
      c = skip_ws(c);
      if (!c || *c != '"') continue;
      ++c;

      const char *start = c;
      while (*c) {
        if (*c == '"' && (c == start || c[-1] != '\\')) break;
        ++c;
      }
      if (*c != '"') return 0;

      size_t raw_len = (size_t)(c - start);
      (void)json_unescape_into(dst, dstsz, start, raw_len);
      return 1;
    }
  }
  return 0;
}

/**
 * @brief Locate the interior byte range of an object value "{ ... }" for a key.
 *
 * @details
 * This helper attempts to locate `"key": { ... }` and returns a range that
 * covers the interior of the braces. Strings are respected when tracking brace
 * depth to avoid premature termination.
 *
 * @param json JSON-ish text.
 * @param key Object key to find.
 * @param out_begin Output begin pointer (inside the braces).
 * @param out_end Output end pointer (points to closing brace).
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
      c = skip_ws(c);
      if (!c || *c != ':') continue;
      ++c;
      c = skip_ws(c);
      if (!c || *c != '{') return 0;

      int depth = 0;
      const char *start = c + 1;
      const char *q = c;
      int in_str = 0;
      while (*q) {
        char ch = *q;
        if (in_str) {
          if (ch == '\\' && q[1]) {
            q += 2;
            continue;
          }
          if (ch == '"') in_str = 0;
          ++q;
          continue;
        }

        if (ch == '"') {
          in_str = 1;
          ++q;
          continue;
        }
        if (ch == '{') ++depth;
        if (ch == '}') {
          --depth;
          if (depth == 0) {
            *out_begin = start;
            *out_end = q;
            return 1;
          }
        }
        ++q;
      }
      return 0;
    }
  }
  return 0;
}

/**
 * @brief Scan for a string key within a bounded [begin,end] range.
 *
 * @details
 * This is used for nested objects discovered by scan_json_object_range().
 *
 * @param begin Range begin.
 * @param end Range end (inclusive, points to closing brace or last valid byte).
 * @param key Key name.
 * @param dst Output buffer.
 * @param dstsz Output capacity.
 * @return 1 if found, 0 if not found, negative on invalid args.
 */
static int scan_json_key_string_in_range(const char *begin, const char *end, const char *key,
                                         char *dst, size_t dstsz) {
  if (!begin || !end || !key || !dst || dstsz == 0) return -1;
  dst[0] = '\0';
  if (end < begin) return -1;

  const size_t klen = strlen(key);
  const char *p = begin;
  const char *limit = end + 1;

  while (p < limit) {
    const char *q = memchr(p, '"', (size_t)(limit - p));
    if (!q) break;
    ++q;
    if ((size_t)(limit - q) >= klen && memcmp(q, key, klen) == 0 && q[klen] == '"') {
      const char *c = q + klen + 1;
      c = skip_ws(c);
      if (!c || c >= limit || *c != ':') {
        p = q + 1;
        continue;
      }
      ++c;
      c = skip_ws(c);
      if (!c || c >= limit || *c != '"') {
        p = q + 1;
        continue;
      }
      ++c;

      const char *start = c;
      while (c < limit) {
        if (*c == '"' && (c == start || c[-1] != '\\')) break;
        ++c;
      }
      if (c >= limit || *c != '"') return 0;

      size_t raw_len = (size_t)(c - start);
      (void)json_unescape_into(dst, dstsz, start, raw_len);
      return 1;
    }
    p = q + 1;
  }
  return 0;
}

/**
 * @brief Try multiple schema variants to locate the weights binary path.
 *
 * @details
 * The loader tolerates different JSON shapes by checking:
 *  - "bin"
 *  - "weights_bin"
 *  - "weightsBin"
 *  - "weights_path"
 *  - "weightsPath"
 *  - "weights": { "bin": ... }
 *  - "iebin":   { "bin": ... }
 *
 * @param json NUL-terminated JSON-ish text.
 * @param dst Output buffer for the path value.
 * @param dstsz Output capacity.
 * @return 1 if found, 0 if not found, negative on invalid args.
 */
static int scan_json_best_effort_bin(const char *json, char *dst, size_t dstsz) {
  if (!json || !dst || dstsz == 0) return -1;
  dst[0] = '\0';

  /* Common direct keys. */
  if (scan_json_key_string(json, "bin", dst, dstsz) == 1 && dst[0]) {
    WLOGD("scan_json_best_effort_bin: found key='bin' value='%s'", dst);
    return 1;
  }
  if (scan_json_key_string(json, "weights_bin", dst, dstsz) == 1 && dst[0]) {
    WLOGD("scan_json_best_effort_bin: found key='weights_bin' value='%s'", dst);
    return 1;
  }
  if (scan_json_key_string(json, "weightsBin", dst, dstsz) == 1 && dst[0]) {
    WLOGD("scan_json_best_effort_bin: found key='weightsBin' value='%s'", dst);
    return 1;
  }
  if (scan_json_key_string(json, "weights_path", dst, dstsz) == 1 && dst[0]) {
    WLOGD("scan_json_best_effort_bin: found key='weights_path' value='%s'", dst);
    return 1;
  }
  if (scan_json_key_string(json, "weightsPath", dst, dstsz) == 1 && dst[0]) {
    WLOGD("scan_json_best_effort_bin: found key='weightsPath' value='%s'", dst);
    return 1;
  }

  /* Nested objects. */
  {
    const char *b = NULL;
    const char *e = NULL;
    if (scan_json_object_range(json, "weights", &b, &e) == 1 && b && e && b <= e) {
      if (scan_json_key_string_in_range(b, e, "bin", dst, dstsz) == 1 && dst[0]) {
        WLOGD("scan_json_best_effort_bin: found key='weights.bin' value='%s'", dst);
        return 1;
      }
      if (scan_json_key_string_in_range(b, e, "weights_bin", dst, dstsz) == 1 && dst[0]) {
        WLOGD("scan_json_best_effort_bin: found key='weights.weights_bin' value='%s'", dst);
        return 1;
      }
    }
  }
  {
    const char *b = NULL;
    const char *e = NULL;
    if (scan_json_object_range(json, "iebin", &b, &e) == 1 && b && e && b <= e) {
      if (scan_json_key_string_in_range(b, e, "bin", dst, dstsz) == 1 && dst[0]) {
        WLOGD("scan_json_best_effort_bin: found key='iebin.bin' value='%s'", dst);
        return 1;
      }
      if (scan_json_key_string_in_range(b, e, "weights_bin", dst, dstsz) == 1 && dst[0]) {
        WLOGD("scan_json_best_effort_bin: found key='iebin.weights_bin' value='%s'", dst);
        return 1;
      }
    }
  }

  WLOGW("scan_json_best_effort_bin: no recognized bin key found");
  return 0;
}

/**
 * @brief Portable pread wrapper with fallback.
 *
 * @details
 * When pread(2) is unavailable, falls back to lseek+read and restores the file
 * offset. This is slower but sufficient for small diagnostic reads.
 *
 * @param fd Open file descriptor.
 * @param buf Destination buffer.
 * @param count Number of bytes to read.
 * @param offset Offset from start.
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
 * @param a NUL-terminated string.
 * @param b NUL-terminated string.
 * @return 1 if equal (ASCII case-insensitive), else 0.
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

/**
 * @brief Parse environment flag-like strings into a boolean.
 *
 * @details
 * Accepts: 0/1, false/true, no/yes, off/on (ASCII case-insensitive).
 * Unknown values default to enabled (1).
 *
 * @param name Environment variable name.
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
 * @brief Create a best-effort symlink model.dedup.json -> dedup_manifest.json.
 *
 * @details
 * Some toolchains may produce dedup artifacts under a different canonical name.
 * This helper attempts to create a symlink for compatibility. Failures are not
 * fatal unless the caller is operating under strict mode.
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
  if (file_exists_readable(linkpath)) {
    WLOGD("ensure_dedup_manifest_link: already exists path='%s'", linkpath);
    return 1;
  }

  join_path(manifest, sizeof(manifest), dir, "dedup_manifest.json");
  if (!file_exists_readable(manifest)) {
    WLOGD("ensure_dedup_manifest_link: missing manifest path='%s'", manifest);
    return 0;
  }

  cpyz(target, sizeof(target), "dedup_manifest.json");
  WLOGI("ensure_dedup_manifest_link: creating symlink '%s' -> '%s'", linkpath, target);

  int rc = symlink(target, linkpath);
  if (rc != 0) {
    if (errno != EEXIST) {
      wlog_errno_("symlink", linkpath);
    }
  }

  if (!file_exists_readable(linkpath)) {
    WLOGW("ensure_dedup_manifest_link: symlink not readable path='%s'", linkpath);
    return 0;
  }
  return 1;
}

/**
 * @brief Touch a file by doing small reads near start/end.
 *
 * @details
 * This is intended as a fast readability check and optional OS cache warming.
 * It is deliberately minimal and does not attempt to validate file contents.
 *
 * @param path Path to file.
 * @return 1 if read succeeded, 0 otherwise.
 */
static int touch_file_small(const char *path) {
  if (!path || !*path) return 0;
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    wlog_errno_("open", path);
    return 0;
  }

  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 0) {
    wlog_errno_("fstat", path);
    close(fd);
    return 0;
  }

  unsigned char buf[4096];
  ssize_t r1 = ie_pread(fd, buf, sizeof(buf), 0);
  ssize_t r2 = 0;
  if ((size_t)st.st_size > sizeof(buf)) {
    off_t off = (off_t)((size_t)st.st_size - sizeof(buf));
    r2 = ie_pread(fd, buf, sizeof(buf), off);
  }
  close(fd);

  if (r1 < 0 || r2 < 0) {
    WLOGW("touch_file_small: read failed path='%s' size=%zu", path, (size_t)st.st_size);
    return 0;
  }

  WLOGD("touch_file_small: ok path='%s' size=%zu", path, (size_t)st.st_size);
  return 1;
}

/* -------------------------------------------------------------------------- */
/* Public IEBIN v1 loader                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Open and parse an IEBIN v1 model description.
 *
 * @details
 * This function:
 *  - reads model.ie.json,
 *  - parses minimal metadata (version, dtype),
 *  - resolves the path to model.ie.bin using overrides and schema variants,
 *  - validates the weights binary size via stat(2),
 *  - optionally opens dedup artifacts when enabled.
 *
 * In all error cases, the output structure is left in a safe-to-close state.
 *
 * @param json_path Path to model.ie.json.
 * @param bin_path Optional override for the binary path (may be NULL).
 * @param out Output weights descriptor.
 * @return IE_IO_OK on success; negative ie_io_status_t on failure.
 */
int ie_weights_open(const char *json_path, const char *bin_path, ie_weights_t *out) {
  if (!out) return IE_IO_ERR_ARGS;
  memset(out, 0, sizeof(*out));

  if (!json_path || !*json_path) return IE_IO_ERR_ARGS;
  cpyz(out->json_path, sizeof(out->json_path), json_path);

  WLOGI("ie_weights_open: begin json_path='%s' bin_override='%s'", json_path,
        (bin_path && *bin_path) ? bin_path : "");

  char *jbuf = NULL;
  size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    WLOGE("ie_weights_open: failed to read json (path='%s')", json_path);
    return IE_IO_ERR_JSON;
  }
  WLOGD("ie_weights_open: loaded json bytes=%zu", jlen);

  int ver = 0;
  int found_ver = scan_json_key_int(jbuf, "version", &ver);
  if (ver <= 0) ver = 1;
  out->version = ver;

  char dtype[16];
  dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  if (dtype[0] == '\0') cpyz(dtype, sizeof(dtype), "float32");
  cpyz(out->dtype, sizeof(out->dtype), dtype);

  WLOGI("ie_weights_open: parsed version=%d (found=%d) dtype='%s'", out->version, found_ver,
        out->dtype);

  char resolved_bin[512];
  resolved_bin[0] = '\0';

  /* Priority order:
   *  1) explicit bin_path parameter
   *  2) IE_WEIGHTS_BIN env override
   *  3) JSON keys (bin / weights_bin / nested weights{bin} / iebin{bin})
   *  4) fallback: sibling 'model.ie.bin' next to json_path
   */
  if (bin_path && *bin_path) {
    cpyz(resolved_bin, sizeof(resolved_bin), bin_path);
    WLOGI("ie_weights_open: using bin override (arg) path='%s'", resolved_bin);
  } else {
    const char *env_bin = getenv("IE_WEIGHTS_BIN");
    if (env_bin && *env_bin) {
      cpyz(resolved_bin, sizeof(resolved_bin), env_bin);
      WLOGI("ie_weights_open: using bin override (env IE_WEIGHTS_BIN) path='%s'", resolved_bin);
    } else {
      char bin_key[256];
      bin_key[0] = '\0';

      const int found_bin_key = scan_json_best_effort_bin(jbuf, bin_key, sizeof(bin_key));
      if (found_bin_key == 1 && bin_key[0] != '\0') {
        const char *slash = last_slash(json_path);
        if (slash) {
          char dir[512];
          size_t dn = (size_t)(slash - json_path);
          if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
          memcpy(dir, json_path, dn);
          dir[dn] = '\0';
          join_path(resolved_bin, sizeof(resolved_bin), dir, bin_key);
          WLOGI("ie_weights_open: resolved bin from json key='%s' dir='%s' => '%s'", bin_key, dir,
                resolved_bin);
        } else {
          cpyz(resolved_bin, sizeof(resolved_bin), bin_key);
          WLOGI("ie_weights_open: resolved bin from json key='%s' => '%s'", bin_key, resolved_bin);
        }
      } else {
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

        char fallback[512];
        fallback[0] = '\0';
        join_path(fallback, sizeof(fallback), dir, "model.ie.bin");

        if (file_exists_readable(fallback)) {
          cpyz(resolved_bin, sizeof(resolved_bin), fallback);
          WLOGW(
              "ie_weights_open: JSON did not specify weights bin path; defaulting to sibling "
              "model.ie.bin path='%s'",
              resolved_bin);
        } else {
          WLOGE("ie_weights_open: missing weights bin path in JSON (no known key matched)");
          WLOGE("ie_weights_open: also missing fallback sibling path='%s'", fallback);
          free(jbuf);
          return IE_IO_ERR_BIN_UNSPEC;
        }
      }
    }
  }

  cpyz(out->weights_path, sizeof(out->weights_path), resolved_bin);

  struct stat st;
  if (stat(out->weights_path, &st) != 0 || st.st_size < 0) {
    wlog_errno_("stat", out->weights_path);
    WLOGE("ie_weights_open: weights binary not accessible path='%s'", out->weights_path);
    free(jbuf);
    return IE_IO_ERR_STAT;
  }
  out->bin_size_bytes = (size_t)st.st_size;
  out->loaded = 1;

  WLOGI("ie_weights_open: weights binary ok path='%s' size_bytes=%zu", out->weights_path,
        out->bin_size_bytes);

#if (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
  {
    const int dedup_enabled = env_flag_get("IE_DEDUP", 0);
    const int dedup_strict = env_flag_get("IE_DEDUP_STRICT", 0);

    WLOGD("ie_weights_open: dedup flags IE_DEDUP=%d IE_DEDUP_STRICT=%d", dedup_enabled,
          dedup_strict);

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

      if (!file_exists_readable(dedup_json)) {
        WLOGI("ie_weights_open: dedup enabled but model.dedup.json missing; attempting link");
        (void)ensure_dedup_manifest_link(dir);
      }

      if (file_exists_readable(dedup_json)) {
        ie_weights_dedup_opts_t opts;
        memset(&opts, 0, sizeof(opts));
        opts.prefault_policy = 0;

        ie_weights_dedup_t *dh = NULL;
        ie_wdedup_status_t dst = ie_weights_dedup_open(&dh, dir, &opts);
        if (dst == IE_WDEDUP_OK && dh) {
          out->is_dedup = 1;
          out->dedup_handle = (void *)dh;
          WLOGI("ie_weights_open: dedup artifacts opened (dir='%s')", dir);
        } else {
          WLOGW("ie_weights_open: dedup open failed (status=%d dir='%s')", (int)dst, dir);
          if (dedup_strict) {
            WLOGE("ie_weights_open: dedup strict mode: failing open");
            free(jbuf);
            return IE_IO_ERR_JSON;
          }
          out->is_dedup = 0;
          out->dedup_handle = NULL;
        }
      } else {
        WLOGW("ie_weights_open: dedup enabled but model.dedup.json still missing (dir='%s')", dir);
        if (dedup_strict) {
          WLOGE("ie_weights_open: dedup strict mode: failing open");
          free(jbuf);
          return IE_IO_ERR_JSON;
        }
      }
    }
  }
#endif

  free(jbuf);
  WLOGI("ie_weights_open: success");
  return IE_IO_OK;
}

/**
 * @brief Touch the weights binary to verify readability and warm OS caches.
 *
 * @details
 * This performs small reads near the start and end of the binary. When dedup
 * is enabled, also checks auxiliary dedup files.
 *
 * @param w Opened weights descriptor.
 * @return IE_IO_OK on success; negative ie_io_status_t on failure.
 */
int ie_weights_touch(const ie_weights_t *w) {
  if (!w || !w->weights_path[0]) return IE_IO_ERR_ARGS;

  WLOGI("ie_weights_touch: begin path='%s' size_bytes=%zu", w->weights_path, w->bin_size_bytes);

  int fd = open(w->weights_path, O_RDONLY);
  if (fd < 0) {
    wlog_errno_("open", w->weights_path);
    return IE_IO_ERR_OPEN;
  }

  unsigned char buf[4096];
  ssize_t r1 = ie_pread(fd, buf, sizeof(buf), 0);
  ssize_t r2 = 0;
  if (w->bin_size_bytes > sizeof(buf)) {
    off_t off = (off_t)(w->bin_size_bytes - sizeof(buf));
    r2 = ie_pread(fd, buf, sizeof(buf), off);
  }
  close(fd);

  if (r1 < 0 || r2 < 0) {
    WLOGE("ie_weights_touch: read failed (r1=%zd r2=%zd) path='%s'", r1, r2, w->weights_path);
    return IE_IO_ERR_READ;
  }

  WLOGD("ie_weights_touch: ok primary binary reads (r1=%zd r2=%zd)", r1, r2);

#if (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
  {
    const int dedup_enabled = env_flag_get("IE_DEDUP", 0);
    const int dedup_strict = env_flag_get("IE_DEDUP_STRICT", 0);

    WLOGD("ie_weights_touch: dedup flags IE_DEDUP=%d IE_DEDUP_STRICT=%d", dedup_enabled,
          dedup_strict);

    if (dedup_enabled) {
      const char *base = (w->json_path[0] ? w->json_path : w->weights_path);
      const char *slash = last_slash(base);

      char dir[512];
      dir[0] = '\0';
      if (slash) {
        size_t dn = (size_t)(slash - base);
        if (dn >= sizeof(dir)) dn = sizeof(dir) - 1;
        memcpy(dir, base, dn);
        dir[dn] = '\0';
      } else {
        cpyz(dir, sizeof(dir), ".");
      }

      char p_dedup[512], p_def[512], p_exc[512], p_msk[512];
      p_dedup[0] = p_def[0] = p_exc[0] = p_msk[0] = '\0';

      join_path(p_dedup, sizeof(p_dedup), dir, "model.dedup.json");
      join_path(p_def, sizeof(p_def), dir, "model.defaults.bin");
      join_path(p_exc, sizeof(p_exc), dir, "model.exceptions.bin");
      join_path(p_msk, sizeof(p_msk), dir, "model.masks.bin");

      int ok_dedup = touch_file_small(p_dedup);
      int ok_def = touch_file_small(p_def);
      int ok_exc = touch_file_small(p_exc);
      int ok_msk = touch_file_small(p_msk);

      WLOGI("ie_weights_touch: dedup touch results dedup=%d defaults=%d exceptions=%d masks=%d",
            ok_dedup, ok_def, ok_exc, ok_msk);

      if (dedup_strict && (!(ok_dedup && ok_def && ok_exc && ok_msk))) {
        WLOGE("ie_weights_touch: dedup strict mode: missing or unreadable artifact(s)");
        return IE_IO_ERR_JSON;
      }
    }
  }
#endif

  WLOGI("ie_weights_touch: success");
  return IE_IO_OK;
}

/**
 * @brief Close an opened weights descriptor and release optional dedup handles.
 *
 * @details
 * This function is idempotent and safe to call on partially initialized
 * structures. It does not unlink or delete any files; it only releases
 * in-memory handles.
 *
 * @param w Weights descriptor (may be NULL).
 */
void ie_weights_close(ie_weights_t *w) {
  if (!w) return;

  WLOGD("ie_weights_close: begin loaded=%d is_dedup=%d", w->loaded, w->is_dedup);

#if (IE_WEIGHTS_HAS_DEDUP_FIELDS == 1)
  if (w->is_dedup && w->dedup_handle) {
    ie_weights_dedup_t *dh = (ie_weights_dedup_t *)w->dedup_handle;
    ie_weights_dedup_close(&dh);
    w->dedup_handle = NULL;
    w->is_dedup = 0;
    WLOGD("ie_weights_close: dedup handle released");
  }
#endif
}

/* -------------------------------------------------------------------------- */
/* INT4 weight-only helpers (exploratory; not yet in public header)           */
/* -------------------------------------------------------------------------- */

/**
 * @struct ie_int4_quant_meta
 * @brief Minimal INT4 quantization metadata extracted from model.ie.json.
 *
 * @details
 * This structure is currently used by diagnostics and exploratory tooling.
 * It is not part of the stable public API and may change without notice.
 */
struct ie_int4_quant_meta {
  int present;        /**< Non-zero if INT4 metadata is present and supported. */
  int per_row;        /**< Non-zero if scales are per-row. */
  char scale_bin[512];/**< Resolved path to the scales binary. */
  char pack[32];      /**< Packing scheme identifier (best-effort string). */
  int zero_point;     /**< Zero-point value when applicable. */
  int symmetric;      /**< Non-zero if symmetric quantization is indicated. */
};

/**
 * @brief Read INT4 quantization metadata from a model.ie.json file.
 *
 * @details
 * This function:
 *  - checks the top-level "dtype" for int4-like modes,
 *  - looks for a "quant" object,
 *  - extracts fields such as "per", "scale_bin", "pack", "zp", and "symmetric".
 *
 * If INT4 metadata is absent or incompatible, the function returns IE_IO_OK
 * with out_meta->present set to 0.
 *
 * @param json_path Path to model.ie.json.
 * @param out_meta Output metadata structure.
 * @return IE_IO_OK on success; negative ie_io_status_t on failure.
 */
int ie_weights_read_int4_meta(const char *json_path, struct ie_int4_quant_meta *out_meta) {
  if (!json_path || !out_meta) return IE_IO_ERR_ARGS;
  memset(out_meta, 0, sizeof(*out_meta));

  WLOGI("ie_weights_read_int4_meta: begin json_path='%s'", json_path);

  char *jbuf = NULL;
  size_t jlen = 0;
  if (read_all_text(json_path, &jbuf, &jlen) != 0 || !jbuf || jlen == 0) {
    WLOGE("ie_weights_read_int4_meta: failed to read json");
    return IE_IO_ERR_JSON;
  }

  char dtype[16];
  dtype[0] = '\0';
  (void)scan_json_key_string(jbuf, "dtype", dtype, sizeof(dtype));
  WLOGD("ie_weights_read_int4_meta: dtype='%s'", dtype);

  if (!(ascii_ieq(dtype, "int4") || ascii_ieq(dtype, "q4") || ascii_ieq(dtype, "mixed"))) {
    free(jbuf);
    out_meta->present = 0;
    WLOGI("ie_weights_read_int4_meta: not an int4 dtype; present=0");
    return IE_IO_OK;
  }

  const char *qb = NULL, *qe = NULL;
  if (scan_json_object_range(jbuf, "quant", &qb, &qe) != 1 || !qb || !qe || qb > qe) {
    free(jbuf);
    out_meta->present = 0;
    WLOGW("ie_weights_read_int4_meta: missing quant object; present=0");
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
    out_meta->symmetric = (ascii_ieq(sym_str, "true") || strcmp(sym_str, "1") == 0) ? 1 : 0;
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

  WLOGI("ie_weights_read_int4_meta: present=1 per_row=%d pack='%s' zp=%d symmetric=%d scale_bin='%s'",
        out_meta->per_row, out_meta->pack, out_meta->zero_point, out_meta->symmetric,
        out_meta->scale_bin);

  free(jbuf);
  return IE_IO_OK;
}

/**
 * @brief Convert an IEEE-754 binary16 (FP16) value to binary32 (float).
 *
 * @details
 * This is a compact conversion used when scales are stored as FP16.
 *
 * @param h Half-precision bits.
 * @return Converted float.
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
 * @brief Decode INT4 packed weights into a float matrix.
 *
 * @details
 * This helper is intended for diagnostics and tooling. It reads:
 *  - packed int4 bytes from @p packed_path
 *  - per-row or per-tensor scale factors from @p scales_path (FP16 or FP32)
 *
 * It then uses ie_quant_int4 helpers to dequantize into @p dst.
 *
 * @param packed_path Path to packed int4 weight data.
 * @param scales_path Path to scale data (FP16 or FP32).
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param per_row Non-zero for per-row scales; 0 for per-tensor scale.
 * @param dst Output float buffer (rows*cols).
 * @return IE_IO_OK on success; negative ie_io_status_t on failure.
 */
int ie_weights_decode_int4(const char *packed_path, const char *scales_path, size_t rows, size_t cols,
                           int per_row, float *dst) {
  if (!packed_path || !scales_path || !dst || rows == 0 || cols == 0) return IE_IO_ERR_ARGS;

  WLOGI("ie_weights_decode_int4: begin packed='%s' scales='%s' rows=%zu cols=%zu per_row=%d",
        packed_path, scales_path, rows, cols, per_row);

  int fdw = open(packed_path, O_RDONLY);
  if (fdw < 0) {
    wlog_errno_("open", packed_path);
    return IE_IO_ERR_OPEN;
  }

  struct stat stw;
  if (fstat(fdw, &stw) != 0) {
    wlog_errno_("fstat", packed_path);
    close(fdw);
    return IE_IO_ERR_STAT;
  }

  const size_t need_packed = rows * ie_int4_rowbytes(cols);
  if ((size_t)stw.st_size < need_packed) {
    WLOGE("ie_weights_decode_int4: packed file too small (have=%zu need=%zu) path='%s'",
          (size_t)stw.st_size, need_packed, packed_path);
    close(fdw);
    return IE_IO_ERR_READ;
  }

  uint8_t *buf_packed = (uint8_t *)malloc(need_packed);
  if (!buf_packed) {
    WLOGE("ie_weights_decode_int4: malloc failed (bytes=%zu) for packed", need_packed);
    close(fdw);
    return IE_IO_ERR_ALLOC;
  }

  ssize_t rd = ie_pread(fdw, buf_packed, need_packed, 0);
  close(fdw);
  if (rd < 0 || (size_t)rd != need_packed) {
    WLOGE("ie_weights_decode_int4: packed read failed (got=%zd want=%zu) path='%s'", rd, need_packed,
          packed_path);
    free(buf_packed);
    return IE_IO_ERR_READ;
  }

  int fds = open(scales_path, O_RDONLY);
  if (fds < 0) {
    wlog_errno_("open", scales_path);
    free(buf_packed);
    return IE_IO_ERR_OPEN;
  }

  struct stat sts;
  if (fstat(fds, &sts) != 0) {
    wlog_errno_("fstat", scales_path);
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
    WLOGE("ie_weights_decode_int4: unexpected scales size (have=%zu need16=%zu need32=%zu) path='%s'",
          (size_t)sts.st_size, need16, need32, scales_path);
    close(fds);
    free(buf_packed);
    return IE_IO_ERR_READ;
  }

  float *buf_scales = (float *)malloc(need32);
  if (!buf_scales) {
    WLOGE("ie_weights_decode_int4: malloc failed (bytes=%zu) for scales", need32);
    close(fds);
    free(buf_packed);
    return IE_IO_ERR_ALLOC;
  }

  if (is32) {
    ssize_t rds = ie_pread(fds, buf_scales, need32, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need32) {
      WLOGE("ie_weights_decode_int4: scales read failed (got=%zd want=%zu) path='%s'", rds, need32,
            scales_path);
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_READ;
    }
    WLOGD("ie_weights_decode_int4: loaded scales as FP32 (count=%zu)", nsc);
  } else {
    uint16_t *tmp = (uint16_t *)malloc(need16);
    if (!tmp) {
      WLOGE("ie_weights_decode_int4: malloc failed (bytes=%zu) for FP16 scales", need16);
      close(fds);
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_ALLOC;
    }
    ssize_t rds = ie_pread(fds, tmp, need16, 0);
    close(fds);
    if (rds < 0 || (size_t)rds != need16) {
      WLOGE("ie_weights_decode_int4: scales read failed (got=%zd want=%zu) path='%s'", rds, need16,
            scales_path);
      free(tmp);
      free(buf_scales);
      free(buf_packed);
      return IE_IO_ERR_READ;
    }
    for (size_t i = 0; i < nsc; ++i) buf_scales[i] = fp16_to_fp32(tmp[i]);
    free(tmp);
    WLOGD("ie_weights_decode_int4: loaded scales as FP16->FP32 (count=%zu)", nsc);
  }

  int qst = 0;
  if (per_row) {
    qst = ie_int4_dequantize_per_row(buf_packed, rows, cols, buf_scales, dst);
  } else {
    qst = ie_int4_dequantize_per_tensor(buf_packed, rows, cols, buf_scales[0], dst);
  }

  free(buf_scales);
  free(buf_packed);

  if (qst != IE_INT4_STATUS_OK) {
    WLOGE("ie_weights_decode_int4: dequantize failed status=%d", qst);
    return IE_IO_ERR_DECODE;
  }

  WLOGI("ie_weights_decode_int4: success");
  return IE_IO_OK;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
