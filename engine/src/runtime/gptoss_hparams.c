/* ============================================================================
 * File: engine/src/runtime/gptoss_hparams.c
 * ============================================================================
 */
/**
 * @file gptoss_hparams.c
 * @brief HuggingFace `config.json` loader for GPT-OSS hyperparameters.
 *
 * @details
 * This file implements a small, relaxed JSON scanner to extract the subset of
 * `config.json` fields required for real inference.
 *
 * The parser:
 *  - reads the entire file into memory (NUL-terminated),
 *  - finds `"key"` occurrences and parses primitive values after `:`,
 *  - does not build a full JSON AST (no third-party dependencies).
 *
 * The goal is robustness and clarity, not strict JSON compliance.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gptoss_hparams.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ============================================================================
 * Small path and file helpers
 * ========================================================================== */

/**
 * @brief Safely join a directory and filename into an output buffer.
 *
 * @param out   Output buffer.
 * @param outsz Output buffer size in bytes.
 * @param dir   Directory path (may be NULL or empty).
 * @param file  File name (must be non-NULL and non-empty).
 * @return IE_IO_OK on success, IE_IO_ERR_ARGS on invalid arguments, or
 *         IE_IO_ERR_DECODE if the resulting path would be truncated.
 */
static int gptoss_join_path(char *out, size_t outsz, const char *dir, const char *file) {
  if (!out || outsz == 0 || !file || !*file) return IE_IO_ERR_ARGS;
  out[0] = '\0';

  if (!dir || !*dir) {
    const int n = snprintf(out, outsz, "%s", file);
    if (n < 0 || (size_t)n >= outsz) return IE_IO_ERR_DECODE;
    return IE_IO_OK;
  }

  const size_t ld = strlen(dir);
  const int need_slash = (ld > 0 && dir[ld - 1] != '/');

  const int n = snprintf(out, outsz, need_slash ? "%s/%s" : "%s%s", dir, file);
  if (n < 0 || (size_t)n >= outsz) return IE_IO_ERR_DECODE;
  return IE_IO_OK;
}

/**
 * @brief Check whether a regular file exists at @p path.
 *
 * @param path File path.
 * @return 1 if the file exists and is a regular file, 0 otherwise.
 */
static int gptoss_file_exists_regular(const char *path) {
  if (!path || !*path) return 0;
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) ? 1 : 0;
}

/**
 * @brief Read a whole text file into a newly allocated NUL-terminated buffer.
 *
 * @details
 * If the file begins with a UTF-8 BOM (0xEF 0xBB 0xBF), it is stripped.
 *
 * @param path    File path.
 * @param out_buf Output pointer receiving a malloc'd buffer (caller frees).
 * @param out_len Optional output receiving length in bytes excluding NUL.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int gptoss_read_all_text(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf) return IE_IO_ERR_ARGS;
  *out_buf = NULL;
  if (out_len) *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) return IE_IO_ERR_OPEN;

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return IE_IO_ERR_READ;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return IE_IO_ERR_READ;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return IE_IO_ERR_READ;
  }

  size_t size = (size_t)n;
  char *buf = (char *)malloc(size + 1u);
  if (!buf) {
    fclose(f);
    return IE_IO_ERR_ALLOC;
  }

  const size_t rd = fread(buf, 1, size, f);
  fclose(f);
  if (rd != size) {
    free(buf);
    return IE_IO_ERR_READ;
  }

  /* Strip UTF-8 BOM if present. */
  if (size >= 3u &&
      (unsigned char)buf[0] == 0xEFu &&
      (unsigned char)buf[1] == 0xBBu &&
      (unsigned char)buf[2] == 0xBFu) {
    memmove(buf, buf + 3, size - 3u);
    size -= 3u;
  }

  buf[size] = '\0';
  *out_buf = buf;
  if (out_len) *out_len = size;
  return IE_IO_OK;
}

/* ============================================================================
 * Relaxed JSON scanning helpers
 * ========================================================================== */

/**
 * @brief Skip ASCII whitespace characters.
 *
 * @param p Input pointer (may be NULL).
 * @return Pointer advanced past whitespace, or NULL if @p p is NULL.
 */
static const char *gptoss_skip_ws(const char *p) {
  if (!p) return NULL;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  return p;
}

/**
 * @brief Find the start of a JSON value corresponding to a top-level string key.
 *
 * @details
 * This scans for an occurrence of `"key"` and then returns a pointer to the
 * first non-whitespace character after the colon.
 *
 * The scan is relaxed and may match nested objects as well. This is acceptable
 * for HuggingFace `config.json` in practice because keys are typically unique.
 *
 * @param json NUL-terminated JSON text.
 * @param key  Key name without quotes.
 * @return Pointer to the value start, or NULL if not found.
 */
static const char *gptoss_find_value_start(const char *json, const char *key) {
  if (!json || !key || !*key) return NULL;

  const size_t klen = strlen(key);
  const char *p = json;

  for (;;) {
    const char *q = strstr(p, "\"");
    if (!q) break;
    q++; /* move past the quote */
    if (strncmp(q, key, klen) == 0 && q[klen] == '"') {
      const char *c = q + klen + 1;
      c = gptoss_skip_ws(c);
      if (!c || *c != ':') {
        p = q;
        continue;
      }
      c++;
      c = gptoss_skip_ws(c);
      return c;
    }
    p = q;
  }
  return NULL;
}

/**
 * @brief Scan an unsigned 32-bit integer JSON value at @p key.
 *
 * @param json     NUL-terminated JSON text.
 * @param key      Key name without quotes.
 * @param out_val  Output value.
 * @return 1 if found and parsed, 0 if not found, negative on invalid args.
 */
static int gptoss_scan_u32(const char *json, const char *key, uint32_t *out_val) {
  if (!json || !key || !out_val) return -1;
  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  errno = 0;
  char *end = NULL;
  unsigned long x = strtoul(v, &end, 10);
  if (end == v || errno != 0) return 0;
  if (x > 0xFFFFFFFFul) return 0;

  *out_val = (uint32_t)x;
  return 1;
}

/**
 * @brief Scan a floating-point JSON value at @p key.
 *
 * @param json     NUL-terminated JSON text.
 * @param key      Key name without quotes.
 * @param out_val  Output value.
 * @return 1 if found and parsed, 0 if not found, negative on invalid args.
 */
static int gptoss_scan_f32(const char *json, const char *key, float *out_val) {
  if (!json || !key || !out_val) return -1;
  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  errno = 0;
  char *end = NULL;
  double x = strtod(v, &end);
  if (end == v || errno != 0) return 0;

  *out_val = (float)x;
  return 1;
}

/**
 * @brief Scan a boolean JSON value at @p key.
 *
 * @param json     NUL-terminated JSON text.
 * @param key      Key name without quotes.
 * @param out_val  Output value (0 or 1).
 * @return 1 if found and parsed, 0 if not found, negative on invalid args.
 */
static int gptoss_scan_bool(const char *json, const char *key, int *out_val) {
  if (!json || !key || !out_val) return -1;
  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  if (strncmp(v, "true", 4) == 0) {
    *out_val = 1;
    return 1;
  }
  if (strncmp(v, "false", 5) == 0) {
    *out_val = 0;
    return 1;
  }
  return 0;
}

/**
 * @brief Scan a JSON string value at @p key into a bounded output buffer.
 *
 * @details
 * This performs minimal unescaping for `\"` and `\\`. Other escape sequences
 * are copied as-is (best-effort for config files).
 *
 * @param json     NUL-terminated JSON text.
 * @param key      Key name without quotes.
 * @param out      Output buffer.
 * @param outsz    Output buffer capacity in bytes.
 * @return 1 if found and copied, 0 if not found, negative on invalid args.
 */
static int gptoss_scan_string(const char *json, const char *key, char *out, size_t outsz) {
  if (!json || !key || !out || outsz == 0) return -1;
  out[0] = '\0';

  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  v = gptoss_skip_ws(v);
  if (!v || *v != '"') return 0;
  v++; /* start of string content */

  size_t di = 0;
  for (const char *p = v; *p; ++p) {
    if (*p == '"') break;
    if (*p == '\\') {
      const char n = *(p + 1);
      if (n == '"' || n == '\\' || n == '/') {
        if (di + 1 < outsz) out[di++] = n;
        p++;
        continue;
      }
      /* Best-effort: keep unknown escapes as the escaped char. */
      if (n != '\0') {
        if (di + 1 < outsz) out[di++] = n;
        p++;
        continue;
      }
    }
    if (di + 1 < outsz) out[di++] = *p;
  }

  out[di] = '\0';
  return 1;
}

/**
 * @brief Extract the JSON object text for a given key (value must be `{...}`).
 *
 * @details
 * This finds `"key": { ... }` and returns a pointer/length to the object body,
 * including the outer braces. Strings are handled so braces inside strings do
 * not affect nesting.
 *
 * @param json     NUL-terminated JSON text.
 * @param key      Key name without quotes.
 * @param out_obj  Output pointer receiving the object start (`{`).
 * @param out_len  Output length in bytes of the object region.
 * @return 1 if found, 0 if not found, negative on invalid args.
 */
static int gptoss_scan_object_region(const char *json,
                                     const char *key,
                                     const char **out_obj,
                                     size_t *out_len) {
  if (!json || !key || !out_obj || !out_len) return -1;
  *out_obj = NULL;
  *out_len = 0;

  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  v = gptoss_skip_ws(v);
  if (!v || *v != '{') return 0;

  const char *start = v;
  int depth = 0;
  int in_str = 0;
  int esc = 0;

  for (const char *p = v; *p; ++p) {
    const char ch = *p;

    if (in_str) {
      if (esc) {
        esc = 0;
        continue;
      }
      if (ch == '\\') {
        esc = 1;
        continue;
      }
      if (ch == '"') {
        in_str = 0;
        continue;
      }
      continue;
    }

    if (ch == '"') {
      in_str = 1;
      continue;
    }
    if (ch == '{') {
      depth++;
      continue;
    }
    if (ch == '}') {
      depth--;
      if (depth == 0) {
        *out_obj = start;
        *out_len = (size_t)((p + 1) - start);
        return 1;
      }
      continue;
    }
  }

  return 0;
}

/* ============================================================================
 * Parsing and validation
 * ========================================================================== */

/**
 * @brief Initialize an extended hyperparameter struct with conservative defaults.
 *
 * @param out_ex Output struct to initialize.
 */
static void gptoss_hparams_ex_defaults(gptoss_hparams_ex_t *out_ex) {
  memset(out_ex, 0, sizeof(*out_ex));
  out_ex->rms_norm_eps = 1.0e-5f;
  out_ex->rope_theta = 10000.0f;
  out_ex->rope_scaling_type[0] = '\0';
  out_ex->rope_scaling_factor = 1.0f;
  out_ex->tie_word_embeddings = 0;
}

/**
 * @brief Parse required and optional fields from a config.json buffer.
 *
 * @param json   NUL-terminated JSON text.
 * @param out_ex Output extended hyperparameters (must be initialized by caller).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int gptoss_parse_config_json(const char *json, gptoss_hparams_ex_t *out_ex) {
  if (!json || !out_ex) return IE_IO_ERR_ARGS;

  /* Required fields. */
  uint32_t n_layers = 0, n_heads = 0, d_model = 0, d_ff = 0, vocab = 0, max_seq = 0;

  if (gptoss_scan_u32(json, "num_hidden_layers", &n_layers) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "num_attention_heads", &n_heads) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "hidden_size", &d_model) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "intermediate_size", &d_ff) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "vocab_size", &vocab) != 1) return IE_IO_ERR_JSON;

  /* max_position_embeddings has several common aliases; prefer the canonical. */
  if (gptoss_scan_u32(json, "max_position_embeddings", &max_seq) != 1) {
    if (gptoss_scan_u32(json, "max_seq_len", &max_seq) != 1) {
      if (gptoss_scan_u32(json, "seq_length", &max_seq) != 1) {
        return IE_IO_ERR_JSON;
      }
    }
  }

  uint32_t n_kv_heads = 0;
  if (gptoss_scan_u32(json, "num_key_value_heads", &n_kv_heads) != 1) {
    n_kv_heads = n_heads;
  }

  if (n_layers == 0 || n_heads == 0 || n_kv_heads == 0 || d_model == 0 || d_ff == 0 ||
      vocab == 0 || max_seq == 0) {
    return IE_IO_ERR_JSON;
  }

  if ((d_model % n_heads) != 0) return IE_IO_ERR_DECODE;
  const uint32_t d_head = d_model / n_heads;

  out_ex->core.n_layers = n_layers;
  out_ex->core.n_heads = n_heads;
  out_ex->core.n_kv_heads = n_kv_heads;
  out_ex->core.d_model = d_model;
  out_ex->core.d_head = d_head;
  out_ex->core.d_ff = d_ff;
  out_ex->core.vocab_size = vocab;
  out_ex->core.max_seq = max_seq;

  /* Optional fields with defaults. */
  (void)gptoss_scan_bool(json, "tie_word_embeddings", &out_ex->tie_word_embeddings);

  /* RMSNorm epsilon aliases. */
  {
    float eps = 0.0f;
    if (gptoss_scan_f32(json, "rms_norm_eps", &eps) == 1) out_ex->rms_norm_eps = eps;
    else if (gptoss_scan_f32(json, "rms_epsilon", &eps) == 1) out_ex->rms_norm_eps = eps;
    else if (gptoss_scan_f32(json, "layer_norm_eps", &eps) == 1) out_ex->rms_norm_eps = eps;
  }

  {
    float theta = 0.0f;
    if (gptoss_scan_f32(json, "rope_theta", &theta) == 1) out_ex->rope_theta = theta;
  }

  /* Optional rope_scaling object. */
  {
    const char *obj = NULL;
    size_t obj_len = 0;
    if (gptoss_scan_object_region(json, "rope_scaling", &obj, &obj_len) == 1 && obj && obj_len > 2) {
      /* The object is NUL-terminated JSON overall; scanning within the slice is OK. */
      (void)gptoss_scan_string(obj, "type", out_ex->rope_scaling_type, sizeof(out_ex->rope_scaling_type));
      {
        float factor = 0.0f;
        if (gptoss_scan_f32(obj, "factor", &factor) == 1) out_ex->rope_scaling_factor = factor;
      }
    }
  }

  return IE_IO_OK;
}

/**
 * @brief Resolve the most appropriate `config.json` path for a model directory.
 *
 * @param model_dir Model directory.
 * @param out_path  Output buffer receiving resolved path.
 * @param out_sz    Output buffer capacity in bytes.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int gptoss_resolve_config_path(const char *model_dir, char *out_path, size_t out_sz) {
  if (!model_dir || !*model_dir || !out_path || out_sz == 0) return IE_IO_ERR_ARGS;
  out_path[0] = '\0';

  char p1[1024];
  char p2[1024];

  if (gptoss_join_path(p1, sizeof(p1), model_dir, "config.json") != IE_IO_OK) return IE_IO_ERR_DECODE;
  if (gptoss_file_exists_regular(p1)) {
    const int n = snprintf(out_path, out_sz, "%s", p1);
    if (n < 0 || (size_t)n >= out_sz) return IE_IO_ERR_DECODE;
    return IE_IO_OK;
  }

  if (gptoss_join_path(p2, sizeof(p2), model_dir, "hf/original/config.json") != IE_IO_OK) return IE_IO_ERR_DECODE;
  if (gptoss_file_exists_regular(p2)) {
    const int n = snprintf(out_path, out_sz, "%s", p2);
    if (n < 0 || (size_t)n >= out_sz) return IE_IO_ERR_DECODE;
    return IE_IO_OK;
  }

  return IE_IO_ERR_OPEN;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

int gptoss_hparams_load_ex_from_file(const char *config_json_path, gptoss_hparams_ex_t *out_ex) {
  if (!config_json_path || !out_ex) return IE_IO_ERR_ARGS;

  gptoss_hparams_ex_defaults(out_ex);

  char *buf = NULL;
  size_t len = 0;
  const int rc = gptoss_read_all_text(config_json_path, &buf, &len);
  if (rc != IE_IO_OK) return rc;
  if (!buf || len == 0) {
    free(buf);
    return IE_IO_ERR_READ;
  }

  const int prc = gptoss_parse_config_json(buf, out_ex);
  free(buf);
  return prc;
}

int gptoss_hparams_load_ex(const char *model_dir, gptoss_hparams_ex_t *out_ex) {
  if (!model_dir || !out_ex) return IE_IO_ERR_ARGS;

  char cfg[1024];
  const int r = gptoss_resolve_config_path(model_dir, cfg, sizeof(cfg));
  if (r != IE_IO_OK) return r;

  return gptoss_hparams_load_ex_from_file(cfg, out_ex);
}

int gptoss_hparams_load(const char *model_dir, ie_gptoss_hparams_t *out_hp) {
  if (!model_dir || !out_hp) return IE_IO_ERR_ARGS;

  gptoss_hparams_ex_t ex;
  const int rc = gptoss_hparams_load_ex(model_dir, &ex);
  if (rc != IE_IO_OK) return rc;

  *out_hp = ex.core;
  return IE_IO_OK;
}
