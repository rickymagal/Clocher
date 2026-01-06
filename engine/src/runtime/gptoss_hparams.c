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
 * `config.json` fields required for inference.
 *
 * Additionally, GPT-OSS variants may use a `head_dim` that is NOT equal to
 * `hidden_size / num_attention_heads` (e.g., projection dims differ from d_model).
 * When needed, this loader can infer `d_head` from `model.ie.json` tensor shapes
 * (q_proj / k_proj) as a fallback, while still using `config.json` for core counts.
 *
 * The parser:
 *  - reads files into memory (NUL-terminated),
 *  - finds `"key"` occurrences and parses primitive values after `:`,
 *  - does not build a full JSON AST (no third-party dependencies).
 *
 * Logging:
 *  - INFO logs for path resolution and final hyperparameter summary.
 *  - WARN logs for fallbacks (missing keys, inferred head_dim, alternate paths).
 *  - ERROR logs for parse failures with the exact missing/invalid key.
 *  - Optional verbose logs for deeper diagnostics:
 *      - IE_HPARAMS_VERBOSE=1 : emit extra details about scan/probe decisions.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gptoss_hparams.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "ie_io.h"
#include "util_logging.h"

/* ============================================================================
 * Logging helpers
 * ========================================================================== */

/**
 * @brief Read a boolean environment variable as 0/1.
 *
 * @param name Environment variable name.
 * @param default_value Value used when variable is unset or empty.
 * @return 0 or 1.
 */
static int hp_env_flag_(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "FALSE") == 0 ||
      strcmp(v, "no") == 0 || strcmp(v, "NO") == 0 || strcmp(v, "off") == 0 || strcmp(v, "OFF") == 0) {
    return 0;
  }
  if (strcmp(v, "1") == 0 || strcmp(v, "true") == 0 || strcmp(v, "TRUE") == 0 ||
      strcmp(v, "yes") == 0 || strcmp(v, "YES") == 0 || strcmp(v, "on") == 0 || strcmp(v, "ON") == 0) {
    return 1;
  }
  return 1;
}

/**
 * @brief Log a short hparams summary.
 *
 * @param hp Hyperparameter struct (required).
 */
static void hp_log_summary_(const ie_gptoss_hparams_t *hp) {
  if (!hp) return;
  ie_log_info("hparams: layers=%" PRIu32 " d_model=%" PRIu32 " heads=%" PRIu32 " kv_heads=%" PRIu32
              " head_dim=%" PRIu32 " d_ff=%" PRIu32 " vocab=%" PRIu32 " max_seq=%" PRIu32,
              hp->n_layers,
              hp->d_model,
              hp->n_heads,
              hp->n_kv_heads,
              hp->d_head,
              hp->d_ff,
              hp->vocab_size,
              hp->max_seq);
}

/* ============================================================================
 * Small path and file helpers
 * ========================================================================== */

/**
 * @brief Join @p dir and @p file into @p out, inserting a slash if needed.
 *
 * @param out Output buffer.
 * @param outsz Output capacity in bytes.
 * @param dir Directory path (may be NULL/empty).
 * @param file Filename (must not be NULL/empty).
 * @return IE_IO_OK on success, otherwise an IE_IO_ERR_* code.
 */
static int gptoss_join_path(char *out, size_t outsz, const char *dir, const char *file) {
  if (!out || outsz == 0 || !file || !*file) {
    ie_log_error("gptoss_join_path: bad args (out=%p outsz=%zu dir=%s file=%s)",
                 (void *)out,
                 outsz,
                 dir ? dir : "(null)",
                 file ? file : "(null)");
    return IE_IO_ERR_ARGS;
  }
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
 * @brief Check whether a path exists and is a regular file.
 *
 * @param path Filesystem path.
 * @return 1 if regular file exists, 0 otherwise.
 */
static int gptoss_file_exists_regular(const char *path) {
  if (!path || !*path) return 0;
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) ? 1 : 0;
}

/**
 * @brief Read an entire text file into a NUL-terminated buffer.
 *
 * @details
 * Performs a best-effort UTF-8 BOM strip (0xEF 0xBB 0xBF) at the beginning.
 *
 * @param path Input file path.
 * @param out_buf Output buffer pointer (malloc'd, caller frees).
 * @param out_len Optional output length (excluding NUL).
 * @return IE_IO_OK on success, otherwise an IE_IO_ERR_* code.
 */
static int gptoss_read_all_text(const char *path, char **out_buf, size_t *out_len) {
  if (!path || !out_buf) {
    ie_log_error("gptoss_read_all_text: bad args (path=%p out_buf=%p)", (const void *)path, (void *)out_buf);
    return IE_IO_ERR_ARGS;
  }
  *out_buf = NULL;
  if (out_len) *out_len = 0;

  FILE *f = fopen(path, "rb");
  if (!f) {
    ie_log_error("gptoss_read_all_text: open failed (path=%s errno=%d: %s)", path, errno, strerror(errno));
    return IE_IO_ERR_OPEN;
  }

  if (fseek(f, 0, SEEK_END) != 0) {
    ie_log_error("gptoss_read_all_text: fseek(SEEK_END) failed (path=%s)", path);
    fclose(f);
    return IE_IO_ERR_READ;
  }
  long n = ftell(f);
  if (n < 0) {
    ie_log_error("gptoss_read_all_text: ftell failed (path=%s)", path);
    fclose(f);
    return IE_IO_ERR_READ;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    ie_log_error("gptoss_read_all_text: fseek(SEEK_SET) failed (path=%s)", path);
    fclose(f);
    return IE_IO_ERR_READ;
  }

  size_t size = (size_t)n;
  char *buf = (char *)malloc(size + 1u);
  if (!buf) {
    ie_log_error("gptoss_read_all_text: OOM (path=%s size=%zu)", path, size);
    fclose(f);
    return IE_IO_ERR_ALLOC;
  }

  const size_t rd = fread(buf, 1, size, f);
  fclose(f);
  if (rd != size) {
    ie_log_error("gptoss_read_all_text: fread short read (path=%s got=%zu expected=%zu)", path, rd, size);
    free(buf);
    return IE_IO_ERR_READ;
  }

  if (size >= 3u &&
      (unsigned char)buf[0] == 0xEFu &&
      (unsigned char)buf[1] == 0xBBu &&
      (unsigned char)buf[2] == 0xBFu) {
    memmove(buf, buf + 3, size - 3u);
    size -= 3u;
    ie_log_info("gptoss_read_all_text: stripped UTF-8 BOM (path=%s)", path);
  }

  buf[size] = '\0';
  *out_buf = buf;
  if (out_len) *out_len = size;

  return IE_IO_OK;
}

/* ============================================================================
 * Relaxed JSON scanning helpers (config.json)
 * ========================================================================== */

/**
 * @brief Skip ASCII whitespace in a JSON string.
 *
 * @param p Input pointer.
 * @return Pointer to first non-whitespace char (or NULL if p is NULL).
 */
static const char *gptoss_skip_ws(const char *p) {
  if (!p) return NULL;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  return p;
}

/**
 * @brief Find the start of the JSON value for a given key.
 *
 * @details
 * Looks for occurrences of `"key"` and returns a pointer to the first
 * non-whitespace character after the ':' separator.
 *
 * This is a relaxed scanner: it does not validate full JSON structure.
 *
 * @param json JSON text (NUL-terminated).
 * @param key Key string without quotes.
 * @return Pointer to value start, or NULL if not found.
 */
static const char *gptoss_find_value_start(const char *json, const char *key) {
  if (!json || !key || !*key) return NULL;

  const size_t klen = strlen(key);
  const char *p = json;

  for (;;) {
    const char *q = strstr(p, "\"");
    if (!q) break;
    q++;
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
 * @brief Scan a uint32 field from JSON for a given key.
 *
 * @param json JSON text.
 * @param key Key string.
 * @param out_val Output value.
 * @return 1 if parsed, 0 if not found/invalid, -1 on invalid args.
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
 * @brief Scan a float field from JSON for a given key.
 *
 * @param json JSON text.
 * @param key Key string.
 * @param out_val Output value.
 * @return 1 if parsed, 0 if not found/invalid, -1 on invalid args.
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
 * @brief Scan a boolean field ("true"/"false") from JSON.
 *
 * @param json JSON text.
 * @param key Key string.
 * @param out_val Output value (0/1).
 * @return 1 if parsed, 0 if not found/invalid, -1 on invalid args.
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
 * @brief Scan a JSON string field from JSON for a given key.
 *
 * @details
 * Performs minimal escape handling for \" \\ and \/ and passes through other
 * escaped bytes literally (best effort).
 *
 * @param json JSON text.
 * @param key Key string.
 * @param out Output buffer.
 * @param outsz Output capacity.
 * @return 1 if parsed, 0 if not found/invalid, -1 on invalid args.
 */
static int gptoss_scan_string(const char *json, const char *key, char *out, size_t outsz) {
  if (!json || !key || !out || outsz == 0) return -1;
  out[0] = '\0';

  const char *v = gptoss_find_value_start(json, key);
  if (!v) return 0;

  v = gptoss_skip_ws(v);
  if (!v || *v != '"') return 0;
  v++;

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
 * @brief Extract a JSON object region for a given key into a pointer+length.
 *
 * @details
 * Returns the region including braces: "{ ... }". The scan is brace-depth-based
 * and respects quoted strings and escapes enough to avoid counting braces inside
 * string literals.
 *
 * @param json JSON text.
 * @param key Key string.
 * @param out_obj Output pointer to '{' start.
 * @param out_len Output length in bytes (including final '}').
 * @return 1 if found, 0 if missing/invalid, -1 on invalid args.
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
 * Minimal tensor-shape probing (model.ie.json)
 * ========================================================================== */

/**
 * @brief Find the JSON object region that contains a tensor entry with a given name.
 *
 * @details
 * This performs a relaxed scan for `"name"` fields and matches the string value
 * against @p tensor_name. On match, it attempts to rewind to the opening '{'
 * of the containing object (best-effort heuristic).
 *
 * @param json Full model.ie.json text.
 * @param tensor_name Tensor name to match.
 * @return Pointer into @p json at the best-effort object start, or NULL if not found.
 */
static const char *iejson_find_tensor_region(const char *json, const char *tensor_name) {
  if (!json || !tensor_name || !*tensor_name) return NULL;

  const size_t want_n = strlen(tensor_name);
  const char *p = json;

  while ((p = strstr(p, "\"name\"")) != NULL) {
    const char *q = p + 6;

    q = strchr(q, ':');
    if (!q) break;
    q++;
    q = gptoss_skip_ws(q);
    if (!q || *q != '"') {
      p = q ? q : (p + 6);
      continue;
    }
    q++;

    const char *s = q;
    while (*q) {
      if (*q == '\\' && *(q + 1) != '\0') {
        q += 2;
        continue;
      }
      if (*q == '"') break;
      q++;
    }
    if (*q != '"') break;

    const size_t got_n = (size_t)(q - s);
    int match = 0;
    if (got_n == want_n && strncmp(s, tensor_name, want_n) == 0) match = 1;

    if (match) {
      const char *obj = p;
      for (int i = 0; i < 8192 && obj > json; ++i) {
        if (*obj == '{') break;
        obj--;
      }
      if (*obj == '{') return obj;
    }

    p = q + 1;
  }

  return NULL;
}

/**
 * @brief Scan a 2D shape array (two integers) from a tensor object region.
 *
 * @param obj Pointer to a JSON object that includes a `"shape": [a,b,...]` field.
 * @param out0 Output first dimension.
 * @param out1 Output second dimension.
 * @return 1 on success, 0 on failure.
 */
static int iejson_scan_shape2_at(const char *obj, uint32_t *out0, uint32_t *out1) {
  if (!obj || !out0 || !out1) return 0;

  const char *p = strstr(obj, "\"shape\"");
  if (!p) return 0;

  p = strchr(p, '[');
  if (!p) return 0;
  p++;

  p = gptoss_skip_ws(p);
  if (!p || !isdigit((unsigned char)*p)) return 0;

  errno = 0;
  char *end = NULL;
  unsigned long a = strtoul(p, &end, 10);
  if (end == p || errno != 0 || a > 0xFFFFFFFFul) return 0;

  p = end;
  p = gptoss_skip_ws(p);
  if (!p || *p != ',') return 0;
  p++;
  p = gptoss_skip_ws(p);
  if (!p || !isdigit((unsigned char)*p)) return 0;

  errno = 0;
  end = NULL;
  unsigned long b = strtoul(p, &end, 10);
  if (end == p || errno != 0 || b > 0xFFFFFFFFul) return 0;

  *out0 = (uint32_t)a;
  *out1 = (uint32_t)b;
  return 1;
}

/**
 * @brief Infer attention head dimension from model.ie.json tensor shapes.
 *
 * @details
 * Uses q_proj (and optionally k_proj) shapes. Requires q0 % n_heads == 0 and,
 * if k_proj is found, k0 % n_kv_heads == 0 with matching derived head_dim.
 *
 * @param model_dir Model directory containing model.ie.json.
 * @param n_heads Number of attention heads.
 * @param n_kv_heads Number of key/value heads.
 * @param out_head_dim Output inferred head dimension.
 * @return 1 if inferred, 0 otherwise.
 */
static int gptoss_probe_head_dim_from_iejson(const char *model_dir,
                                             uint32_t n_heads,
                                             uint32_t n_kv_heads,
                                             uint32_t *out_head_dim) {
  const int verbose = hp_env_flag_("IE_HPARAMS_VERBOSE", 0);

  if (!model_dir || !*model_dir || !out_head_dim || n_heads == 0 || n_kv_heads == 0) return 0;

  char path[1024];
  if (gptoss_join_path(path, sizeof(path), model_dir, "model.ie.json") != IE_IO_OK) return 0;
  if (!gptoss_file_exists_regular(path)) {
    if (verbose) ie_log_info("hparams: model.ie.json not found for head_dim probe (path=%s)", path);
    return 0;
  }

  char *json = NULL;
  size_t len = 0;
  if (gptoss_read_all_text(path, &json, &len) != IE_IO_OK || !json || len == 0) {
    free(json);
    if (verbose) ie_log_info("hparams: failed to read model.ie.json for head_dim probe (path=%s)", path);
    return 0;
  }

  const char *q_names[] = {
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.attn.q_proj.weight",
      "transformer.h.0.attn.q_proj.weight",
  };
  const char *k_names[] = {
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.attn.k_proj.weight",
      "transformer.h.0.attn.k_proj.weight",
  };

  uint32_t q0 = 0, q1 = 0;
  uint32_t k0 = 0, k1 = 0;

  int got_q = 0;
  for (size_t i = 0; i < sizeof(q_names) / sizeof(q_names[0]); ++i) {
    const char *obj = iejson_find_tensor_region(json, q_names[i]);
    if (obj && iejson_scan_shape2_at(obj, &q0, &q1) == 1) {
      got_q = 1;
      if (verbose) ie_log_info("hparams: found q_proj shape for probe (%s -> [%" PRIu32 ",%" PRIu32 "])",
                               q_names[i], q0, q1);
      break;
    }
  }

  int got_k = 0;
  for (size_t i = 0; i < sizeof(k_names) / sizeof(k_names[0]); ++i) {
    const char *obj = iejson_find_tensor_region(json, k_names[i]);
    if (obj && iejson_scan_shape2_at(obj, &k0, &k1) == 1) {
      got_k = 1;
      if (verbose) ie_log_info("hparams: found k_proj shape for probe (%s -> [%" PRIu32 ",%" PRIu32 "])",
                               k_names[i], k0, k1);
      break;
    }
  }

  free(json);

  if (!got_q) return 0;
  if (q0 == 0 || (q0 % n_heads) != 0) {
    if (verbose) ie_log_warn("hparams: q_proj shape incompatible for head_dim probe (q0=%" PRIu32 " n_heads=%" PRIu32 ")",
                             q0, n_heads);
    return 0;
  }

  const uint32_t head_dim_q = q0 / n_heads;

  if (got_k && k0 != 0) {
    if ((k0 % n_kv_heads) != 0) {
      if (verbose) ie_log_warn("hparams: k_proj shape incompatible for head_dim probe (k0=%" PRIu32 " n_kv_heads=%" PRIu32 ")",
                               k0, n_kv_heads);
      return 0;
    }
    const uint32_t head_dim_k = k0 / n_kv_heads;
    if (head_dim_k != head_dim_q) {
      if (verbose) ie_log_warn("hparams: head_dim mismatch between q/k probe (q=%" PRIu32 " k=%" PRIu32 ")",
                               head_dim_q, head_dim_k);
      return 0;
    }
  }

  *out_head_dim = head_dim_q;
  return 1;
}

/* ============================================================================
 * Parsing and validation
 * ========================================================================== */

/**
 * @brief Initialize extended hparams structure with defaults.
 *
 * @param out_ex Output structure.
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
 * @brief Parse required and optional fields from config.json into hparams.
 *
 * @details
 * Required fields:
 *  - num_hidden_layers
 *  - num_attention_heads
 *  - hidden_size
 *  - intermediate_size
 *  - vocab_size
 *  - max_position_embeddings (or max_seq_len or seq_length)
 *
 * Optional fields:
 *  - num_key_value_heads (defaults to num_attention_heads)
 *  - head_dim (otherwise inferred from model.ie.json or computed from d_model/n_heads)
 *  - tie_word_embeddings
 *  - rms_norm_eps / rms_epsilon / layer_norm_eps
 *  - rope_theta
 *  - rope_scaling: { type, factor }
 *
 * @param json NUL-terminated config.json content.
 * @param model_dir Model directory (used for model.ie.json probe; may be NULL).
 * @param out_ex Output extended hparams structure (must be pre-defaulted).
 * @return IE_IO_OK on success, otherwise an IE_IO_ERR_* code.
 */
static int gptoss_parse_config_json(const char *json,
                                   const char *model_dir,
                                   gptoss_hparams_ex_t *out_ex) {
  const int verbose = hp_env_flag_("IE_HPARAMS_VERBOSE", 0);

  if (!json || !out_ex) {
    ie_log_error("hparams: parse_config_json bad args (json=%p out_ex=%p)", (const void *)json, (void *)out_ex);
    return IE_IO_ERR_ARGS;
  }

  uint32_t n_layers = 0, n_heads = 0, d_model = 0, d_ff = 0, vocab = 0, max_seq = 0;

  if (gptoss_scan_u32(json, "num_hidden_layers", &n_layers) != 1) {
    ie_log_error("hparams: missing/invalid required key \"num_hidden_layers\"");
    return IE_IO_ERR_JSON;
  }
  if (gptoss_scan_u32(json, "num_attention_heads", &n_heads) != 1) {
    ie_log_error("hparams: missing/invalid required key \"num_attention_heads\"");
    return IE_IO_ERR_JSON;
  }
  if (gptoss_scan_u32(json, "hidden_size", &d_model) != 1) {
    ie_log_error("hparams: missing/invalid required key \"hidden_size\"");
    return IE_IO_ERR_JSON;
  }
  if (gptoss_scan_u32(json, "intermediate_size", &d_ff) != 1) {
    ie_log_error("hparams: missing/invalid required key \"intermediate_size\"");
    return IE_IO_ERR_JSON;
  }
  if (gptoss_scan_u32(json, "vocab_size", &vocab) != 1) {
    ie_log_error("hparams: missing/invalid required key \"vocab_size\"");
    return IE_IO_ERR_JSON;
  }

  if (gptoss_scan_u32(json, "max_position_embeddings", &max_seq) != 1) {
    if (gptoss_scan_u32(json, "max_seq_len", &max_seq) != 1) {
      if (gptoss_scan_u32(json, "seq_length", &max_seq) != 1) {
        ie_log_error("hparams: missing required sequence length key (max_position_embeddings/max_seq_len/seq_length)");
        return IE_IO_ERR_JSON;
      }
    }
  }

  uint32_t n_kv_heads = 0;
  if (gptoss_scan_u32(json, "num_key_value_heads", &n_kv_heads) != 1) {
    n_kv_heads = n_heads;
    if (verbose) ie_log_info("hparams: num_key_value_heads missing; defaulting to num_attention_heads=%" PRIu32, n_heads);
  }

  if (n_layers == 0 || n_heads == 0 || n_kv_heads == 0 || d_model == 0 || d_ff == 0 || vocab == 0 || max_seq == 0) {
    ie_log_error("hparams: invalid zero field after parse (layers=%" PRIu32 " heads=%" PRIu32 " kv_heads=%" PRIu32
                 " d_model=%" PRIu32 " d_ff=%" PRIu32 " vocab=%" PRIu32 " max_seq=%" PRIu32 ")",
                 n_layers, n_heads, n_kv_heads, d_model, d_ff, vocab, max_seq);
    return IE_IO_ERR_JSON;
  }
  if (n_kv_heads > n_heads) {
    ie_log_error("hparams: invalid kv_heads > heads (kv_heads=%" PRIu32 " heads=%" PRIu32 ")", n_kv_heads, n_heads);
    return IE_IO_ERR_DECODE;
  }
  if ((n_heads % n_kv_heads) != 0) {
    ie_log_error("hparams: invalid heads %% kv_heads != 0 (heads=%" PRIu32 " kv_heads=%" PRIu32 ")", n_heads, n_kv_heads);
    return IE_IO_ERR_DECODE;
  }

  uint32_t d_head = 0;

  /* Prefer explicit head_dim if present. */
  if (gptoss_scan_u32(json, "head_dim", &d_head) == 1) {
    if (verbose) ie_log_info("hparams: using explicit head_dim=%" PRIu32 " from config.json", d_head);
  } else {
    d_head = 0;
  }

  /* Otherwise, infer from model.ie.json tensor shapes when possible. */
  if (d_head == 0 && model_dir && *model_dir) {
    uint32_t inferred = 0;
    if (gptoss_probe_head_dim_from_iejson(model_dir, n_heads, n_kv_heads, &inferred) == 1 && inferred != 0) {
      d_head = inferred;
      ie_log_warn("hparams: inferred head_dim=%" PRIu32 " from model.ie.json (q_proj/k_proj shapes)", d_head);
    } else if (verbose) {
      ie_log_info("hparams: head_dim probe from model.ie.json did not succeed");
    }
  }

  /* Fallback: old behavior (may be wrong for GPT-OSS variants). */
  if (d_head == 0) {
    if ((d_model % n_heads) != 0) {
      ie_log_error("hparams: cannot derive head_dim from d_model/heads (d_model=%" PRIu32 " heads=%" PRIu32 ")", d_model, n_heads);
      return IE_IO_ERR_DECODE;
    }
    d_head = d_model / n_heads;
    ie_log_warn("hparams: head_dim missing; using d_model/heads fallback (%" PRIu32 "/%" PRIu32 "=%" PRIu32 ")",
                d_model, n_heads, d_head);
  }

  out_ex->core.n_layers = n_layers;
  out_ex->core.n_heads = n_heads;
  out_ex->core.n_kv_heads = n_kv_heads;
  out_ex->core.d_model = d_model;
  out_ex->core.d_head = d_head;
  out_ex->core.d_ff = d_ff;
  out_ex->core.vocab_size = vocab;
  out_ex->core.max_seq = max_seq;

  (void)gptoss_scan_bool(json, "tie_word_embeddings", &out_ex->tie_word_embeddings);

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

  {
    const char *obj = NULL;
    size_t obj_len = 0;
    if (gptoss_scan_object_region(json, "rope_scaling", &obj, &obj_len) == 1 && obj && obj_len > 2) {
      (void)gptoss_scan_string(obj, "type", out_ex->rope_scaling_type, sizeof(out_ex->rope_scaling_type));
      {
        float factor = 0.0f;
        if (gptoss_scan_f32(obj, "factor", &factor) == 1) out_ex->rope_scaling_factor = factor;
      }
      if (verbose) {
        ie_log_info("hparams: rope_scaling type=\"%s\" factor=%g",
                    out_ex->rope_scaling_type[0] ? out_ex->rope_scaling_type : "(empty)",
                    (double)out_ex->rope_scaling_factor);
      }
    }
  }

  if (verbose) {
    ie_log_info("hparams: extras eps=%g rope_theta=%g tie_word_embeddings=%d",
                (double)out_ex->rms_norm_eps,
                (double)out_ex->rope_theta,
                out_ex->tie_word_embeddings);
  }

  return IE_IO_OK;
}

/**
 * @brief Resolve config.json path within a model directory.
 *
 * @details
 * Tries:
 *  1) <model_dir>/config.json
 *  2) <model_dir>/hf/original/config.json
 *
 * @param model_dir Model directory.
 * @param out_path Output path buffer.
 * @param out_sz Output capacity.
 * @return IE_IO_OK on success, otherwise IE_IO_ERR_* code.
 */
static int gptoss_resolve_config_path(const char *model_dir, char *out_path, size_t out_sz) {
  if (!model_dir || !*model_dir || !out_path || out_sz == 0) {
    ie_log_error("hparams: resolve_config_path bad args (model_dir=%p out_path=%p out_sz=%zu)",
                 (const void *)model_dir, (void *)out_path, out_sz);
    return IE_IO_ERR_ARGS;
  }
  out_path[0] = '\0';

  char p1[1024];
  char p2[1024];

  if (gptoss_join_path(p1, sizeof(p1), model_dir, "config.json") != IE_IO_OK) return IE_IO_ERR_DECODE;
  if (gptoss_file_exists_regular(p1)) {
    const int n = snprintf(out_path, out_sz, "%s", p1);
    if (n < 0 || (size_t)n >= out_sz) return IE_IO_ERR_DECODE;
    ie_log_info("hparams: using config path %s", out_path);
    return IE_IO_OK;
  }

  if (gptoss_join_path(p2, sizeof(p2), model_dir, "hf/original/config.json") != IE_IO_OK) return IE_IO_ERR_DECODE;
  if (gptoss_file_exists_regular(p2)) {
    const int n = snprintf(out_path, out_sz, "%s", p2);
    if (n < 0 || (size_t)n >= out_sz) return IE_IO_ERR_DECODE;
    ie_log_warn("hparams: using fallback config path %s", out_path);
    return IE_IO_OK;
  }

  ie_log_error("hparams: config.json not found (tried %s and %s)", p1, p2);
  return IE_IO_ERR_OPEN;
}

/* ============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Load extended hyperparameters from an explicit config.json path.
 *
 * @details
 * Note: In this entrypoint, model_dir is unknown, so model.ie.json head_dim probing
 * is not available. If config.json does not contain head_dim, the loader may fall
 * back to d_model/heads.
 *
 * @param config_json_path Path to config.json.
 * @param out_ex Output extended hyperparameters.
 * @return IE_IO_OK on success, otherwise IE_IO_ERR_* code.
 */
int gptoss_hparams_load_ex_from_file(const char *config_json_path, gptoss_hparams_ex_t *out_ex) {
  if (!config_json_path || !out_ex) {
    ie_log_error("hparams: load_ex_from_file bad args (path=%p out_ex=%p)",
                 (const void *)config_json_path, (void *)out_ex);
    return IE_IO_ERR_ARGS;
  }

  gptoss_hparams_ex_defaults(out_ex);

  ie_log_info("hparams: load_ex_from_file (path=%s)", config_json_path);

  char *buf = NULL;
  size_t len = 0;
  const int rc = gptoss_read_all_text(config_json_path, &buf, &len);
  if (rc != IE_IO_OK) {
    ie_log_error("hparams: read config failed (rc=%d path=%s)", rc, config_json_path);
    return rc;
  }
  if (!buf || len == 0) {
    ie_log_error("hparams: empty config file (path=%s)", config_json_path);
    free(buf);
    return IE_IO_ERR_READ;
  }

  const int prc = gptoss_parse_config_json(buf, NULL, out_ex);
  free(buf);

  if (prc != IE_IO_OK) {
    ie_log_error("hparams: parse failed (rc=%d path=%s)", prc, config_json_path);
    return prc;
  }

  hp_log_summary_(&out_ex->core);
  return IE_IO_OK;
}

/**
 * @brief Load extended hyperparameters from a model directory.
 *
 * @param model_dir Model directory.
 * @param out_ex Output extended hyperparameters.
 * @return IE_IO_OK on success, otherwise IE_IO_ERR_* code.
 */
int gptoss_hparams_load_ex(const char *model_dir, gptoss_hparams_ex_t *out_ex) {
  if (!model_dir || !out_ex) {
    ie_log_error("hparams: load_ex bad args (model_dir=%p out_ex=%p)",
                 (const void *)model_dir, (void *)out_ex);
    return IE_IO_ERR_ARGS;
  }

  gptoss_hparams_ex_defaults(out_ex);

  char cfg[1024];
  const int r = gptoss_resolve_config_path(model_dir, cfg, sizeof(cfg));
  if (r != IE_IO_OK) {
    ie_log_error("hparams: resolve config path failed (rc=%d model_dir=%s)", r, model_dir);
    return r;
  }

  char *buf = NULL;
  size_t len = 0;
  const int rc = gptoss_read_all_text(cfg, &buf, &len);
  if (rc != IE_IO_OK) {
    ie_log_error("hparams: read config failed (rc=%d path=%s)", rc, cfg);
    return rc;
  }
  if (!buf || len == 0) {
    ie_log_error("hparams: empty config file (path=%s)", cfg);
    free(buf);
    return IE_IO_ERR_READ;
  }

  ie_log_info("hparams: parsing config (path=%s)", cfg);

  const int prc = gptoss_parse_config_json(buf, model_dir, out_ex);
  free(buf);

  if (prc != IE_IO_OK) {
    ie_log_error("hparams: parse failed (rc=%d path=%s)", prc, cfg);
    return prc;
  }

  hp_log_summary_(&out_ex->core);
  return IE_IO_OK;
}

/**
 * @brief Load core GPT-OSS hyperparameters from a model directory.
 *
 * @details
 * This is a convenience wrapper that loads the extended structure and returns
 * only the core fields required by the runtime.
 *
 * @param model_dir Model directory.
 * @param out_hp Output core hyperparameters.
 * @return IE_IO_OK on success, otherwise an IE_IO_ERR_* code.
 */
int gptoss_hparams_load(const char *model_dir, ie_gptoss_hparams_t *out_hp) {
  if (!model_dir || !out_hp) {
    ie_log_error("hparams: load bad args (model_dir=%p out_hp=%p)", (const void *)model_dir, (void *)out_hp);
    return IE_IO_ERR_ARGS;
  }

  gptoss_hparams_ex_t ex;
  const int rc = gptoss_hparams_load_ex(model_dir, &ex);
  if (rc != IE_IO_OK) {
    ie_log_error("hparams: load_ex failed (rc=%d model_dir=%s)", rc, model_dir);
    return rc;
  }

  *out_hp = ex.core;
  return IE_IO_OK;
}
