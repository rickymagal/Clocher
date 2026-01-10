/* ============================================================================
 * File: engine/src/runtime/gptoss_hparams.c
 * ============================================================================
 */
/**
 * @file gptoss_hparams.c
 * @brief HuggingFace config.json loader for GPT-OSS hyperparameters (with IEJSON shape inference).
 *
 * @details
 * This loader is intentionally small and dependency-free (no JSON library).
 *
 * Why this file exists:
 *  - HF config.json for this model reports hidden_size=2880 and num_attention_heads=64,
 *    which implies d_head=45 if you do d_model/n_heads.
 *  - But model.ie.json shows q_proj.weight shape=[4096, 2880] and k_proj.weight shape=[512, 2880],
 *    which implies d_head=64 with n_heads=64 and n_kv_heads=8.
 *
 * So we:
 *  - read required fields from config.json (layers, heads, vocab, max_position_embeddings, etc),
 *  - then read model.ie.json to infer:
 *      - q_dim, kv_dim, d_model (from q_proj),
 *      - d_ff (from MoE down_proj blocks),
 *      - and set d_head = q_dim / n_heads (not d_model / n_heads).
 *
 * Also important:
 *  - config.json max_position_embeddings may be huge (e.g. 131072). Allocating KV for that
 *    will often fail on a dev machine. We clamp max_seq by default.
 *    Override with IE_MAX_SEQ environment variable.
 *
 * RoPE scaling integration:
 *  - We parse rope_scaling.{type,factor,original_max_position_embeddings} into gptoss_hparams_ex_t.
 *  - If the model config uses "linear" RoPE scaling AND the user did not set
 *    IE_ROPE_LINEAR_SCALE / IE_ROPE_POS_SCALE, we export the scale to the RoPE helper
 *    by calling setenv("IE_ROPE_LINEAR_SCALE", "<factor>", 0).
 *  - This keeps scaling "inside the RoPE helper" without changing inference call sites.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gptoss_hparams.h"
#include "ie_io.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ============================================================================
 * Small path and file helpers
 * ========================================================================== */

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

static int gptoss_file_exists_regular(const char *path) {
  if (!path || !*path) return 0;
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return S_ISREG(st.st_mode) ? 1 : 0;
}

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
 * Relaxed JSON scanning helpers (config.json)
 * ========================================================================== */

static const char *gptoss_skip_ws(const char *p) {
  if (!p) return NULL;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
  return p;
}

static const char *gptoss_find_value_start(const char *json, const char *key) {
  if (!json || !key || !*key) return NULL;

  const size_t klen = strlen(key);
  const char *p = json;

  for (;;) {
    const char *q = strstr(p, "\"");
    if (!q) break;
    q++; /* past quote */
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
 * IEJSON scanning helpers (model.ie.json)
 * ========================================================================== */

static const char *iejson_find_tensor_region(const char *json, const char *tensor_name) {
  if (!json || !tensor_name || !*tensor_name) return NULL;

  const size_t nlen = strlen(tensor_name);

  size_t need1 = strlen("\"name\":\"\"") + nlen;
  char *needle1 = (char *)malloc(need1 + 1u);
  if (!needle1) return NULL;
  snprintf(needle1, need1 + 1u, "\"name\":\"%s\"", tensor_name);

  const char *p = strstr(json, needle1);
  free(needle1);
  if (p) return p;

  size_t need2 = strlen("\"name\": \"\"") + nlen;
  char *needle2 = (char *)malloc(need2 + 1u);
  if (!needle2) return NULL;
  snprintf(needle2, need2 + 1u, "\"name\": \"%s\"", tensor_name);

  p = strstr(json, needle2);
  free(needle2);
  return p;
}

static int iejson_parse_shape_dims(const char *p, uint32_t *dims, size_t max_dims, size_t *out_ndims) {
  if (!p || !dims || max_dims == 0 || !out_ndims) return 0;
  *out_ndims = 0;

  const char *s = strstr(p, "\"shape\"");
  if (!s) return 0;
  s = strchr(s, '[');
  if (!s) return 0;
  s++;

  size_t nd = 0;
  for (;;) {
    s = gptoss_skip_ws(s);
    if (!s || !*s) return 0;
    if (*s == ']') break;
    if (!isdigit((unsigned char)*s)) return 0;

    errno = 0;
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (end == s || errno != 0) return 0;
    if (v > 0xFFFFFFFFul) return 0;

    if (nd < max_dims) dims[nd] = (uint32_t)v;
    nd++;
    s = end;
    s = gptoss_skip_ws(s);
    if (*s == ',') {
      s++;
      continue;
    }
    if (*s == ']') break;
  }

  *out_ndims = nd;
  return 1;
}

static int gptoss_iejson_get_shape(const char *iejson,
                                  const char *tensor_name,
                                  uint32_t *dims,
                                  size_t max_dims,
                                  size_t *out_ndims) {
  if (!iejson || !tensor_name || !dims || !out_ndims) return 0;
  *out_ndims = 0;

  const char *reg = iejson_find_tensor_region(iejson, tensor_name);
  if (!reg) return 0;

  return iejson_parse_shape_dims(reg, dims, max_dims, out_ndims);
}

/* ============================================================================
 * Parsing and validation
 * ========================================================================== */

static void gptoss_hparams_ex_defaults(gptoss_hparams_ex_t *out_ex) {
  memset(out_ex, 0, sizeof(*out_ex));
  out_ex->rms_norm_eps = 1.0e-5f;
  out_ex->rope_theta = 10000.0f;
  out_ex->rope_scaling_type[0] = '\0';
  out_ex->rope_scaling_factor = 1.0f;
  out_ex->rope_scaling_original_max_position_embeddings = 0u;
  out_ex->tie_word_embeddings = 0;
}

static uint32_t gptoss_env_u32(const char *name, uint32_t defval) {
  const char *s = getenv(name);
  if (!s || !*s) return defval;
  errno = 0;
  char *end = NULL;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || errno != 0) return defval;
  if (v == 0ul || v > 0xFFFFFFFFul) return defval;
  return (uint32_t)v;
}

static void gptoss_apply_rope_scaling_env_if_absent(const gptoss_hparams_ex_t *ex) {
  if (!ex) return;

  /* User override wins. */
  if (getenv("IE_ROPE_LINEAR_SCALE") || getenv("IE_ROPE_POS_SCALE")) return;

  if (ex->rope_scaling_type[0] == '\0') return;
  if (!(ex->rope_scaling_factor > 0.0f)) return;

  /* Only "linear" is a constant position rescale. */
  if (strcmp(ex->rope_scaling_type, "linear") != 0) return;

  char tmp[64];
  const int n = snprintf(tmp, sizeof(tmp), "%.9g", (double)ex->rope_scaling_factor);
  if (n <= 0 || (size_t)n >= sizeof(tmp)) return;

  if (setenv("IE_ROPE_LINEAR_SCALE", tmp, 0) == 0) {
    fprintf(stderr,
            "[gptoss_hparams] Note: exported RoPE linear scaling factor=%s to IE_ROPE_LINEAR_SCALE.\n",
            tmp);
  }
}

static int gptoss_parse_config_json(const char *json, gptoss_hparams_ex_t *out_ex) {
  if (!json || !out_ex) return IE_IO_ERR_ARGS;

  uint32_t n_layers = 0, n_heads = 0, d_model_cfg = 0, d_ff_cfg = 0, vocab_cfg = 0, max_seq_cfg = 0;

  if (gptoss_scan_u32(json, "num_hidden_layers", &n_layers) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "num_attention_heads", &n_heads) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "hidden_size", &d_model_cfg) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "intermediate_size", &d_ff_cfg) != 1) return IE_IO_ERR_JSON;
  if (gptoss_scan_u32(json, "vocab_size", &vocab_cfg) != 1) return IE_IO_ERR_JSON;

  if (gptoss_scan_u32(json, "max_position_embeddings", &max_seq_cfg) != 1) {
    if (gptoss_scan_u32(json, "max_seq_len", &max_seq_cfg) != 1) {
      if (gptoss_scan_u32(json, "seq_length", &max_seq_cfg) != 1) {
        return IE_IO_ERR_JSON;
      }
    }
  }

  uint32_t n_kv_heads = 0;
  if (gptoss_scan_u32(json, "num_key_value_heads", &n_kv_heads) != 1) {
    n_kv_heads = n_heads;
  }

  if (n_layers == 0 || n_heads == 0 || n_kv_heads == 0 || d_model_cfg == 0 || d_ff_cfg == 0 ||
      vocab_cfg == 0 || max_seq_cfg == 0) {
    return IE_IO_ERR_JSON;
  }

  out_ex->core.n_layers = n_layers;
  out_ex->core.n_heads = n_heads;
  out_ex->core.n_kv_heads = n_kv_heads;
  out_ex->core.d_model = d_model_cfg;
  out_ex->core.d_ff = d_ff_cfg;
  out_ex->core.vocab_size = vocab_cfg;
  out_ex->core.max_seq = max_seq_cfg;

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
      int got_type = gptoss_scan_string(obj, "type", out_ex->rope_scaling_type,
                                        sizeof(out_ex->rope_scaling_type));
      if (got_type != 1) {
        (void)gptoss_scan_string(obj, "rope_type", out_ex->rope_scaling_type,
                                 sizeof(out_ex->rope_scaling_type));
      }

      {
        float factor = 0.0f;
        int got_factor = gptoss_scan_f32(obj, "factor", &factor);
        if (got_factor != 1) {
          got_factor = gptoss_scan_f32(obj, "scaling_factor", &factor);
        }
        if (got_factor == 1 && factor > 0.0f) out_ex->rope_scaling_factor = factor;
      }

      {
        uint32_t orig = 0;
        if (gptoss_scan_u32(obj, "original_max_position_embeddings", &orig) == 1) {
          out_ex->rope_scaling_original_max_position_embeddings = orig;
        }
      }
    }
  }

  return IE_IO_OK;
}

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

static int gptoss_resolve_iejson_path(const char *model_dir, char *out_path, size_t out_sz) {
  if (!model_dir || !*model_dir || !out_path || out_sz == 0) return IE_IO_ERR_ARGS;
  out_path[0] = '\0';

  char p1[1024];
  if (gptoss_join_path(p1, sizeof(p1), model_dir, "model.ie.json") != IE_IO_OK) return IE_IO_ERR_DECODE;
  if (gptoss_file_exists_regular(p1)) {
    const int n = snprintf(out_path, out_sz, "%s", p1);
    if (n < 0 || (size_t)n >= out_sz) return IE_IO_ERR_DECODE;
    return IE_IO_OK;
  }

  return IE_IO_ERR_OPEN;
}

static int gptoss_apply_iejson_inference(const char *model_dir, gptoss_hparams_ex_t *io_ex) {
  if (!model_dir || !io_ex) return IE_IO_ERR_ARGS;

  char iep[1024];
  const int r = gptoss_resolve_iejson_path(model_dir, iep, sizeof(iep));
  if (r != IE_IO_OK) return r;

  char *buf = NULL;
  size_t len = 0;
  const int rc = gptoss_read_all_text(iep, &buf, &len);
  if (rc != IE_IO_OK) return rc;
  if (!buf || len == 0) {
    free(buf);
    return IE_IO_ERR_READ;
  }

  /* Infer q_dim, kv_dim, and d_model from q_proj/k_proj shapes. */
  uint32_t q_dims[4] = {0, 0, 0, 0};
  uint32_t k_dims[4] = {0, 0, 0, 0};
  uint32_t v_dims[4] = {0, 0, 0, 0};
  size_t qn = 0, kn = 0, vn = 0;

  const int has_q = gptoss_iejson_get_shape(buf, "model.layers.0.self_attn.q_proj.weight", q_dims, 4, &qn);
  const int has_k = gptoss_iejson_get_shape(buf, "model.layers.0.self_attn.k_proj.weight", k_dims, 4, &kn);
  const int has_v = gptoss_iejson_get_shape(buf, "model.layers.0.self_attn.v_proj.weight", v_dims, 4, &vn);

  if (has_q && has_k && has_v && qn >= 2 && kn >= 2 && vn >= 2) {
    const uint32_t q_dim = q_dims[0];
    const uint32_t d_model_ie = q_dims[1];
    const uint32_t kv_dim_k = k_dims[0];
    const uint32_t kv_dim_v = v_dims[0];

    if (q_dim > 0 && d_model_ie > 0 && kv_dim_k > 0 && kv_dim_k == kv_dim_v) {
      if (io_ex->core.d_model != d_model_ie) {
        fprintf(stderr,
                "[gptoss_hparams] Note: config hidden_size=%u but model.ie.json q_proj in_dim=%u; using %u.\n",
                (unsigned)io_ex->core.d_model,
                (unsigned)d_model_ie,
                (unsigned)d_model_ie);
        io_ex->core.d_model = d_model_ie;
      }

      if (io_ex->core.n_heads == 0) {
        free(buf);
        return IE_IO_ERR_JSON;
      }
      if ((q_dim % io_ex->core.n_heads) != 0) {
        fprintf(stderr,
                "[gptoss_hparams] Error: q_dim=%u not divisible by n_heads=%u.\n",
                (unsigned)q_dim, (unsigned)io_ex->core.n_heads);
        free(buf);
        return IE_IO_ERR_DECODE;
      }
      const uint32_t d_head = q_dim / io_ex->core.n_heads;
      if (d_head == 0) {
        free(buf);
        return IE_IO_ERR_DECODE;
      }

      uint32_t n_kv = io_ex->core.n_kv_heads;
      if (n_kv == 0 || (kv_dim_k != (n_kv * d_head))) {
        if ((kv_dim_k % d_head) != 0) {
          fprintf(stderr,
                  "[gptoss_hparams] Error: kv_dim=%u not divisible by d_head=%u.\n",
                  (unsigned)kv_dim_k, (unsigned)d_head);
          free(buf);
          return IE_IO_ERR_DECODE;
        }
        n_kv = kv_dim_k / d_head;
        fprintf(stderr,
                "[gptoss_hparams] Note: overriding n_kv_heads to %u based on kv_dim=%u and d_head=%u.\n",
                (unsigned)n_kv, (unsigned)kv_dim_k, (unsigned)d_head);
      }

      if (n_kv == 0 || n_kv > io_ex->core.n_heads) {
        fprintf(stderr,
                "[gptoss_hparams] Error: inferred n_kv_heads=%u invalid for n_heads=%u.\n",
                (unsigned)n_kv, (unsigned)io_ex->core.n_heads);
        free(buf);
        return IE_IO_ERR_DECODE;
      }

      io_ex->core.d_head = d_head;
      io_ex->core.n_kv_heads = n_kv;
    }
  }

  /* Infer d_ff from MoE down_proj blocks: shape [n_experts, d_ff, ...] */
  {
    uint32_t dp_dims[8] = {0};
    size_t dpn = 0;
    const int has_dp = gptoss_iejson_get_shape(buf, "model.layers.0.mlp.experts.down_proj_blocks", dp_dims, 8, &dpn);
    if (has_dp && dpn >= 2 && dp_dims[1] > 0) {
      const uint32_t dff = dp_dims[1];
      if (io_ex->core.d_ff != dff) {
        fprintf(stderr,
                "[gptoss_hparams] Note: overriding d_ff from %u to %u based on down_proj_blocks.\n",
                (unsigned)io_ex->core.d_ff, (unsigned)dff);
        io_ex->core.d_ff = dff;
      }
    }
  }

  free(buf);
  return IE_IO_OK;
}

static int gptoss_finalize_and_validate(const char *model_dir, gptoss_hparams_ex_t *io_ex) {
  if (!io_ex) return IE_IO_ERR_ARGS;

  /* Clamp max_seq to avoid accidental enormous KV allocations.
     Override with IE_MAX_SEQ (must be > 0). */
  {
    const uint32_t max_cfg = io_ex->core.max_seq;
    const uint32_t max_env = gptoss_env_u32("IE_MAX_SEQ", 0u);
    uint32_t max_use = max_cfg;

    if (max_env > 0u) {
      max_use = (max_env < max_cfg) ? max_env : max_cfg;
    } else {
      const uint32_t clamp_default = 4096u;
      if (max_use > clamp_default) {
        fprintf(stderr,
                "[gptoss_hparams] Note: clamping max_seq from %u to %u (set IE_MAX_SEQ to override).\n",
                (unsigned)max_use, (unsigned)clamp_default);
        max_use = clamp_default;
      }
    }

    if (max_use == 0u) return IE_IO_ERR_JSON;
    io_ex->core.max_seq = max_use;
  }

  if (io_ex->core.n_layers == 0 || io_ex->core.n_heads == 0 || io_ex->core.n_kv_heads == 0 ||
      io_ex->core.d_model == 0 || io_ex->core.d_ff == 0 ||
      io_ex->core.vocab_size == 0 || io_ex->core.max_seq == 0) {
    fprintf(stderr, "[gptoss_hparams] Error: missing required hyperparameters after load.\n");
    return IE_IO_ERR_JSON;
  }

  /* Apply IEJSON inference (d_head, d_ff fixes, possible kv-head override). */
  if (model_dir) {
    const int rc = gptoss_apply_iejson_inference(model_dir, io_ex);
    if (rc != IE_IO_OK) return rc;
  }

  /* Export linear RoPE scaling to the RoPE helper, if present and not overridden. */
  gptoss_apply_rope_scaling_env_if_absent(io_ex);

  /* Validate after IEJSON overrides. */
  if (io_ex->core.d_head == 0 || io_ex->core.d_model == 0) return IE_IO_ERR_DECODE;
  if (io_ex->core.n_heads == 0 || io_ex->core.n_kv_heads == 0) return IE_IO_ERR_DECODE;
  if (io_ex->core.vocab_size == 0) return IE_IO_ERR_DECODE;

  if (io_ex->core.n_kv_heads > io_ex->core.n_heads) {
    fprintf(stderr,
            "[gptoss_hparams] Error: n_kv_heads=%u > n_heads=%u.\n",
            (unsigned)io_ex->core.n_kv_heads, (unsigned)io_ex->core.n_heads);
    return IE_IO_ERR_DECODE;
  }

  if ((io_ex->core.n_heads % io_ex->core.n_kv_heads) != 0) {
    fprintf(stderr,
            "[gptoss_hparams] Error: n_heads=%u not divisible by n_kv_heads=%u.\n",
            (unsigned)io_ex->core.n_heads, (unsigned)io_ex->core.n_kv_heads);
    return IE_IO_ERR_DECODE;
  }

  fprintf(stderr,
          "[gptoss_hparams] Loaded: layers=%u heads=%u kv_heads=%u d_model=%u d_head=%u d_ff=%u vocab=%u max_seq=%u\n",
          (unsigned)io_ex->core.n_layers,
          (unsigned)io_ex->core.n_heads,
          (unsigned)io_ex->core.n_kv_heads,
          (unsigned)io_ex->core.d_model,
          (unsigned)io_ex->core.d_head,
          (unsigned)io_ex->core.d_ff,
          (unsigned)io_ex->core.vocab_size,
          (unsigned)io_ex->core.max_seq);

  if (io_ex->rope_scaling_type[0] != '\0') {
    fprintf(stderr,
            "[gptoss_hparams] RoPE: theta=%.9g scaling_type='%s' scaling_factor=%.9g original_max_pos=%u\n",
            (double)io_ex->rope_theta,
            io_ex->rope_scaling_type,
            (double)io_ex->rope_scaling_factor,
            (unsigned)io_ex->rope_scaling_original_max_position_embeddings);
  } else {
    fprintf(stderr,
            "[gptoss_hparams] RoPE: theta=%.9g scaling=none\n",
            (double)io_ex->rope_theta);
  }

  return IE_IO_OK;
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

  const int rc = gptoss_hparams_load_ex_from_file(cfg, out_ex);
  if (rc != IE_IO_OK) return rc;

  return gptoss_finalize_and_validate(model_dir, out_ex);
}

int gptoss_hparams_load(const char *model_dir, ie_gptoss_hparams_t *out_hp) {
  if (!model_dir || !out_hp) return IE_IO_ERR_ARGS;

  gptoss_hparams_ex_t ex;
  const int rc = gptoss_hparams_load_ex(model_dir, &ex);
  if (rc != IE_IO_OK) return rc;

  *out_hp = ex.core;
  return IE_IO_OK;
}
