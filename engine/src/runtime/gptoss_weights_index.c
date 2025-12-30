/* ============================================================================
 * File: engine/src/runtime/gptoss_weights_index.c
 * ============================================================================
 */
/**
 * @file gptoss_weights_index.c
 * @brief Implementation of GPT-OSS weight indexing and name-resolution.
 *
 * @details
 * This module bridges the "weights IO world" (tensor_map.json + model.ie.bin +
 * optional dedup artifacts) with the runtime forward pass. It provides:
 *  - A stable way to resolve model tensors by trying common export conventions.
 *  - A unified tensor handle that supports direct pointers and dedup views.
 *
 * The forward pass should treat this module as read-only metadata:
 * no allocations should occur per token, and no file IO should occur after
 * initialization.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gptoss_weights_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__unix__) || defined(__APPLE__)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
  #include <sys/mman.h>
#endif

/* ============================================================================
 * Internal types
 * ========================================================================== */

/**
 * @struct gptoss_mapped_file_t
 * @brief Minimal read-only file mapping abstraction.
 *
 * @details
 * On POSIX systems, this uses mmap() when possible, and falls back to reading the
 * whole file into a malloc() buffer if mapping fails. On non-POSIX builds, only
 * the malloc() path is used.
 */
typedef struct gptoss_mapped_file_t {
  char   *path;     /**< Owned file path string. */
  int     fd;       /**< File descriptor (POSIX only). */
  void   *base;     /**< Base pointer (mmap or malloc buffer). */
  size_t  size;     /**< File size in bytes. */
  int     is_mmap;  /**< Non-zero if base is an mmap mapping. */
} gptoss_mapped_file_t;

/**
 * @struct gptoss_weights_index_t
 * @brief Concrete definition (opaque in the header).
 */
struct gptoss_weights_index_t {
  char model_dir[512];

  tensor_map_t tmap;

  gptoss_mapped_file_t bin;

  const ie_weights_t *weights;          /**< Borrowed pointer to weights descriptor. */
  const ie_weights_dedup_t *dedup;      /**< Borrowed or locally-owned (see dedup_owned). */
  ie_weights_dedup_t *dedup_owned;      /**< If non-NULL, we own this handle and must close. */
};

/* ============================================================================
 * Small utilities
 * ========================================================================== */

/**
 * @brief Duplicate a string using malloc().
 *
 * @param s Source string (may be NULL).
 * @return Newly allocated copy, or NULL on failure/NULL input.
 */
static char *widx_strdup(const char *s) {
  if (!s) return NULL;
  size_t n = strlen(s);
  char *p = (char*)malloc(n + 1u);
  if (!p) return NULL;
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

/**
 * @brief Join model_dir and a leaf filename into a fixed buffer.
 *
 * @param dst Destination buffer.
 * @param dst_n Size of destination buffer.
 * @param model_dir Model directory.
 * @param leaf Leaf filename.
 * @return 0 on success, non-zero on overflow.
 */
static int widx_path_join(char *dst, size_t dst_n, const char *model_dir, const char *leaf) {
  if (!dst || dst_n == 0 || !model_dir || !leaf) return -1;
  size_t a = strlen(model_dir);
  size_t b = strlen(leaf);
  size_t need = a + 1u + b + 1u;
  if (need > dst_n) return -1;
  memcpy(dst, model_dir, a);
  if (a > 0 && model_dir[a - 1] != '/') {
    dst[a] = '/';
    memcpy(dst + a + 1u, leaf, b);
    dst[a + 1u + b] = '\0';
  } else {
    memcpy(dst + a, leaf, b);
    dst[a + b] = '\0';
  }
  return 0;
}

/* ============================================================================
 * File mapping
 * ========================================================================== */

/**
 * @brief Close a mapped file and reset its fields.
 *
 * @param mf Mapped file descriptor (may be NULL).
 */
static void widx_mapped_close(gptoss_mapped_file_t *mf) {
  if (!mf) return;

#if defined(__unix__) || defined(__APPLE__)
  if (mf->is_mmap && mf->base && mf->size) {
    (void)munmap(mf->base, mf->size);
  } else if (!mf->is_mmap && mf->base) {
    free(mf->base);
  }

  if (mf->fd >= 0) {
    (void)close(mf->fd);
  }
#else
  if (mf->base) free(mf->base);
#endif

  free(mf->path);
  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;
}

/**
 * @brief Read a whole file into a malloc buffer.
 *
 * @param path File path.
 * @param out_base Output pointer (malloc buffer).
 * @param out_size Output size.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_read_all(const char *path, void **out_base, size_t *out_size) {
  if (!path || !out_base || !out_size) return IE_IO_ERR_ARGS;

  FILE *f = fopen(path, "rb");
  if (!f) return IE_IO_ERR_OPEN;

  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return IE_IO_ERR_STAT;
  }

  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return IE_IO_ERR_STAT;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return IE_IO_ERR_STAT;
  }

  void *buf = malloc((size_t)n);
  if (!buf) {
    fclose(f);
    return IE_IO_ERR_OOM;
  }

  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);

  if (got != (size_t)n) {
    free(buf);
    return IE_IO_ERR_READ;
  }

  *out_base = buf;
  *out_size = (size_t)n;
  return IE_IO_OK;
}

/**
 * @brief Open a file mapping (mmap preferred) for the weights binary.
 *
 * @param mf Output mapped file handle (reset then populated on success).
 * @param path Path to the file.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_mapped_open(gptoss_mapped_file_t *mf, const char *path) {
  if (!mf || !path) return IE_IO_ERR_ARGS;

  memset(mf, 0, sizeof(*mf));
  mf->fd = -1;

  mf->path = widx_strdup(path);
  if (!mf->path) return IE_IO_ERR_OOM;

#if defined(__unix__) || defined(__APPLE__)
  mf->fd = open(path, O_RDONLY);
  if (mf->fd < 0) {
    widx_mapped_close(mf);
    return IE_IO_ERR_OPEN;
  }

  struct stat st;
  if (fstat(mf->fd, &st) != 0) {
    widx_mapped_close(mf);
    return IE_IO_ERR_STAT;
  }

  if (st.st_size <= 0) {
    widx_mapped_close(mf);
    return IE_IO_ERR_STAT;
  }

  mf->size = (size_t)st.st_size;

  void *base = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
  if (base == MAP_FAILED) {
    /* mmap failed: fall back to buffered read. */
    mf->base = NULL;
    mf->size = 0;
    mf->is_mmap = 0;

    void *buf = NULL;
    size_t n = 0;
    int rc = widx_read_all(path, &buf, &n);
    if (rc != IE_IO_OK) {
      widx_mapped_close(mf);
      return rc;
    }

    mf->base = buf;
    mf->size = n;
    mf->is_mmap = 0;
    return IE_IO_OK;
  }

  mf->base = base;
  mf->is_mmap = 1;
  return IE_IO_OK;
#else
  void *buf = NULL;
  size_t n = 0;
  int rc = widx_read_all(path, &buf, &n);
  if (rc != IE_IO_OK) {
    widx_mapped_close(mf);
    return rc;
  }

  mf->base = buf;
  mf->size = n;
  mf->is_mmap = 0;
  return IE_IO_OK;
#endif
}

/* ============================================================================
 * Tensor resolution
 * ========================================================================== */

/**
 * @brief Reset a tensor handle to an empty state.
 *
 * @param t Tensor handle.
 */
static void widx_tensor_reset(gptoss_tensor_t *t) {
  if (!t) return;
  memset(t, 0, sizeof(*t));
}

/**
 * @brief Populate a tensor handle from a resolved descriptor and optional dedup view.
 *
 * @param idx Weights index.
 * @param desc Tensor descriptor (must be non-NULL).
 * @param out Output tensor handle.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_tensor_from_desc(const gptoss_weights_index_t *idx,
                                const tensor_desc_t *desc,
                                gptoss_tensor_t *out) {
  if (!idx || !desc || !out) return IE_IO_ERR_ARGS;

  widx_tensor_reset(out);
  out->desc = desc;
  out->nbytes = (size_t)desc->size_bytes;

  if (idx->weights && idx->weights->is_dedup && idx->dedup) {
    ie_weight_view_t view;
    memset(&view, 0, sizeof(view));
    ie_wdedup_status_t st = ie_weights_dedup_get_weight_view(idx->dedup, desc->name, &view);
    if (st != IE_WDEDUP_OK) {
      return IE_IO_ERR_JSON;
    }

    out->is_dedup = 1;
    out->view = view;

    if (view.kind == IE_WVIEW_DIRECT && view.data && view.nbytes > 0) {
      out->direct = (const uint8_t*)view.data;
      out->nbytes = view.nbytes;
    } else {
      out->direct = NULL;
      /* Prefer tensor_map size; view.nbytes may be 0 for dedup. */
      if (out->nbytes == 0 && view.nbytes > 0) out->nbytes = view.nbytes;
    }

    return IE_IO_OK;
  }

  /* Non-dedup: resolve direct pointer into mapped model.ie.bin. */
  if (!idx->bin.base || idx->bin.size == 0) return IE_IO_ERR_BIN_UNSPEC;

  uint64_t off = desc->offset;
  uint64_t end = off + desc->size_bytes;
  if (end < off) return IE_IO_ERR_DECODE;
  if (end > (uint64_t)idx->bin.size) return IE_IO_ERR_DECODE;

  out->direct = (const uint8_t*)idx->bin.base + (size_t)off;
  out->is_dedup = 0;
  return IE_IO_OK;
}

/**
 * @brief Find the first matching tensor descriptor among several candidate names.
 *
 * @param map Tensor map.
 * @param names Candidate names.
 * @param n Number of candidate names.
 * @return Pointer to descriptor, or NULL.
 */
static const tensor_desc_t *widx_find_any(const tensor_map_t *map,
                                         const char *const *names,
                                         size_t n) {
  if (!map || !names) return NULL;
  for (size_t i = 0; i < n; ++i) {
    const char *name = names[i];
    if (!name) continue;
    const tensor_desc_t *d = tensor_map_find(map, name);
    if (d) return d;
  }
  return NULL;
}

/**
 * @brief Format a tensor name into a buffer.
 *
 * @param dst Destination buffer.
 * @param dst_n Capacity.
 * @param fmt printf-style format.
 * @param layer Layer index.
 * @return 0 on success, non-zero on failure.
 */
static int widx_fmt_layer(char *dst, size_t dst_n, const char *fmt, uint32_t layer) {
  if (!dst || dst_n == 0 || !fmt) return -1;
  int n = snprintf(dst, dst_n, fmt, (unsigned)layer);
  if (n < 0) return -1;
  if ((size_t)n >= dst_n) return -1;
  return 0;
}

/**
 * @brief Resolve a required tensor by name candidates.
 *
 * @param idx Index.
 * @param names Candidate names.
 * @param n Candidate count.
 * @param out Output tensor handle.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_resolve_required(const gptoss_weights_index_t *idx,
                                const char *const *names,
                                size_t n,
                                gptoss_tensor_t *out) {
  const tensor_desc_t *d = widx_find_any(&idx->tmap, names, n);
  if (!d) return IE_IO_ERR_JSON;
  return widx_tensor_from_desc(idx, d, out);
}

/**
 * @brief Resolve an optional tensor by name candidates.
 *
 * @param idx Index.
 * @param names Candidate names.
 * @param n Candidate count.
 * @param out Output tensor handle (empty if not found).
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_resolve_optional(const gptoss_weights_index_t *idx,
                                const char *const *names,
                                size_t n,
                                gptoss_tensor_t *out) {
  const tensor_desc_t *d = widx_find_any(&idx->tmap, names, n);
  if (!d) {
    widx_tensor_reset(out);
    return IE_IO_OK;
  }
  return widx_tensor_from_desc(idx, d, out);
}

/* ============================================================================
 * Public API: tensor helpers
 * ========================================================================== */

/**
 * @brief Return non-zero if a tensor handle is populated.
 *
 * @param t Tensor handle.
 * @return Non-zero if populated, 0 otherwise.
 */
int gptoss_tensor_is_valid(const gptoss_tensor_t *t) {
  return (t && t->desc != NULL);
}

/**
 * @brief Return a direct pointer to tensor bytes if available.
 *
 * @param t Tensor handle.
 * @return Pointer to bytes, or NULL if materialization is required.
 */
const uint8_t *gptoss_tensor_bytes(const gptoss_tensor_t *t) {
  if (!t) return NULL;
  return t->direct;
}

/**
 * @brief Materialize tensor bytes into a caller-provided buffer.
 *
 * @param t Tensor handle.
 * @param dst Destination buffer.
 * @param dst_nbytes Capacity of destination buffer.
 * @return Number of bytes written (0 on failure).
 */
size_t gptoss_tensor_materialize(const gptoss_tensor_t *t, void *dst, size_t dst_nbytes) {
  if (!t || !t->desc || !dst || dst_nbytes == 0) return 0;
  if (t->direct) {
    size_t n = t->nbytes;
    if (n > dst_nbytes) return 0;
    memcpy(dst, t->direct, n);
    return n;
  }
  if (t->is_dedup) {
    return ie_weights_dedup_materialize(&t->view, dst, dst_nbytes);
  }
  return 0;
}

/* ============================================================================
 * Public API: open/close
 * ========================================================================== */

/**
 * @brief Open a weights index for a model directory.
 *
 * @param out Output index.
 * @param model_dir Model directory.
 * @param weights Weights descriptor opened by ::ie_weights_open.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_weights_index_open(gptoss_weights_index_t *out,
                             const char *model_dir,
                             const ie_weights_t *weights) {
  if (!out || !model_dir || !weights) return IE_IO_ERR_ARGS;

  memset(out, 0, sizeof(*out));
  out->bin.fd = -1;
  out->weights = weights;

  if (strlen(model_dir) >= sizeof(out->model_dir)) return IE_IO_ERR_ARGS;
  memcpy(out->model_dir, model_dir, strlen(model_dir) + 1u);

  /* Load tensor_map.json (try the canonical root path first). */
  char tmap_path[1024];
  memset(tmap_path, 0, sizeof(tmap_path));

  if (widx_path_join(tmap_path, sizeof(tmap_path), model_dir, "tensor_map.json") != 0) {
    return IE_IO_ERR_ARGS;
  }

  if (tensor_map_load(tmap_path, &out->tmap) != 0) {
    /* Fallback: some trees keep it under hf/original. */
    if (widx_path_join(tmap_path, sizeof(tmap_path), model_dir, "hf/original/tensor_map.json") != 0) {
      return IE_IO_ERR_JSON;
    }
    if (tensor_map_load(tmap_path, &out->tmap) != 0) {
      return IE_IO_ERR_JSON;
    }
  }

  /* Map the weights binary. */
  if (!weights->weights_path[0]) {
    tensor_map_free(&out->tmap);
    return IE_IO_ERR_BIN_UNSPEC;
  }

  int rc = widx_mapped_open(&out->bin, weights->weights_path);
  if (rc != IE_IO_OK) {
    tensor_map_free(&out->tmap);
    return rc;
  }

  /* Dedup: reuse borrowed handle if available, else open one. */
  out->dedup = NULL;
  out->dedup_owned = NULL;

  if (weights->is_dedup) {
    if (weights->dedup_handle) {
      out->dedup = (const ie_weights_dedup_t*)weights->dedup_handle;
    } else {
      ie_weights_dedup_t *h = NULL;
      ie_weights_dedup_opts_t opts;
      memset(&opts, 0, sizeof(opts));
      opts.prefault_policy = 0;
      ie_wdedup_status_t st = ie_weights_dedup_open(&h, model_dir, &opts);
      if (st != IE_WDEDUP_OK) {
        /* If dedup is enabled but we cannot open it, keep running in non-dedup mode. */
        out->dedup = NULL;
        out->dedup_owned = NULL;
      } else {
        out->dedup = h;
        out->dedup_owned = h;
      }
    }
  }

  return IE_IO_OK;
}

/**
 * @brief Close an index and free all resources owned by it.
 *
 * @param idx Index handle (may be NULL).
 */
void gptoss_weights_index_close(gptoss_weights_index_t *idx) {
  if (!idx) return;

  if (idx->dedup_owned) {
    ie_weights_dedup_t *h = idx->dedup_owned;
    ie_weights_dedup_close(&h);
    idx->dedup_owned = NULL;
    idx->dedup = NULL;
  }

  tensor_map_free(&idx->tmap);
  widx_mapped_close(&idx->bin);

  memset(idx, 0, sizeof(*idx));
  idx->bin.fd = -1;
}

/* ============================================================================
 * Name probing and build_model
 * ========================================================================== */

/**
 * @brief Detect the model naming scheme and the projection style for layer 0.
 *
 * @param idx Index.
 * @param out_arch Output detected scheme.
 * @param out_fused_qkv Output non-zero if fused QKV is detected.
 * @param out_swiglu Output non-zero if SwiGLU MLP is detected.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_detect_arch(const gptoss_weights_index_t *idx,
                            gptoss_arch_kind_t *out_arch,
                            int *out_fused_qkv,
                            int *out_swiglu) {
  if (!idx || !out_arch || !out_fused_qkv || !out_swiglu) return IE_IO_ERR_ARGS;

  *out_arch = GPTOSS_ARCH_UNKNOWN;
  *out_fused_qkv = 0;
  *out_swiglu = 0;

  /* Probe common LLaMA-style paths. */
  const char *llama_q_names[] = {
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.attn.q_proj.weight"
  };
  const char *llama_fused_names[] = {
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.self_attn.qkv.weight",
    "model.layers.0.attn.qkv_proj.weight",
    "model.layers.0.attn.qkv.weight"
  };

  if (widx_find_any(&idx->tmap, llama_q_names, sizeof(llama_q_names)/sizeof(llama_q_names[0])) ||
      widx_find_any(&idx->tmap, llama_fused_names, sizeof(llama_fused_names)/sizeof(llama_fused_names[0]))) {
    *out_arch = GPTOSS_ARCH_LLAMA;
    if (widx_find_any(&idx->tmap, llama_fused_names, sizeof(llama_fused_names)/sizeof(llama_fused_names[0]))) {
      *out_fused_qkv = 1;
    }
    /* SwiGLU probe: gate_proj is the tell. */
    if (tensor_map_find(&idx->tmap, "model.layers.0.mlp.gate_proj.weight") ||
        tensor_map_find(&idx->tmap, "model.layers.0.feed_forward.gate_proj.weight")) {
      *out_swiglu = 1;
    }
    return IE_IO_OK;
  }

  /* Probe GPT-NeoX style. */
  const char *neox_fused_qkv[] = {
    "gpt_neox.layers.0.attention.query_key_value.weight"
  };
  if (widx_find_any(&idx->tmap, neox_fused_qkv, sizeof(neox_fused_qkv)/sizeof(neox_fused_qkv[0])) ||
      tensor_map_find(&idx->tmap, "gpt_neox.layers.0.attention.query.weight")) {
    *out_arch = GPTOSS_ARCH_GPTNEOX;
    /* NeoX typically uses fused QKV. */
    if (widx_find_any(&idx->tmap, neox_fused_qkv, sizeof(neox_fused_qkv)/sizeof(neox_fused_qkv[0]))) {
      *out_fused_qkv = 1;
    }
    /* NeoX MLP is usually GeLU (dense_h_to_4h + dense_4h_to_h). */
    *out_swiglu = 0;
    return IE_IO_OK;
  }

  return IE_IO_ERR_JSON;
}

/**
 * @brief Resolve a LLaMA-style layer weights set.
 *
 * @param idx Index.
 * @param layer Layer index.
 * @param fused_qkv Non-zero if fused QKV should be used.
 * @param swiglu Non-zero if SwiGLU should be used.
 * @param out Output layer weights.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_resolve_layer_llama(const gptoss_weights_index_t *idx,
                                   uint32_t layer,
                                   int fused_qkv,
                                   int swiglu,
                                   gptoss_layer_weights_t *out) {
  if (!idx || !out) return IE_IO_ERR_ARGS;
  memset(out, 0, sizeof(*out));

  char name[256];

  /* Norm weights */
  if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.input_layernorm.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) {
      if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.attn_norm.weight", layer) != 0) return IE_IO_ERR_ARGS;
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->attn_norm_w);
    if (rc != IE_IO_OK) return rc;
  }

  if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.post_attention_layernorm.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) {
      if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.mlp_norm.weight", layer) != 0) return IE_IO_ERR_ARGS;
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->mlp_norm_w);
    if (rc != IE_IO_OK) return rc;
  }

  /* Attention projections */
  if (fused_qkv) {
    const tensor_desc_t *d = NULL;
    if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.self_attn.qkv_proj.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d && widx_fmt_layer(name, sizeof(name), "model.layers.%u.self_attn.qkv.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d && widx_fmt_layer(name, sizeof(name), "model.layers.%u.attn.qkv_proj.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d && widx_fmt_layer(name, sizeof(name), "model.layers.%u.attn.qkv.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d) return IE_IO_ERR_JSON;

    int rc = widx_tensor_from_desc(idx, d, &out->qkv_proj_w);
    if (rc != IE_IO_OK) return rc;

    widx_tensor_reset(&out->q_proj_w);
    widx_tensor_reset(&out->k_proj_w);
    widx_tensor_reset(&out->v_proj_w);
  } else {
    gptoss_tensor_t *dsts[] = { &out->q_proj_w, &out->k_proj_w, &out->v_proj_w };
    const char *fmts[] = {
      "model.layers.%u.self_attn.q_proj.weight",
      "model.layers.%u.self_attn.k_proj.weight",
      "model.layers.%u.self_attn.v_proj.weight"
    };
    for (int i = 0; i < 3; ++i) {
      if (widx_fmt_layer(name, sizeof(name), fmts[i], layer) != 0) return IE_IO_ERR_ARGS;
      const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
      if (!d) {
        /* Fallback shorter alias. */
        const char *fmt2 = NULL;
        if (i == 0) fmt2 = "model.layers.%u.attn.q_proj.weight";
        if (i == 1) fmt2 = "model.layers.%u.attn.k_proj.weight";
        if (i == 2) fmt2 = "model.layers.%u.attn.v_proj.weight";
        if (fmt2 && widx_fmt_layer(name, sizeof(name), fmt2, layer) != 0) return IE_IO_ERR_ARGS;
        d = fmt2 ? tensor_map_find(&idx->tmap, name) : NULL;
      }
      if (!d) return IE_IO_ERR_JSON;
      int rc = widx_tensor_from_desc(idx, d, dsts[i]);
      if (rc != IE_IO_OK) return rc;
    }
    widx_tensor_reset(&out->qkv_proj_w);
  }

  /* Output projection */
  {
    const tensor_desc_t *d = NULL;
    if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.self_attn.o_proj.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d && widx_fmt_layer(name, sizeof(name), "model.layers.%u.attn.o_proj.weight", layer) == 0) {
      d = tensor_map_find(&idx->tmap, name);
    }
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->o_proj_w);
    if (rc != IE_IO_OK) return rc;
  }

  /* MLP projections */
  if (swiglu) {
    const char *gate_fmt[] = {
      "model.layers.%u.mlp.gate_proj.weight",
      "model.layers.%u.feed_forward.gate_proj.weight"
    };
    const char *up_fmt[] = {
      "model.layers.%u.mlp.up_proj.weight",
      "model.layers.%u.feed_forward.up_proj.weight"
    };
    const char *down_fmt[] = {
      "model.layers.%u.mlp.down_proj.weight",
      "model.layers.%u.feed_forward.down_proj.weight"
    };

    const tensor_desc_t *d_gate = NULL;
    for (size_t k = 0; k < sizeof(gate_fmt)/sizeof(gate_fmt[0]); ++k) {
      if (widx_fmt_layer(name, sizeof(name), gate_fmt[k], layer) == 0) {
        d_gate = tensor_map_find(&idx->tmap, name);
        if (d_gate) break;
      }
    }
    const tensor_desc_t *d_up = NULL;
    for (size_t k = 0; k < sizeof(up_fmt)/sizeof(up_fmt[0]); ++k) {
      if (widx_fmt_layer(name, sizeof(name), up_fmt[k], layer) == 0) {
        d_up = tensor_map_find(&idx->tmap, name);
        if (d_up) break;
      }
    }
    const tensor_desc_t *d_down = NULL;
    for (size_t k = 0; k < sizeof(down_fmt)/sizeof(down_fmt[0]); ++k) {
      if (widx_fmt_layer(name, sizeof(name), down_fmt[k], layer) == 0) {
        d_down = tensor_map_find(&idx->tmap, name);
        if (d_down) break;
      }
    }

    if (!d_gate || !d_up || !d_down) return IE_IO_ERR_JSON;

    int rc = widx_tensor_from_desc(idx, d_gate, &out->gate_proj_w);
    if (rc != IE_IO_OK) return rc;
    rc = widx_tensor_from_desc(idx, d_up, &out->up_proj_w);
    if (rc != IE_IO_OK) return rc;
    rc = widx_tensor_from_desc(idx, d_down, &out->down_proj_w);
    if (rc != IE_IO_OK) return rc;

    widx_tensor_reset(&out->fc_in_w);
  } else {
    /* GeLU-style (dense in + dense out) fallback. */
    const tensor_desc_t *d_in = NULL;
    const tensor_desc_t *d_out = NULL;

    if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.mlp.fc_in.weight", layer) == 0) {
      d_in = tensor_map_find(&idx->tmap, name);
    }
    if (!d_in && widx_fmt_layer(name, sizeof(name), "model.layers.%u.mlp.w1.weight", layer) == 0) {
      d_in = tensor_map_find(&idx->tmap, name);
    }

    if (widx_fmt_layer(name, sizeof(name), "model.layers.%u.mlp.fc_out.weight", layer) == 0) {
      d_out = tensor_map_find(&idx->tmap, name);
    }
    if (!d_out && widx_fmt_layer(name, sizeof(name), "model.layers.%u.mlp.w2.weight", layer) == 0) {
      d_out = tensor_map_find(&idx->tmap, name);
    }

    if (!d_in || !d_out) return IE_IO_ERR_JSON;

    int rc = widx_tensor_from_desc(idx, d_in, &out->fc_in_w);
    if (rc != IE_IO_OK) return rc;
    rc = widx_tensor_from_desc(idx, d_out, &out->down_proj_w);
    if (rc != IE_IO_OK) return rc;

    widx_tensor_reset(&out->gate_proj_w);
    widx_tensor_reset(&out->up_proj_w);
  }

  return IE_IO_OK;
}

/**
 * @brief Resolve a GPT-NeoX-style layer weights set.
 *
 * @param idx Index.
 * @param layer Layer index.
 * @param out Output layer weights.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
static int widx_resolve_layer_neox(const gptoss_weights_index_t *idx,
                                  uint32_t layer,
                                  gptoss_layer_weights_t *out) {
  if (!idx || !out) return IE_IO_ERR_ARGS;
  memset(out, 0, sizeof(*out));

  char name[256];

  /* Norms */
  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.input_layernorm.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->attn_norm_w);
    if (rc != IE_IO_OK) return rc;
  }

  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.post_attention_layernorm.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->mlp_norm_w);
    if (rc != IE_IO_OK) return rc;
  }

  /* Attention projections (prefer fused QKV). */
  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.attention.query_key_value.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->qkv_proj_w);
    if (rc != IE_IO_OK) return rc;
  }

  widx_tensor_reset(&out->q_proj_w);
  widx_tensor_reset(&out->k_proj_w);
  widx_tensor_reset(&out->v_proj_w);

  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.attention.dense.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->o_proj_w);
    if (rc != IE_IO_OK) return rc;
  }

  /* MLP (GeLU-style) */
  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.mlp.dense_h_to_4h.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->fc_in_w);
    if (rc != IE_IO_OK) return rc;
  }

  if (widx_fmt_layer(name, sizeof(name), "gpt_neox.layers.%u.mlp.dense_4h_to_h.weight", layer) != 0) return IE_IO_ERR_ARGS;
  {
    const tensor_desc_t *d = tensor_map_find(&idx->tmap, name);
    if (!d) return IE_IO_ERR_JSON;
    int rc = widx_tensor_from_desc(idx, d, &out->down_proj_w);
    if (rc != IE_IO_OK) return rc;
  }

  widx_tensor_reset(&out->gate_proj_w);
  widx_tensor_reset(&out->up_proj_w);

  return IE_IO_OK;
}

/**
 * @brief Resolve and validate all tensors required by the GPT-OSS forward pass.
 *
 * @param idx Opened weights index.
 * @param hp Hyperparameters (layer count).
 * @param out Output model weights index.
 * @return IE_IO_OK on success, negative ::ie_io_status_t on failure.
 */
int gptoss_weights_index_build_model(const gptoss_weights_index_t *idx,
                                    const ie_gptoss_hparams_t *hp,
                                    gptoss_model_weights_t *out) {
  if (!idx || !hp || !out) return IE_IO_ERR_ARGS;

  memset(out, 0, sizeof(*out));

  gptoss_arch_kind_t arch;
  int fused_qkv = 0;
  int swiglu = 0;

  int rc = widx_detect_arch(idx, &arch, &fused_qkv, &swiglu);
  if (rc != IE_IO_OK) return rc;

  out->arch = arch;
  out->attn_fused_qkv = fused_qkv;
  out->mlp_swiglu = swiglu;
  out->n_layers = hp->n_layers;

  /* Core (embedding / final norm / lm head) */
  {
    const char *embed_names[] = {
      "model.embed_tokens.weight",
      "tok_embeddings.weight",
      "model.tok_embeddings.weight",
      "transformer.wte.weight",
      "gpt_neox.embed_in.weight"
    };
    rc = widx_resolve_required(idx, embed_names, sizeof(embed_names)/sizeof(embed_names[0]), &out->tok_embed_w);
    if (rc != IE_IO_OK) return rc;
  }

  {
    const char *pos_names[] = {
      "model.embed_positions.weight",
      "transformer.wpe.weight",
      "gpt_neox.embed_positions.weight"
    };
    rc = widx_resolve_optional(idx, pos_names, sizeof(pos_names)/sizeof(pos_names[0]), &out->pos_embed_w);
    if (rc != IE_IO_OK) return rc;
  }

  {
    const char *norm_names[] = {
      "model.norm.weight",
      "transformer.norm.weight",
      "gpt_neox.final_layer_norm.weight",
      "final_layer_norm.weight"
    };
    rc = widx_resolve_required(idx, norm_names, sizeof(norm_names)/sizeof(norm_names[0]), &out->final_norm_w);
    if (rc != IE_IO_OK) return rc;
  }

  {
    const char *lm_head_names[] = {
      "lm_head.weight",
      "model.lm_head.weight",
      "transformer.lm_head.weight",
      "embed_out.weight",
      "gpt_neox.embed_out.weight"
    };
    rc = widx_resolve_optional(idx, lm_head_names, sizeof(lm_head_names)/sizeof(lm_head_names[0]), &out->lm_head_w);
    if (rc != IE_IO_OK) return rc;

    /* If lm_head is absent, assume tied embeddings. */
    if (!out->lm_head_w.desc) {
      out->lm_head_w = out->tok_embed_w;
    }
  }

  /* Layers */
  if (hp->n_layers == 0) return IE_IO_ERR_ARGS;

  out->layers = (gptoss_layer_weights_t*)calloc((size_t)hp->n_layers, sizeof(gptoss_layer_weights_t));
  if (!out->layers) return IE_IO_ERR_OOM;

  for (uint32_t l = 0; l < hp->n_layers; ++l) {
    if (arch == GPTOSS_ARCH_LLAMA) {
      rc = widx_resolve_layer_llama(idx, l, fused_qkv, swiglu, &out->layers[l]);
    } else if (arch == GPTOSS_ARCH_GPTNEOX) {
      rc = widx_resolve_layer_neox(idx, l, &out->layers[l]);
    } else {
      rc = IE_IO_ERR_JSON;
    }

    if (rc != IE_IO_OK) {
      gptoss_model_weights_free(out);
      return rc;
    }
  }

  return IE_IO_OK;
}

/**
 * @brief Free the per-layer array allocated by ::gptoss_weights_index_build_model.
 *
 * @param mw Model weights.
 */
void gptoss_model_weights_free(gptoss_model_weights_t *mw) {
  if (!mw) return;
  free(mw->layers);
  memset(mw, 0, sizeof(*mw));
}
