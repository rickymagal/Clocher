/**
 * @file ie_device_common.c
 * @brief C11 device abstraction with CPU backend and a real CUDA backend.
 *
 * The CUDA backend is implemented through a narrow C ABI wrapper layer
 * (ie_device_cuda.cu + ie_device_cuda.h) to avoid including CUDA headers in C
 * translation units.
 *
 * Key properties:
 * - If CUDA is explicitly requested and initialization fails, we return an error
 *   (no silent CPU fallback that would create fake GPU reports).
 * - GEMV FP32 on CUDA is correctness-first: it copies inputs to the device,
 *   launches a simple kernel, and copies outputs back.
 *   Optimization (persistent weight residency, streams, fusion, etc.) can follow.
 */

#include "ie_device.h"
#include "ie_device_cuda.h"
#include "ie_kernels.h"
#include "sparse_format.h"
#include "util_logging.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =============================================================================
 * VTable
 * ========================================================================== */

/**
 * @brief Virtual method table for a device backend.
 */
typedef struct ie_device_vtbl {
  int  (*caps)(const void *self, ie_device_caps_t *out);
  int  (*gemv_f32)(void *self, const float *W, const float *x, float *y,
                   size_t rows, size_t cols, const float *bias, size_t blk_k);
  int  (*gemv_q4_0_f32)(void *self,
                       const uint8_t *w_q4,
                       const uint8_t *w_scales,
                       size_t scale_bytes,
                       const float *x,
                       float *y,
                       size_t rows,
                       size_t cols,
                       const uint16_t *bias_bf16);
  int  (*memcpy)(void *self, void *dst, const void *src, size_t nbytes);
  int  (*gemv_block_sparse_f32)(void *self,
                                const ie_block_sparse_matrix_t *m,
                                const float *x,
                                float *y,
                                const float *bias);
  void (*destroy)(void *self);
} ie_device_vtbl_t;

/**
 * @brief Public device object combining vtable and backend-specific impl pointer.
 */
struct ie_device {
  ie_device_vtbl_t vt;      /**< Virtual function table. */
  void *impl;               /**< Backend-specific implementation struct. */
  ie_device_kind_t kind;    /**< Logical device kind (CPU, CUDA, ZE). */
};

/* =============================================================================
 * CPU backend
 * ========================================================================== */

typedef struct cpu_impl {
  char name[64];
} cpu_impl_t;

/**
 * @brief Allocate a CPU backend instance.
 *
 * @return Pointer to newly allocated cpu_impl_t or NULL on OOM.
 */
static cpu_impl_t *cpu_new(void) {
  cpu_impl_t *p = (cpu_impl_t*)calloc(1, sizeof(*p));
  if (!p) return NULL;
  snprintf(p->name, sizeof(p->name), "CPU");
  return p;
}

/**
 * @brief Report CPU device capabilities.
 *
 * CPU backend supports GEMV FP32 and memcpy.
 */
static int cpu_caps(const void *self, ie_device_caps_t *out) {
  (void)self;
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  out->has_gemv_f32 = 1;
  out->has_mem_copy = 1;
  out->has_streams  = 0;
  snprintf(out->name,   sizeof(out->name),   "CPU");
  snprintf(out->driver, sizeof(out->driver), "N/A");
  return 0;
}

/**
 * @brief Execute dense GEMV FP32 on CPU using existing kernels.
 */
static int cpu_gemv_f32(void *self,
                        const float *W, const float *x, float *y,
                        size_t rows, size_t cols,
                        const float *bias, size_t blk_k) {
  (void)self;
  (void)blk_k;
  if (!W || !x || !y || rows == 0 || cols == 0) return -1;
  ie_gemv_f32(W, x, y, rows, cols, bias, 0);
  return 0;
}

/**
 * @brief Execute Q4_0 GEMV on CPU using existing kernels.
 */
static int cpu_gemv_q4_0_f32(void *self,
                             const uint8_t *w_q4,
                             const uint8_t *w_scales,
                             size_t scale_bytes,
                             const float *x,
                             float *y,
                             size_t rows,
                             size_t cols,
                             const uint16_t *bias_bf16) {
  (void)self;
  return ie_gemv_q4_0_f32(w_q4, w_scales, scale_bytes, x, y, rows, cols, bias_bf16);
}

/**
 * @brief CPU memcpy implementation.
 */
static int cpu_memcpy(void *self, void *dst, const void *src, size_t nbytes) {
  (void)self;
  if (!dst || !src) return -1;
  memcpy(dst, src, nbytes);
  return 0;
}

/**
 * @brief CPU implementation of block-sparse GEMV FP32.
 */
static int cpu_gemv_block_sparse_f32(void *self,
                                     const ie_block_sparse_matrix_t *m,
                                     const float *x,
                                     float *y,
                                     const float *bias) {
  (void)self;
  if (!m || !x || !y) return -1;
  ie_gemv_block_sparse_f32(m, x, y, bias);
  return 0;
}

/**
 * @brief Destroy CPU backend instance.
 */
static void cpu_destroy(void *self) {
  free(self);
}

/* =============================================================================
 * CUDA backend (real)
 * ========================================================================== */

/**
 * @brief CUDA backend implementation state.
 *
 * This implementation is correctness-first and uses temporary device buffers.
 * A later optimization step should keep weights resident on-device and use
 * streams to overlap transfers and compute.
 */
typedef struct cuda_impl {
  int device_ordinal;

  char name[64];
  char driver[64];

  void  *dW;
  size_t dW_cap;

  void  *dx;
  size_t dx_cap;

  void  *dy;
  size_t dy_cap;

  void  *dbias;
  size_t dbias_cap;

  void  *dW_scales;
  size_t dW_scales_cap;

  /* Optional device mirror for a large, immutable host blob. */
  const uint8_t *blob_host;
  void  *blob_dev;
  size_t blob_bytes;

  /* Optional weight cache (host ptr -> device ptr). */
  size_t cache_cap_bytes;
  size_t cache_bytes;
  struct cuda_cache_entry *cache_head;
  struct cuda_cache_entry *cache_tail;
  uint64_t cache_hits;
  uint64_t cache_misses;
  uint64_t cache_evicts;
} cuda_impl_t;

typedef struct cuda_cache_entry {
  const void *host_ptr;
  size_t nbytes;
  void *dev_ptr;
  struct cuda_cache_entry *prev;
  struct cuda_cache_entry *next;
} cuda_cache_entry_t;

static size_t cuda_cache_cap_from_env(void) {
  const char *s = getenv("IE_CUDA_CACHE_MB");
  if (!s || !*s) return 0;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (end == s || v <= 0) return 0;
  return (size_t)v * (size_t)1024u * (size_t)1024u;
}

static void cuda_cache_move_front(cuda_impl_t *ci, cuda_cache_entry_t *e) {
  if (!ci || !e || ci->cache_head == e) return;
  if (e->prev) e->prev->next = e->next;
  if (e->next) e->next->prev = e->prev;
  if (ci->cache_tail == e) ci->cache_tail = e->prev;
  e->prev = NULL;
  e->next = ci->cache_head;
  if (ci->cache_head) ci->cache_head->prev = e;
  ci->cache_head = e;
  if (!ci->cache_tail) ci->cache_tail = e;
}

static void cuda_cache_evict_tail(cuda_impl_t *ci) {
  if (!ci || !ci->cache_tail) return;
  cuda_cache_entry_t *e = ci->cache_tail;
  if (e->prev) e->prev->next = NULL;
  ci->cache_tail = e->prev;
  if (ci->cache_head == e) ci->cache_head = NULL;
  ci->cache_bytes -= e->nbytes;
  ie_cuda_free(e->dev_ptr);
  ci->cache_evicts++;
  free(e);
}

static void cuda_cache_clear(cuda_impl_t *ci) {
  if (!ci) return;
  while (ci->cache_tail) cuda_cache_evict_tail(ci);
  ci->cache_head = NULL;
  ci->cache_tail = NULL;
  ci->cache_bytes = 0;
}

static void *cuda_cache_get(cuda_impl_t *ci, const void *host_ptr, size_t nbytes) {
  if (!ci || !host_ptr || nbytes == 0 || ci->cache_cap_bytes == 0) return NULL;
  for (cuda_cache_entry_t *e = ci->cache_head; e; e = e->next) {
    if (e->host_ptr == host_ptr && e->nbytes == nbytes) {
      cuda_cache_move_front(ci, e);
      ci->cache_hits++;
      return e->dev_ptr;
    }
  }
  ci->cache_misses++;

  if (nbytes > ci->cache_cap_bytes) return NULL;
  while (ci->cache_bytes + nbytes > ci->cache_cap_bytes && ci->cache_tail) {
    cuda_cache_evict_tail(ci);
  }

  void *d = ie_cuda_malloc(nbytes);
  if (!d) {
    ie_cuda_clear_last_error();
    return NULL;
  }
  if (ie_cuda_memcpy(d, host_ptr, nbytes, IE_CUDA_COPY_H2D) != 0) {
    ie_cuda_free(d);
    ie_cuda_clear_last_error();
    return NULL;
  }

  cuda_cache_entry_t *e = (cuda_cache_entry_t *)calloc(1, sizeof(*e));
  if (!e) {
    ie_cuda_free(d);
    return NULL;
  }
  e->host_ptr = host_ptr;
  e->nbytes = nbytes;
  e->dev_ptr = d;
  e->next = ci->cache_head;
  if (ci->cache_head) ci->cache_head->prev = e;
  ci->cache_head = e;
  if (!ci->cache_tail) ci->cache_tail = e;
  ci->cache_bytes += nbytes;
  return d;
}

static int cuda_q4_map(cuda_impl_t *ci,
                       const uint8_t *w_q4, size_t W_bytes,
                       const uint8_t *w_scales, size_t S_bytes,
                       const uint8_t **dW, const uint8_t **dS) {
  if (!ci || !w_q4 || !w_scales || !dW || !dS) return -1;
  *dW = NULL;
  *dS = NULL;

  if (ci->blob_dev && ci->blob_host && ci->blob_bytes > 0) {
    const uint8_t *base = ci->blob_host;
    const uint8_t *end = base + ci->blob_bytes;
    if (w_q4 >= base && (w_q4 + W_bytes) <= end &&
        w_scales >= base && (w_scales + S_bytes) <= end) {
      const size_t w_off = (size_t)(w_q4 - base);
      const size_t s_off = (size_t)(w_scales - base);
      *dW = (const uint8_t *)ci->blob_dev + w_off;
      *dS = (const uint8_t *)ci->blob_dev + s_off;
      return 0;
    }
  }

  if (ci->cache_cap_bytes > 0) {
    const size_t want = W_bytes + S_bytes;
    if (want <= ci->cache_cap_bytes) {
      *dW = (const uint8_t *)cuda_cache_get(ci, w_q4, W_bytes);
      *dS = (const uint8_t *)cuda_cache_get(ci, w_scales, S_bytes);
      if (*dW && *dS) return 0;
    }
  }

  return -2;
}

/**
 * @brief Ensure a device buffer has at least nbytes capacity.
 *
 * @param pbuf Pointer to device buffer pointer.
 * @param pcap Pointer to capacity value.
 * @param nbytes Required bytes.
 * @return 0 on success, negative on failure.
 */
static int cuda_ensure_capacity(void **pbuf, size_t *pcap, size_t nbytes) {
  if (!pbuf || !pcap) return -1;
  if (*pbuf && *pcap >= nbytes) return 0;

  if (*pbuf) {
    ie_cuda_free(*pbuf);
    *pbuf = NULL;
    *pcap = 0;
  }

  void *p = ie_cuda_malloc(nbytes);
  if (!p) return -2;

  *pbuf = p;
  *pcap = nbytes;
  return 0;
}

/**
 * @brief Report CUDA device capabilities.
 */
static int cuda_caps_thunk(const void *self, ie_device_caps_t *out) {
  const cuda_impl_t *ci = (const cuda_impl_t*)self;
  if (!out || !ci) return -1;
  memset(out, 0, sizeof(*out));

  out->has_gemv_f32 = 1;
  out->has_mem_copy = 1;
  out->has_streams  = 0;

  snprintf(out->name,   sizeof(out->name),   "%s", ci->name);
  snprintf(out->driver, sizeof(out->driver), "%s", ci->driver);
  return 0;
}

/**
 * @brief CUDA memcpy implementation using the CUDA wrapper.
 *
 * This uses IE_CUDA_COPY_DEFAULT to allow the runtime to decide direction
 * under unified virtual addressing.
 */
static int cuda_memcpy_thunk(void *self, void *dst, const void *src, size_t nbytes) {
  (void)self;
  if (!dst || !src) return -1;
  if (nbytes == 0) return 0;

  if (ie_cuda_memcpy(dst, src, nbytes, IE_CUDA_COPY_DEFAULT) != 0) {
    return -2;
  }
  return 0;
}

/**
 * @brief CUDA GEMV FP32 implementation.
 *
 * This copies W/x/bias to device buffers, launches a CUDA kernel, then copies y back.
 */
static int cuda_gemv_f32_thunk(void *self,
                              const float *W,
                              const float *x,
                              float *y,
                              size_t rows,
                              size_t cols,
                              const float *bias,
                              size_t blk_k) {
  (void)blk_k;

  cuda_impl_t *ci = (cuda_impl_t*)self;
  if (!ci || !W || !x || !y || rows == 0 || cols == 0) return -1;

  const size_t W_bytes = rows * cols * sizeof(float);
  const size_t x_bytes = cols * sizeof(float);
  const size_t y_bytes = rows * sizeof(float);
  const size_t b_bytes = bias ? (rows * sizeof(float)) : 0;

  if (cuda_ensure_capacity(&ci->dW, &ci->dW_cap, W_bytes) != 0) return -2;
  if (cuda_ensure_capacity(&ci->dx, &ci->dx_cap, x_bytes) != 0) return -3;
  if (cuda_ensure_capacity(&ci->dy, &ci->dy_cap, y_bytes) != 0) return -4;
  if (bias) {
    if (cuda_ensure_capacity(&ci->dbias, &ci->dbias_cap, b_bytes) != 0) return -5;
  }

  if (ie_cuda_memcpy(ci->dW, W, W_bytes, IE_CUDA_COPY_H2D) != 0) return -6;
  if (ie_cuda_memcpy(ci->dx, x, x_bytes, IE_CUDA_COPY_H2D) != 0) return -7;
  if (bias) {
    if (ie_cuda_memcpy(ci->dbias, bias, b_bytes, IE_CUDA_COPY_H2D) != 0) return -8;
  }

  if (ie_cuda_gemv_f32((const float*)ci->dW,
                      (const float*)ci->dx,
                      (float*)ci->dy,
                      rows,
                      cols,
                      bias ? (const float*)ci->dbias : NULL) != 0) {
    return -9;
  }

  if (ie_cuda_memcpy(y, ci->dy, y_bytes, IE_CUDA_COPY_D2H) != 0) return -10;
  return 0;
}

/**
 * @brief CUDA GEMV Q4_0 implementation.
 *
 * This copies packed weights/scales/x/bias to device buffers, launches a CUDA kernel,
 * then copies y back.
 */
static int cuda_gemv_q4_0_f32_thunk(void *self,
                                   const uint8_t *w_q4,
                                   const uint8_t *w_scales,
                                   size_t scale_bytes,
                                   const float *x,
                                   float *y,
                                   size_t rows,
                                   size_t cols,
                                   const uint16_t *bias_bf16) {
  cuda_impl_t *ci = (cuda_impl_t*)self;
  if (!ci || !w_q4 || !w_scales || !x || !y || rows == 0 || cols == 0) return -1;
  if (scale_bytes != 1u && scale_bytes != 2u) return -2;
  ie_cuda_clear_last_error();

  const size_t blocks_per_row = (cols + 31u) / 32u;
  const size_t row_w_bytes = blocks_per_row * 16u;
  const size_t row_s_bytes = blocks_per_row * (size_t)scale_bytes;

  const size_t W_bytes = rows * row_w_bytes;
  const size_t S_bytes = rows * row_s_bytes;
  const size_t x_bytes = cols * sizeof(float);
  const size_t y_bytes = rows * sizeof(float);
  const size_t b_bytes = bias_bf16 ? (rows * sizeof(uint16_t)) : 0;

  if (getenv("IE_CUDA_DEBUG_MEM")) {
    size_t free_b = 0, total_b = 0;
    if (ie_cuda_mem_get_info(&free_b, &total_b) == 0) {
      ie_log_info("cuda: mem free=%zu total=%zu (W=%zu S=%zu x=%zu y=%zu bias=%zu)",
                  free_b, total_b, W_bytes, S_bytes, x_bytes, y_bytes, b_bytes);
    }
  }

  void *dW = NULL;
  void *dS = NULL;
  int blob_ok = 0;
  if (ci->blob_dev && ci->blob_host && ci->blob_bytes > 0) {
    const uint8_t *base = ci->blob_host;
    const uint8_t *end = base + ci->blob_bytes;
    const uint8_t *w_ptr = w_q4;
    const uint8_t *s_ptr = w_scales;
    if (w_ptr >= base && (w_ptr + W_bytes) <= end &&
        s_ptr >= base && (s_ptr + S_bytes) <= end) {
      const size_t w_off = (size_t)(w_ptr - base);
      const size_t s_off = (size_t)(s_ptr - base);
      dW = (uint8_t*)ci->blob_dev + w_off;
      dS = (uint8_t*)ci->blob_dev + s_off;
      blob_ok = 1;
    }
  }
  if (!blob_ok && ci->cache_cap_bytes > 0) {
    const size_t want = W_bytes + S_bytes;
    if (want <= ci->cache_cap_bytes) {
      dW = cuda_cache_get(ci, w_q4, W_bytes);
      dS = cuda_cache_get(ci, w_scales, S_bytes);
    }
  }

  if (!dW) {
    if (cuda_ensure_capacity(&ci->dW, &ci->dW_cap, W_bytes) != 0) {
      ie_log_error("cuda: q4 ensure dW failed (bytes=%zu)", W_bytes);
      return -3;
    }
  }
  if (!dS) {
    if (cuda_ensure_capacity(&ci->dW_scales, &ci->dW_scales_cap, S_bytes) != 0) {
      ie_log_error("cuda: q4 ensure dS failed (bytes=%zu)", S_bytes);
      return -4;
    }
  }
  if (cuda_ensure_capacity(&ci->dx, &ci->dx_cap, x_bytes) != 0) {
    ie_log_error("cuda: q4 ensure dx failed (bytes=%zu)", x_bytes);
    return -5;
  }
  if (cuda_ensure_capacity(&ci->dy, &ci->dy_cap, y_bytes) != 0) {
    ie_log_error("cuda: q4 ensure dy failed (bytes=%zu)", y_bytes);
    return -6;
  }
  if (bias_bf16) {
    if (cuda_ensure_capacity(&ci->dbias, &ci->dbias_cap, b_bytes) != 0) {
      ie_log_error("cuda: q4 ensure dbias failed (bytes=%zu)", b_bytes);
      return -7;
    }
  }

  if (!dW) {
    if (ie_cuda_memcpy(ci->dW, w_q4, W_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_log_error("cuda: q4 H2D W failed (bytes=%zu) err=%s",
                   W_bytes, ie_cuda_last_error_string());
      return -8;
    }
    dW = ci->dW;
  }
  if (!dS) {
    if (ie_cuda_memcpy(ci->dW_scales, w_scales, S_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_log_error("cuda: q4 H2D scales failed (bytes=%zu) err=%s",
                   S_bytes, ie_cuda_last_error_string());
      return -9;
    }
    dS = ci->dW_scales;
  }
  if (ie_cuda_memcpy(ci->dx, x, x_bytes, IE_CUDA_COPY_H2D) != 0) {
    ie_log_error("cuda: q4 H2D x failed (bytes=%zu) err=%s",
                 x_bytes, ie_cuda_last_error_string());
    return -10;
  }
  if (bias_bf16) {
    if (ie_cuda_memcpy(ci->dbias, bias_bf16, b_bytes, IE_CUDA_COPY_H2D) != 0) {
      ie_log_error("cuda: q4 H2D bias failed (bytes=%zu) err=%s",
                   b_bytes, ie_cuda_last_error_string());
      return -11;
    }
  }

  const int rc_kernel = ie_cuda_gemv_q4_0_f32((const uint8_t*)dW,
                                             (const uint8_t*)dS,
                                             scale_bytes,
                                             (const float*)ci->dx,
                                             (float*)ci->dy,
                                             rows,
                                             cols,
                                             bias_bf16 ? (const uint16_t*)ci->dbias : NULL);
  if (rc_kernel != 0) {
    char err_buf[128];
    const char *err_str = ie_cuda_last_error_string();
    if (!err_str) err_str = "unknown";
    snprintf(err_buf, sizeof(err_buf), "%s", err_str);
    if (getenv("IE_CUDA_DEBUG_MEM")) {
      size_t free_b = 0, total_b = 0;
      if (ie_cuda_mem_get_info(&free_b, &total_b) == 0) {
        ie_log_error("cuda: mem on q4 fail free=%zu total=%zu", free_b, total_b);
      }
    }
    ie_log_error("cuda: q4 kernel failed rc=%d (rows=%zu cols=%zu scale_bytes=%zu) err=%s",
                 rc_kernel, rows, cols, scale_bytes, err_buf);
    ie_log_error("cuda: q4 args dW=%p dS=%p dx=%p dy=%p dbias=%p",
                 dW, dS, ci->dx, ci->dy, bias_bf16 ? ci->dbias : NULL);
    return -12;
  }

  if (ie_cuda_memcpy(y, ci->dy, y_bytes, IE_CUDA_COPY_D2H) != 0) {
    ie_log_error("cuda: q4 D2H y failed (bytes=%zu) err=%s",
                 y_bytes, ie_cuda_last_error_string());
    return -13;
  }
  return 0;
}

/**
 * @brief CUDA block-sparse GEMV implementation.
 *
 * Not implemented yet. We return an error to avoid silently producing CPU results
 * under a CUDA label.
 */
static int cuda_gemv_block_sparse_f32_unimpl(void *self,
                                             const ie_block_sparse_matrix_t *m,
                                             const float *x,
                                             float *y,
                                             const float *bias) {
  (void)self;
  (void)m;
  (void)x;
  (void)y;
  (void)bias;
  return -2;
}

/**
 * @brief Destroy CUDA backend instance and free device buffers.
 */
static void cuda_destroy_thunk(void *self) {
  cuda_impl_t *ci = (cuda_impl_t*)self;
  if (!ci) return;

  if (ci->dW)    ie_cuda_free(ci->dW);
  if (ci->dW_scales) ie_cuda_free(ci->dW_scales);
  if (ci->dx)    ie_cuda_free(ci->dx);
  if (ci->dy)    ie_cuda_free(ci->dy);
  if (ci->dbias) ie_cuda_free(ci->dbias);
  if (ci->blob_dev) ie_cuda_free(ci->blob_dev);
  cuda_cache_clear(ci);
  if (ci->cache_cap_bytes > 0) {
    ie_log_info("cuda: cache stats hits=%llu misses=%llu evicts=%llu",
                (unsigned long long)ci->cache_hits,
                (unsigned long long)ci->cache_misses,
                (unsigned long long)ci->cache_evicts);
  }

  free(ci);
}

/**
 * @brief Initialize CUDA backend.
 *
 * This calls ie_cuda_init(), which forces CUDA runtime initialization (driver activity).
 *
 * @param out_impl Output impl pointer.
 * @param out_vt Output vtable.
 * @return 0 on success, negative on failure.
 */
static int cuda_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  if (!out_impl || !out_vt) return -1;

  if (!ie_cuda_is_available()) {
    return -2;
  }

  cuda_impl_t *ci = (cuda_impl_t*)calloc(1, sizeof(*ci));
  if (!ci) return -3;

  ci->device_ordinal = 0;

  if (ie_cuda_init(ci->device_ordinal,
                   ci->name, sizeof(ci->name),
                   ci->driver, sizeof(ci->driver)) != 0) {
    const char *err = ie_cuda_last_error_string();
    ie_log_error("cuda: init failed: %s", err ? err : "(null)");
    free(ci);
    return -4;
  }

  ci->cache_cap_bytes = cuda_cache_cap_from_env();
  if (ci->cache_cap_bytes > 0) {
    ie_log_info("cuda: weight cache enabled (cap=%.2f MiB)", (double)ci->cache_cap_bytes / (1024.0 * 1024.0));
  }

  out_vt->caps                   = cuda_caps_thunk;
  out_vt->gemv_f32               = cuda_gemv_f32_thunk;
  out_vt->gemv_q4_0_f32           = cuda_gemv_q4_0_f32_thunk;
  out_vt->memcpy                 = cuda_memcpy_thunk;
  out_vt->gemv_block_sparse_f32  = cuda_gemv_block_sparse_f32_unimpl;
  out_vt->destroy                = cuda_destroy_thunk;

  *out_impl = ci;
  return 0;
}

/* =============================================================================
 * Level Zero backend (stub)
 * ========================================================================== */

typedef struct ze_impl {
  char name[64];
  char driver[64];
} ze_impl_t;

/**
 * @brief Level Zero caps thunk used in the vtable.
 */
static int ze_caps_thunk(const void *self, ie_device_caps_t *out) {
  const ze_impl_t *zi = (const ze_impl_t*)self;
  if (!out || !zi) return -1;
  memset(out, 0, sizeof(*out));
  snprintf(out->name,   sizeof(out->name),   "%s", zi->name);
  snprintf(out->driver, sizeof(out->driver), "%s", zi->driver);
  out->has_gemv_f32 = 0;
  out->has_mem_copy = 0;
  out->has_streams  = 0;
  return 0;
}

/**
 * @brief Level Zero GEMV FP32 stub (unimplemented).
 */
static int ze_gemv_f32_stub(void *self,
                            const float *W,
                            const float *x,
                            float *y,
                            size_t rows,
                            size_t cols,
                            const float *bias,
                            size_t blk_k) {
  (void)self; (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)bias; (void)blk_k;
  return -2;
}

/**
 * @brief Level Zero GEMV Q4_0 stub (unimplemented).
 */
static int ze_gemv_q4_0_f32_stub(void *self,
                                const uint8_t *w_q4,
                                const uint8_t *w_scales,
                                size_t scale_bytes,
                                const float *x,
                                float *y,
                                size_t rows,
                                size_t cols,
                                const uint16_t *bias_bf16) {
  (void)self; (void)w_q4; (void)w_scales; (void)scale_bytes;
  (void)x; (void)y; (void)rows; (void)cols; (void)bias_bf16;
  return -2;
}

/**
 * @brief Level Zero memcpy stub (unimplemented).
 */
static int ze_memcpy_stub(void *self, void *dst, const void *src, size_t nbytes) {
  (void)self; (void)dst; (void)src; (void)nbytes;
  return -2;
}

/**
 * @brief Level Zero block-sparse GEMV FP32 stub (unimplemented).
 */
static int ze_gemv_block_sparse_f32_stub(void *self,
                                         const ie_block_sparse_matrix_t *m,
                                         const float *x,
                                         float *y,
                                         const float *bias) {
  (void)self; (void)m; (void)x; (void)y; (void)bias;
  return -2;
}

/**
 * @brief Destroy Level Zero backend instance.
 */
static void ze_destroy_thunk(void *self) {
  free(self);
}

/**
 * @brief Try to initialize Level Zero backend (stub).
 */
static int ze_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  if (!out_impl || !out_vt) return -1;

  ze_impl_t *zi = (ze_impl_t*)calloc(1, sizeof(*zi));
  if (!zi) return -2;

  snprintf(zi->name, sizeof(zi->name), "LevelZero");
  snprintf(zi->driver, sizeof(zi->driver), "stub");

  out_vt->caps                  = ze_caps_thunk;
  out_vt->gemv_f32              = ze_gemv_f32_stub;
  out_vt->gemv_q4_0_f32          = ze_gemv_q4_0_f32_stub;
  out_vt->memcpy                = ze_memcpy_stub;
  out_vt->gemv_block_sparse_f32 = ze_gemv_block_sparse_f32_stub;
  out_vt->destroy               = ze_destroy_thunk;

  *out_impl = zi;
  return 0;
}

/* =============================================================================
 * Public API
 * ========================================================================== */

/**
 * @brief Create a device backend.
 *
 * If CUDA is requested and cannot be initialized, this returns an error to avoid
 * producing CPU output labeled as CUDA.
 */
int ie_device_create(ie_device_kind_t kind, ie_device_t **out_dev) {
  if (!out_dev) return -1;

  ie_device_t *d = (ie_device_t*)calloc(1, sizeof(*d));
  if (!d) return -2;

  if (kind == IE_DEV_CPU) {
    cpu_impl_t *ci = cpu_new();
    if (!ci) {
      free(d);
      return -3;
    }
    d->impl                     = ci;
    d->vt.caps                  = cpu_caps;
    d->vt.gemv_f32              = cpu_gemv_f32;
    d->vt.gemv_q4_0_f32          = cpu_gemv_q4_0_f32;
    d->vt.memcpy                = cpu_memcpy;
    d->vt.gemv_block_sparse_f32 = cpu_gemv_block_sparse_f32;
    d->vt.destroy               = cpu_destroy;
    d->kind                     = IE_DEV_CPU;

    *out_dev = d;
    return 0;
  }

  if (kind == IE_DEV_CUDA) {
    if (cuda_try_create(&d->impl, &d->vt) != 0) {
      free(d);
      return -4;
    }
    d->kind = IE_DEV_CUDA;
    *out_dev = d;
    return 0;
  }

  if (kind == IE_DEV_ZE) {
    if (ze_try_create(&d->impl, &d->vt) != 0) {
      free(d);
      return -5;
    }
    d->kind = IE_DEV_ZE;
    *out_dev = d;
    return 0;
  }

  free(d);
  return -6;
}

/**
 * @brief Destroy a device instance.
 */
void ie_device_destroy(ie_device_t *dev) {
  if (!dev) return;
  if (dev->vt.destroy) dev->vt.destroy(dev->impl);
  free(dev);
}

/**
 * @brief Query device capabilities.
 */
int ie_device_caps(const ie_device_t *dev, ie_device_caps_t *out_caps) {
  if (!dev || !out_caps) return -1;
  return dev->vt.caps ? dev->vt.caps(dev->impl, out_caps) : -2;
}

/**
 * @brief Execute dense GEMV FP32 through backend.
 */
int ie_device_gemv_f32(ie_device_t *dev,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k) {
  if (!dev || !dev->vt.gemv_f32) return -1;
  return dev->vt.gemv_f32(dev->impl, W, x, y, rows, cols, bias, blk_k);
}

/**
 * @brief Execute GEMV Q4_0 through backend.
 */
int ie_device_gemv_q4_0_f32(ie_device_t *dev,
                            const uint8_t *w_q4,
                            const uint8_t *w_scales,
                            size_t scale_bytes,
                            const float *x,
                            float *y,
                            size_t rows,
                            size_t cols,
                            const uint16_t *bias_bf16) {
  if (!dev || !dev->vt.gemv_q4_0_f32) return -1;
  return dev->vt.gemv_q4_0_f32(dev->impl, w_q4, w_scales, scale_bytes,
                              x, y, rows, cols, bias_bf16);
}

int ie_device_q4_map(ie_device_t *dev,
                     const uint8_t *w_q4, size_t W_bytes,
                     const uint8_t *w_scales, size_t S_bytes,
                     const uint8_t **dW, const uint8_t **dS) {
  if (!dev || !w_q4 || !w_scales || !dW || !dS) return -1;
  if (dev->kind != IE_DEV_CUDA) return -2;
  cuda_impl_t *ci = (cuda_impl_t*)dev->impl;
  if (!ci) return -3;
  return cuda_q4_map(ci, w_q4, W_bytes, w_scales, S_bytes, dW, dS);
}

int ie_device_blob_ptr(ie_device_t *dev,
                        const void *host_ptr, size_t nbytes,
                        const void **out_dev_ptr) {
  if (!dev || !host_ptr || nbytes == 0 || !out_dev_ptr) return -1;
  if (dev->kind != IE_DEV_CUDA) return -2;
  cuda_impl_t *ci = (cuda_impl_t*)dev->impl;
  if (!ci || !ci->blob_dev || !ci->blob_host || ci->blob_bytes == 0) return -3;

  const uint8_t *base = (const uint8_t *)ci->blob_host;
  const uint8_t *end = base + ci->blob_bytes;
  const uint8_t *p = (const uint8_t *)host_ptr;
  if (p < base || (p + nbytes) > end) return -4;

  const size_t off = (size_t)(p - base);
  *out_dev_ptr = (const uint8_t *)ci->blob_dev + off;
  return 0;
}

int ie_device_register_blob(ie_device_t *dev, const void *host_ptr, size_t nbytes) {
  if (!dev || !host_ptr || nbytes == 0) return -1;
  if (dev->kind != IE_DEV_CUDA) return 0;
  cuda_impl_t *ci = (cuda_impl_t*)dev->impl;
  if (!ci) return -2;
  if (ci->blob_dev) return 0;
  void *d = ie_cuda_malloc(nbytes);
  if (!d) {
    ie_cuda_clear_last_error();
    return -3;
  }
  if (ie_cuda_memcpy(d, host_ptr, nbytes, IE_CUDA_COPY_H2D) != 0) {
    ie_cuda_free(d);
    ie_cuda_clear_last_error();
    return -4;
  }
  ci->blob_host = (const uint8_t *)host_ptr;
  ci->blob_dev = d;
  ci->blob_bytes = nbytes;
  ie_log_info("cuda: mirrored weights blob (%zu bytes)", nbytes);
  return 0;
}

/**
 * @brief Execute block-sparse GEMV FP32 through backend.
 */
int ie_device_gemv_block_sparse_f32(ie_device_t *dev,
                                    const ie_block_sparse_matrix_t *m,
                                    const float *x,
                                    float *y,
                                    const float *bias) {
  if (!dev || !dev->vt.gemv_block_sparse_f32) return -1;
  return dev->vt.gemv_block_sparse_f32(dev->impl, m, x, y, bias);
}

/**
 * @brief Execute memcpy through backend.
 */
int ie_device_memcpy(ie_device_t *dev, void *dst, const void *src, size_t nbytes) {
  if (!dev || !dev->vt.memcpy) return -1;
  return dev->vt.memcpy(dev->impl, dst, src, nbytes);
}

/**
 * @brief Return the device kind.
 */
ie_device_kind_t ie_device_kind(const ie_device_t *dev) {
  if (!dev) return IE_DEV_CPU;
  return dev->kind;
}

/**
 * @brief Parse a device kind from string.
 */
ie_device_kind_t ie_device_kind_from_str(const char *s) {
  if (!s) return IE_DEV_CPU;

  char tmp[16] = {0};
  size_t n = strlen(s);
  if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;

  for (size_t i = 0; i < n; ++i) {
    tmp[i] = (char)tolower((unsigned char)s[i]);
  }

  if (!strcmp(tmp, "cpu"))  return IE_DEV_CPU;
  if (!strcmp(tmp, "cuda")) return IE_DEV_CUDA;
  if (!strcmp(tmp, "ze"))   return IE_DEV_ZE;

  return IE_DEV_CPU;
}
