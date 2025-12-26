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
} cuda_impl_t;

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
  if (ci->dx)    ie_cuda_free(ci->dx);
  if (ci->dy)    ie_cuda_free(ci->dy);
  if (ci->dbias) ie_cuda_free(ci->dbias);

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
    free(ci);
    return -4;
  }

  out_vt->caps                   = cuda_caps_thunk;
  out_vt->gemv_f32               = cuda_gemv_f32_thunk;
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
