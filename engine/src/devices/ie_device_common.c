/**
 * @file ie_device_common.c
 * @brief C11 implementation of the device abstraction with CPU backend and
 *        CUDA/Level Zero stubs resolved via dlopen (graceful fallback).
 */

#include "ie_device.h"
#include "ie_kernels.h"
#include "sparse_format.h"

#include <ctype.h>
#include <dlfcn.h>
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
 * CPU backend always supports GEMV FP32 and memcpy.
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
 * Shared: utility to try opening a dynamic library
 * ========================================================================== */

/**
 * @brief Try opening one of a list of shared objects.
 *
 * @param names NULL-terminated list of soname strings.
 * @return Handle from dlopen() or NULL if none succeeded.
 */
static void *try_open_any(const char *const *names) {
  for (size_t i = 0; names[i]; ++i) {
    void *h = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

/* =============================================================================
 * CUDA backend (stub)
 * ========================================================================== */

typedef struct cuda_impl {
  void *h_cuda;
  char name[64];
  char driver[64];
} cuda_impl_t;

/**
 * @brief CUDA caps thunk used in the vtable.
 */
static int cuda_caps_thunk(const void *self, ie_device_caps_t *out) {
  const cuda_impl_t *ci = (const cuda_impl_t*)self;
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  snprintf(out->name,   sizeof(out->name),   "%s", ci->name);
  snprintf(out->driver, sizeof(out->driver), "%s", ci->driver);
  /* This is a stub backend: no real GEMV implementation yet. */
  out->has_gemv_f32 = 0;
  out->has_mem_copy = 0;
  out->has_streams  = 0;
  return 0;
}

/**
 * @brief CUDA GEMV FP32 stub (unimplemented).
 */
static int cuda_gemv_f32_stub(void *self,
                              const float *W,
                              const float *x,
                              float *y,
                              size_t rows,
                              size_t cols,
                              const float *bias,
                              size_t blk_k) {
  (void)self;
  (void)W;
  (void)x;
  (void)y;
  (void)rows;
  (void)cols;
  (void)bias;
  (void)blk_k;
  return -2; /* unimplemented */
}

/**
 * @brief CUDA memcpy stub (unimplemented).
 */
static int cuda_memcpy_stub(void *self,
                            void *dst,
                            const void *src,
                            size_t nbytes) {
  (void)self;
  (void)dst;
  (void)src;
  (void)nbytes;
  return -2; /* unimplemented */
}

/**
 * @brief CUDA block-sparse GEMV FP32 stub (unimplemented).
 */
static int cuda_gemv_block_sparse_f32_stub(void *self,
                                           const ie_block_sparse_matrix_t *m,
                                           const float *x,
                                           float *y,
                                           const float *bias) {
  (void)self;
  (void)m;
  (void)x;
  (void)y;
  (void)bias;
  return -2; /* unimplemented */
}

/**
 * @brief Destroy CUDA backend instance.
 */
static void cuda_destroy_thunk(void *self) {
  cuda_impl_t *ci = (cuda_impl_t*)self;
  if (ci->h_cuda) dlclose(ci->h_cuda);
  free(ci);
}

/**
 * @brief Try to initialize CUDA backend via dlopen.
 */
static int cuda_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  static const char *candidates[] = {"libcuda.so.1", "libcuda.so", NULL};
  void *h = try_open_any(candidates);
  if (!h) return -1;

  cuda_impl_t *ci = (cuda_impl_t*)calloc(1, sizeof(*ci));
  if (!ci) {
    dlclose(h);
    return -2;
  }

  ci->h_cuda = h;
  snprintf(ci->name, sizeof(ci->name), "CUDA");
  snprintf(ci->driver, sizeof(ci->driver), "libcuda");

  out_vt->caps                  = cuda_caps_thunk;
  out_vt->gemv_f32              = cuda_gemv_f32_stub;
  out_vt->memcpy                = cuda_memcpy_stub;
  out_vt->gemv_block_sparse_f32 = cuda_gemv_block_sparse_f32_stub;
  out_vt->destroy               = cuda_destroy_thunk;

  *out_impl = ci;
  return 0;
}

/* =============================================================================
 * Level Zero backend (stub)
 * ========================================================================== */

typedef struct ze_impl {
  void *h_ze;
  char name[64];
  char driver[64];
} ze_impl_t;

/**
 * @brief Level Zero caps thunk used in the vtable.
 */
static int ze_caps_thunk(const void *self, ie_device_caps_t *out) {
  const ze_impl_t *zi = (const ze_impl_t*)self;
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  snprintf(out->name,   sizeof(out->name),   "%s", zi->name);
  snprintf(out->driver, sizeof(out->driver), "%s", zi->driver);
  /* Stub backend: no real GEMV implementation. */
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
  (void)self;
  (void)W;
  (void)x;
  (void)y;
  (void)rows;
  (void)cols;
  (void)bias;
  (void)blk_k;
  return -2;
}

/**
 * @brief Level Zero memcpy stub (unimplemented).
 */
static int ze_memcpy_stub(void *self,
                          void *dst,
                          const void *src,
                          size_t nbytes) {
  (void)self;
  (void)dst;
  (void)src;
  (void)nbytes;
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
  (void)self;
  (void)m;
  (void)x;
  (void)y;
  (void)bias;
  return -2;
}

/**
 * @brief Destroy Level Zero backend instance.
 */
static void ze_destroy_thunk(void *self) {
  ze_impl_t *zi = (ze_impl_t*)self;
  if (zi->h_ze) dlclose(zi->h_ze);
  free(zi);
}

/**
 * @brief Try to initialize Level Zero backend via dlopen.
 */
static int ze_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  static const char *candidates[] = {"libze_loader.so.1", "libze_loader.so", NULL};
  void *h = try_open_any(candidates);
  if (!h) return -1;

  ze_impl_t *zi = (ze_impl_t*)calloc(1, sizeof(*zi));
  if (!zi) {
    dlclose(h);
    return -2;
  }

  zi->h_ze = h;
  snprintf(zi->name, sizeof(zi->name), "LevelZero");
  snprintf(zi->driver, sizeof(zi->driver), "ze_loader");

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
 * @brief Create a device backend. Falls back to CPU if GPU backend fails.
 */
int ie_device_create(ie_device_kind_t kind, ie_device_t **out_dev) {
  if (!out_dev) return -1;

  ie_device_t *d = (ie_device_t*)calloc(1, sizeof(*d));
  if (!d) return -2;

  int ok = -1;

  if (kind == IE_DEV_CUDA) ok = cuda_try_create(&d->impl, &d->vt);
  else if (kind == IE_DEV_ZE) ok = ze_try_create(&d->impl, &d->vt);

  if (ok != 0) {
    cpu_impl_t *ci = cpu_new();
    if (!ci) {
      free(d);
      return -3;
    }
    d->impl                    = ci;
    d->vt.caps                 = cpu_caps;
    d->vt.gemv_f32             = cpu_gemv_f32;
    d->vt.memcpy               = cpu_memcpy;
    d->vt.gemv_block_sparse_f32 = cpu_gemv_block_sparse_f32;
    d->vt.destroy              = cpu_destroy;
    d->kind                    = IE_DEV_CPU;
  } else {
    d->kind = kind;
  }

  *out_dev = d;
  return 0;
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
 * @brief Execute dense GEMV FP32 through backend or CPU fallback.
 */
int ie_device_gemv_f32(ie_device_t *dev,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k) {
  if (!dev || !dev->vt.gemv_f32) return -1;
  int rc = dev->vt.gemv_f32(dev->impl, W, x, y, rows, cols, bias, blk_k);
  if (rc == 0) return 0;

  /* Fallback to CPU if backend is stubbed */
  if (dev->kind != IE_DEV_CPU) {
    ie_device_t *cpu = NULL;
    if (ie_device_create(IE_DEV_CPU, &cpu) == 0) {
      int rc2 = ie_device_gemv_f32(cpu, W, x, y, rows, cols, bias, blk_k);
      ie_device_destroy(cpu);
      return rc2;
    }
  }
  return rc;
}

/**
 * @brief Execute block-sparse GEMV FP32 or CPU fallback.
 */
int ie_device_gemv_block_sparse_f32(ie_device_t *dev,
                                    const ie_block_sparse_matrix_t *m,
                                    const float *x,
                                    float *y,
                                    const float *bias) {
  if (!dev || !dev->vt.gemv_block_sparse_f32) return -1;
  int rc = dev->vt.gemv_block_sparse_f32(dev->impl, m, x, y, bias);
  if (rc == 0) return 0;

  /* Fallback */
  if (dev->kind != IE_DEV_CPU) {
    ie_device_t *cpu = NULL;
    if (ie_device_create(IE_DEV_CPU, &cpu) == 0) {
      int rc2 = ie_device_gemv_block_sparse_f32(cpu, m, x, y, bias);
      ie_device_destroy(cpu);
      return rc2;
    }
  }
  return rc;
}

/**
 * @brief Execute memcpy via backend or CPU fallback.
 */
int ie_device_memcpy(ie_device_t *dev, void *dst, const void *src, size_t nbytes) {
  if (!dev || !dev->vt.memcpy) return -1;
  int rc = dev->vt.memcpy(dev->impl, dst, src, nbytes);
  if (rc == 0) return 0;

  if (dev->kind != IE_DEV_CPU) {
    ie_device_t *cpu = NULL;
    if (ie_device_create(IE_DEV_CPU, &cpu) == 0) {
      int rc2 = ie_device_memcpy(cpu, dst, src, nbytes);
      ie_device_destroy(cpu);
      return rc2;
    }
  }
  return rc;
}

/**
 * @brief Parse a device kind from string.
 */
ie_device_kind_t ie_device_kind_from_str(const char *s) {
  if (!s) return IE_DEV_CPU;
  char tmp[16] = {0};
  size_t n = strlen(s);
  if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
  for (size_t i = 0; i < n; ++i)
    tmp[i] = (char)tolower((unsigned char)s[i]);
  if (!strcmp(tmp, "cpu"))  return IE_DEV_CPU;
  if (!strcmp(tmp, "cuda")) return IE_DEV_CUDA;
  if (!strcmp(tmp, "ze"))   return IE_DEV_ZE;
  return IE_DEV_CPU;
}
