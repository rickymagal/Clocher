/**
 * @file ie_device_common.c
 * @brief C11 implementation of the device abstraction with CPU backend and
 *        CUDA/Level Zero stubs resolved via dlopen (graceful fallback).
 */
#include "ie_device.h"
#include "ie_kernels.h"  /* cpu gemv entry points */

#include <ctype.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================== VTable ================================== */

/**
 * @brief Method table for a device backend.
 */
typedef struct ie_device_vtbl {
  int  (*caps)(const void *self, ie_device_caps_t *out);
  int  (*gemv_f32)(void *self, const float *W, const float *x, float *y,
                   size_t rows, size_t cols, const float *bias, size_t blk_k);
  int  (*memcpy)(void *self, void *dst, const void *src, size_t nbytes);
  void (*destroy)(void *self);
} ie_device_vtbl_t;

/**
 * @brief Concrete device object combining vtable and private impl.
 */
struct ie_device {
  ie_device_vtbl_t vt;    /**< Virtual method table. */
  void *impl;             /**< Backend-specific implementation pointer. */
  ie_device_kind_t kind;  /**< Kind requested or resolved. */
};

/* ============================ Declarations ============================== */

/* CPU backend */
typedef struct cpu_impl {
  char name[64];
} cpu_impl_t;

static cpu_impl_t *cpu_new(void);
static int  cpu_caps(const void *self, ie_device_caps_t *out);
static int  cpu_gemv_f32(void *self, const float *W, const float *x, float *y,
                         size_t rows, size_t cols, const float *bias, size_t blk_k);
static int  cpu_memcpy(void *self, void *dst, const void *src, size_t nbytes);
static void cpu_destroy(void *self);

/* CUDA backend (dlopen stubs) */
typedef struct cuda_impl {
  void *h_cuda;     /* libcuda handle */
  char  name[64];
  char  driver[64];
} cuda_impl_t;

static int  cuda_try_create(void **out_impl, ie_device_vtbl_t *out_vt);
static int  cuda_caps_c(const void *self, ie_device_caps_t *out);
static int  cuda_gemv_c(void *self, const float *W, const float *x, float *y,
                        size_t rows, size_t cols, const float *bias, size_t blk_k);
static int  cuda_memcpy_c(void *self, void *dst, const void *src, size_t n);
static void cuda_destroy_c(void *self);

/* Level Zero backend (dlopen stubs) */
typedef struct ze_impl {
  void *h_ze;       /* ze_loader handle */
  char  name[64];
  char  driver[64];
} ze_impl_t;

static int  ze_try_create(void **out_impl, ie_device_vtbl_t *out_vt);
static int  ze_caps_c(const void *self, ie_device_caps_t *out);
static int  ze_gemv_c(void *self, const float *W, const float *x, float *y,
                      size_t rows, size_t cols, const float *bias, size_t blk_k);
static int  ze_memcpy_c(void *self, void *dst, const void *src, size_t n);
static void ze_destroy_c(void *self);

/* Utilities */
static void *try_open_any(const char *const *names);

/* ============================== Utilities =============================== */

/**
 * @brief Try to open one of the candidate shared libraries using dlopen.
 *
 * @param names NULL-terminated array of soname candidates.
 * @return dlopen handle on success; NULL if none can be opened.
 */
static void *try_open_any(const char *const *names) {
  for (size_t i = 0; names[i]; ++i) {
    void *h = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

/**
 * @brief Parse a device kind from a string (case-insensitive).
 *
 * Accepts "cpu", "cuda", "ze". Unknown strings return IE_DEV_CPU.
 *
 * @param s Input string (may be NULL).
 * @return Parsed device kind.
 */
ie_device_kind_t ie_device_kind_from_str(const char *s) {
  if (!s || !*s) return IE_DEV_CPU;
  char buf[16] = {0};
  size_t n = strlen(s);
  if (n >= sizeof(buf)) n = sizeof(buf) - 1;
  for (size_t i = 0; i < n; ++i) buf[i] = (char)tolower((unsigned char)s[i]);
  if (strcmp(buf, "cpu") == 0)  return IE_DEV_CPU;
  if (strcmp(buf, "cuda") == 0) return IE_DEV_CUDA;
  if (strcmp(buf, "ze") == 0)   return IE_DEV_ZE;
  return IE_DEV_CPU;
}

/* =============================== CPU Impl =============================== */

/**
 * @brief Allocate and initialize the CPU backend implementation.
 *
 * @return Newly allocated cpu_impl_t pointer or NULL on OOM.
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
 * @param self Implementation pointer (unused).
 * @param out Output capabilities structure.
 * @return 0 on success; non-zero on error.
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
 * @brief CPU implementation of GEMV FP32 via existing kernels.
 *
 * @param self Implementation pointer (unused).
 * @param W Row-major FP32 weights matrix.
 * @param x FP32 input vector.
 * @param y FP32 output vector.
 * @param rows Number of rows in W (and y).
 * @param cols Number of columns in W (and x).
 * @param bias Optional FP32 bias (NULL if none).
 * @param blk_k Optional blocked-K hint (ignored by CPU).
 * @return 0 on success; non-zero on invalid parameters.
 */
static int cpu_gemv_f32(void *self,
                        const float *W, const float *x, float *y,
                        size_t rows, size_t cols,
                        const float *bias, size_t blk_k) {
  (void)self; (void)blk_k;
  if (!W || !x || !y || rows == 0 || cols == 0) return -1;
  ie_gemv_f32(W, x, y, rows, cols, bias, 0);
  return 0;
}

/**
 * @brief CPU implementation of memcpy wrapper.
 *
 * @param self Implementation pointer (unused).
 * @param dst Destination buffer.
 * @param src Source buffer.
 * @param nbytes Number of bytes to copy.
 * @return 0 on success; non-zero on invalid parameters.
 */
static int cpu_memcpy(void *self, void *dst, const void *src, size_t nbytes) {
  (void)self;
  if (!dst || !src || nbytes == 0) return 0;
  memcpy(dst, src, nbytes);
  return 0;
}

/**
 * @brief Destroy CPU implementation object.
 *
 * @param self Implementation pointer returned by cpu_new().
 */
static void cpu_destroy(void *self) {
  free(self);
}

/* =============================== CUDA Impl ============================== */

/**
 * @brief Attempt to construct a CUDA backend via dlopen (stub).
 *
 * Resolves libcuda only; kernels are not implemented yet. The device object
 * will exist, but all compute methods return unimplemented until kernels land.
 *
 * @param out_impl Output: implementation pointer on success.
 * @param out_vt Output: vtable populated with CUDA stubs.
 * @return 0 on success; non-zero if CUDA cannot be initialized.
 */
static int cuda_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  static const char *cand_cuda[] = {"libcuda.so.1", "libcuda.so", NULL};
  void *h = try_open_any(cand_cuda);
  if (!h) return -1;

  cuda_impl_t *p = (cuda_impl_t*)calloc(1, sizeof(*p));
  if (!p) { dlclose(h); return -2; }
  p->h_cuda = h;
  snprintf(p->name, sizeof(p->name), "CUDA");
  snprintf(p->driver, sizeof(p->driver), "libcuda");

  out_vt->caps     = cuda_caps_c;
  out_vt->gemv_f32 = cuda_gemv_c;
  out_vt->memcpy   = cuda_memcpy_c;
  out_vt->destroy  = cuda_destroy_c;

  *out_impl = p;
  return 0;
}

/**
 * @brief Report CUDA capabilities (stub: unimplemented compute).
 *
 * @param self CUDA implementation pointer.
 * @param out Output capabilities structure.
 * @return 0 on success; non-zero on error.
 */
static int cuda_caps_c(const void *self, ie_device_caps_t *out) {
  const cuda_impl_t *ci = (const cuda_impl_t*)self;
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  out->has_gemv_f32 = 0;
  out->has_mem_copy = 0;
  out->has_streams  = 0;
  snprintf(out->name,   sizeof(out->name),   "%s", ci->name);
  snprintf(out->driver, sizeof(out->driver), "%s", ci->driver);
  return 0;
}

/**
 * @brief CUDA GEMV stub (returns unimplemented).
 *
 * @return Always -2 until kernels are implemented.
 */
static int cuda_gemv_c(void *self,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k) {
  (void)self; (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)bias; (void)blk_k;
  return -2;
}

/**
 * @brief CUDA memcpy stub (returns unimplemented).
 *
 * @return Always -2 until copies are implemented.
 */
static int cuda_memcpy_c(void *self, void *dst, const void *src, size_t n) {
  (void)self; (void)dst; (void)src; (void)n;
  return -2;
}

/**
 * @brief Destroy CUDA implementation and close libcuda handle.
 *
 * @param self CUDA implementation pointer.
 */
static void cuda_destroy_c(void *self) {
  cuda_impl_t *ci = (cuda_impl_t*)self;
  if (ci->h_cuda) dlclose(ci->h_cuda);
  free(ci);
}

/* ============================== Level Zero ============================== */

/**
 * @brief Attempt to construct a Level Zero backend via dlopen (stub).
 *
 * Resolves ze_loader only; kernels are not implemented yet.
 *
 * @param out_impl Output: implementation pointer on success.
 * @param out_vt Output: vtable populated with ZE stubs.
 * @return 0 on success; non-zero if ZE cannot be initialized.
 */
static int ze_try_create(void **out_impl, ie_device_vtbl_t *out_vt) {
  static const char *cand_ze[] = {"libze_loader.so.1", "libze_loader.so", NULL};
  void *h = try_open_any(cand_ze);
  if (!h) return -1;

  ze_impl_t *p = (ze_impl_t*)calloc(1, sizeof(*p));
  if (!p) { dlclose(h); return -2; }
  p->h_ze = h;
  snprintf(p->name, sizeof(p->name), "LevelZero");
  snprintf(p->driver, sizeof(p->driver), "ze_loader");

  out_vt->caps     = ze_caps_c;
  out_vt->gemv_f32 = ze_gemv_c;
  out_vt->memcpy   = ze_memcpy_c;
  out_vt->destroy  = ze_destroy_c;

  *out_impl = p;
  return 0;
}

/**
 * @brief Report Level Zero capabilities (stub: unimplemented compute).
 *
 * @param self ZE implementation pointer.
 * @param out Output capabilities structure.
 * @return 0 on success; non-zero on error.
 */
static int ze_caps_c(const void *self, ie_device_caps_t *out) {
  const ze_impl_t *zi = (const ze_impl_t*)self;
  if (!out) return -1;
  memset(out, 0, sizeof(*out));
  out->has_gemv_f32 = 0;
  out->has_mem_copy = 0;
  out->has_streams  = 0;
  snprintf(out->name,   sizeof(out->name),   "%s", zi->name);
  snprintf(out->driver, sizeof(out->driver), "%s", zi->driver);
  return 0;
}

/**
 * @brief Level Zero GEMV stub (returns unimplemented).
 *
 * @return Always -2 until kernels are implemented.
 */
static int ze_gemv_c(void *self,
                     const float *W, const float *x, float *y,
                     size_t rows, size_t cols,
                     const float *bias, size_t blk_k) {
  (void)self; (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)bias; (void)blk_k;
  return -2;
}

/**
 * @brief Level Zero memcpy stub (returns unimplemented).
 *
 * @return Always -2 until copies are implemented.
 */
static int ze_memcpy_c(void *self, void *dst, const void *src, size_t n) {
  (void)self; (void)dst; (void)src; (void)n;
  return -2;
}

/**
 * @brief Destroy Level Zero implementation and close loader handle.
 *
 * @param self ZE implementation pointer.
 */
static void ze_destroy_c(void *self) {
  ze_impl_t *zi = (ze_impl_t*)self;
  if (zi->h_ze) dlclose(zi->h_ze);
  free(zi);
}

/* ============================== Public API ============================== */

/**
 * @brief Create a device for the requested kind, with CPU fallback.
 *
 * @param kind Requested kind.
 * @param out_dev Output: allocated device handle on success.
 * @return 0 on success; non-zero on error.
 */
int ie_device_create(ie_device_kind_t kind, ie_device_t **out_dev) {
  if (!out_dev) return -1;

  ie_device_t *d = (ie_device_t*)calloc(1, sizeof(*d));
  if (!d) return -2;

  int ok = -1;
  if (kind == IE_DEV_CUDA) {
    ok = cuda_try_create(&d->impl, &d->vt);
  } else if (kind == IE_DEV_ZE) {
    ok = ze_try_create(&d->impl, &d->vt);
  }

  if (ok != 0) {
    /* Fallback to CPU */
    cpu_impl_t *ci = cpu_new();
    if (!ci) { free(d); return -3; }
    d->impl = ci;
    d->vt.caps     = cpu_caps;
    d->vt.gemv_f32 = cpu_gemv_f32;
    d->vt.memcpy   = cpu_memcpy;
    d->vt.destroy  = cpu_destroy;
    d->kind        = IE_DEV_CPU;
  } else {
    d->kind = kind;
  }

  *out_dev = d;
  return 0;
}

/**
 * @brief Destroy a device handle and its backend implementation.
 *
 * @param dev Device handle created by ie_device_create().
 */
void ie_device_destroy(ie_device_t *dev) {
  if (!dev) return;
  if (dev->vt.destroy) dev->vt.destroy(dev->impl);
  free(dev);
}

/**
 * @brief Retrieve capabilities from a device backend.
 *
 * @param dev Device handle.
 * @param out_caps Output capabilities structure (non-NULL).
 * @return 0 on success; non-zero on error.
 */
int ie_device_caps(const ie_device_t *dev, ie_device_caps_t *out_caps) {
  if (!dev || !out_caps) return -1;
  return dev->vt.caps ? dev->vt.caps(dev->impl, out_caps) : -2;
}

/**
 * @brief Execute GEMV on the device, with CPU fallback if unimplemented.
 *
 * @param dev Device handle.
 * @param W Row-major weights.
 * @param x Input vector.
 * @param y Output vector.
 * @param rows Rows in W / length of y.
 * @param cols Cols in W / length of x.
 * @param bias Optional bias or NULL.
 * @param blk_k Optional blocked-K hint (backend-specific).
 * @return 0 on success; non-zero on error.
 */
int ie_device_gemv_f32(ie_device_t *dev,
                       const float *W, const float *x, float *y,
                       size_t rows, size_t cols,
                       const float *bias, size_t blk_k) {
  if (!dev || !dev->vt.gemv_f32) return -1;
  int rc = dev->vt.gemv_f32(dev->impl, W, x, y, rows, cols, bias, blk_k);
  if (rc == 0) return 0;

  /* If GPU backend is stubbed, transparently fall back to CPU once. */
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
 * @brief Perform a memcpy using the device backend, with CPU fallback.
 *
 * @param dev Device handle.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param nbytes Number of bytes to copy.
 * @return 0 on success; non-zero on error.
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
