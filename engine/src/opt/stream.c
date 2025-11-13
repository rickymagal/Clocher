/**
 * @file stream.c
 * @brief Heuristics and implementations for prefetch and non-temporal (streaming) stores.
 *
 * Implementation highlights:
 *  - Cache size detection on Linux via sysfs:
 *      "/sys/devices/system/cpu/cpu0/cache/index<N>/size"
 *    with units "K" or "M".
 *  - Environment overrides (see ie_stream.h header doc).
 *  - Careful handling of alignment before issuing wide streaming stores.
 */

#include "ie_stream.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/types.h>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define IE_X86 1
#else
  #define IE_X86 0
#endif

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Parse a sysfs cache size file that looks like "32K\n" or "12M\n".
 *
 * @param path  NUL-terminated file path.
 * @return size in bytes (>=0), or 0 if unavailable.
 */
static size_t ie_read_cache_size_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[64] = {0};
  if (!fgets(buf, sizeof buf, f)) { fclose(f); return 0; }
  fclose(f);

  char *end = buf;
  while (*end && (*end == ' ' || *end == '\t')) end++;
  long long v = 0;
  char unit = 0;
  if (sscanf(end, "%lld%c", &v, &unit) < 1) return 0;

  size_t bytes = 0;
  if (unit == 'M' || unit == 'm') bytes = (size_t)v * 1024ull * 1024ull;
  else if (unit == 'K' || unit == 'k') bytes = (size_t)v * 1024ull;
  else bytes = (size_t)v;
  return bytes;
}

/**
 * @brief Best-effort cache discovery on Linux sysfs.
 *
 * Fills l1, l2, l3 with detected values or leaves zeros if not found.
 *
 * @param l1  out L1D size in bytes
 * @param l2  out L2 size in bytes
 * @param l3  out L3/LLC size in bytes
 */
static void ie_detect_caches_linux(size_t *l1, size_t *l2, size_t *l3) {
  /* Common indices: 0=L1D, 2=L2, 3=L3 on many systems, but vendor layouts vary. */
  const char *paths[] = {
    "/sys/devices/system/cpu/cpu0/cache/index0/size",
    "/sys/devices/system/cpu/cpu0/cache/index1/size",
    "/sys/devices/system/cpu/cpu0/cache/index2/size",
    "/sys/devices/system/cpu/cpu0/cache/index3/size",
    "/sys/devices/system/cpu/cpu0/cache/index4/size",
  };
  size_t found[5] = {0};
  for (int i = 0; i < 5; ++i) found[i] = ie_read_cache_size_file(paths[i]);

  /* Pick largest as L3, next as L2, next as L1D (very rough but robust enough). */
  size_t a[5]; memcpy(a, found, sizeof a);
  /* Simple selection sort for 5 elements. */
  for (int i = 0; i < 5; ++i) {
    for (int j = i + 1; j < 5; ++j) if (a[j] > a[i]) { size_t t = a[i]; a[i] = a[j]; a[j] = t; }
  }
  *l3 = a[0];
  *l2 = a[1];
  *l1 = a[2];
}

/**
 * @brief Clamp a double to [0.0, 1.0].
 */
static double ie_clamp01(double x) {
  if (x < 0.0) return 0.0;
  if (x > 1.0) return 1.0;
  return x;
}

/* ------------------------------- policy init ------------------------------ */

int ie_stream_policy_init(struct ie_stream_policy *p) {
  if (!p) return -EINVAL;
  memset(p, 0, sizeof *p);

  size_t l1 = 0, l2 = 0, l3 = 0;
  ie_detect_caches_linux(&l1, &l2, &l3);

  p->l1_bytes = l1;
  p->l2_bytes = l2;
  p->l3_bytes = l3;

  /* Defaults if unknown: conservative values. */
  if (p->l1_bytes == 0) p->l1_bytes = 32u * 1024u;
  if (p->l2_bytes == 0) p->l2_bytes = 1u * 1024u * 1024u;
  if (p->l3_bytes == 0) p->l3_bytes = 8u * 1024u * 1024u;

  /* Heuristic: lead ~ 6 cache lines unless overridden. */
  p->prefetch_distance = 6u * IE_CACHELINE_BYTES;

  /* Heuristic: stream when larger than 2x LLC (write-combining likely better). */
  p->nt_threshold_bytes = 2u * p->l3_bytes;

  p->prefetch_hint = 0;      /* T0 by default */
  p->enable_prefetch = true;
  p->force_nt_on = false;
  p->force_nt_off = false;

  /* Environment overrides */
  const char *s;
  if ((s = getenv("IE_STREAM_PREFETCH"))) {
    size_t v = (size_t)strtoull(s, NULL, 10);
    if (v > 0) p->prefetch_distance = v;
  }
  if ((s = getenv("IE_STREAM_NT_THRESHOLD"))) {
    size_t v = (size_t)strtoull(s, NULL, 10);
    if (v > 0) p->nt_threshold_bytes = v;
  }
  if ((s = getenv("IE_STREAM_FORCE_NT"))) {
    int v = atoi(s);
    p->force_nt_on  = (v == 1);
    p->force_nt_off = (v == 0) ? p->force_nt_off : p->force_nt_off; /* keep default */
    if (v == 0) { p->force_nt_on = false; p->force_nt_off = true; }
  }

  return 0;
}

/* -------------------------- decision / recommendation --------------------- */

bool ie_stream_should_use_nt(size_t bytes, const struct ie_stream_policy *p) {
  if (!p) return false;
  if (p->force_nt_on)  return true;
  if (p->force_nt_off) return false;
  return bytes >= p->nt_threshold_bytes;
}

size_t ie_stream_recommend_prefetch_distance(const struct ie_stream_policy *p,
                                             size_t stride_bytes) {
  if (!p) return 4u * IE_CACHELINE_BYTES;
  /* Do not lead more than ~ half L1 by default; also not less than one line. */
  size_t lead = p->prefetch_distance ? p->prefetch_distance : (6u * IE_CACHELINE_BYTES);
  if (lead < IE_CACHELINE_BYTES) lead = IE_CACHELINE_BYTES;
  if (lead > p->l1_bytes / 2u)   lead = p->l1_bytes / 2u;
  /* If strides are large, keep at least one stride ahead. */
  if (stride_bytes > 0 && lead < stride_bytes) lead = stride_bytes;
  return lead;
}

/* -------------------------------- prefetch -------------------------------- */

void ie_stream_prefetch_t0(const void *base, size_t distance) {
  const char *p = (const char *)base + distance;
#if IE_X86
  _mm_prefetch(p, _MM_HINT_T0);
#else
  (void)p;
#endif
}

void ie_stream_prefetch_nta(const void *base, size_t distance) {
  const char *p = (const char *)base + distance;
#if IE_X86
  _mm_prefetch(p, _MM_HINT_NTA);
#else
  (void)p;
#endif
}

void ie_stream_prefetch_range_t0(const void *base, size_t bytes,
                                 size_t stride, size_t distance) {
  const char *c = (const char *)base;
  size_t step = (stride > 0 ? stride : IE_CACHELINE_BYTES);
  for (size_t off = 0; off < bytes; off += step) {
    ie_stream_prefetch_t0(c + off, distance);
  }
}

void ie_stream_prefetch_range_nta(const void *base, size_t bytes,
                                  size_t stride, size_t distance) {
  const char *c = (const char *)base;
  size_t step = (stride > 0 ? stride : IE_CACHELINE_BYTES);
  for (size_t off = 0; off < bytes; off += step) {
    ie_stream_prefetch_nta(c + off, distance);
  }
}

/* ---------------------------- streaming kernels --------------------------- */

static inline void ie_scalar_copy_f32(float *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

static inline void ie_scalar_set_f32(float *dst, float v, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = v;
}

/**
 * @brief Streaming store copy (AVX-512 if available).
 */
static void ie_nt_copy_f32_avx512(float *dst, const float *src, size_t n) {
#if IE_X86 && defined(__AVX512F__)
  size_t i = 0;

  /* Prologue until destination aligns to 64B (best for 512-bit streams). */
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = addr & (64u - 1u);
  if (mis) {
    size_t pro = (64u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_copy_f32(dst, src, pro);
    dst += pro; src += pro; n -= pro;
  }

  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    _mm512_stream_ps(dst + i, v);
  }
  ie_scalar_copy_f32(dst + i, src + i, n - i);
#else
  (void)dst; (void)src; (void)n;
#endif
}

/**
 * @brief Streaming store copy (AVX2 if available).
 */
static void ie_nt_copy_f32_avx2(float *dst, const float *src, size_t n) {
#if IE_X86 && defined(__AVX2__)
  size_t i = 0;

  /* Align to 32B for 256-bit streams. */
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = addr & (32u - 1u);
  if (mis) {
    size_t pro = (32u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_copy_f32(dst, src, pro);
    dst += pro; src += pro; n -= pro;
  }

  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    _mm256_stream_ps(dst + i, v);
  }
  ie_scalar_copy_f32(dst + i, src + i, n - i);
#else
  (void)dst; (void)src; (void)n;
#endif
}

/**
 * @brief Streaming store set (AVX-512 if available).
 */
static void ie_nt_set_f32_avx512(float *dst, float value, size_t n) {
#if IE_X86 && defined(__AVX512F__)
  size_t i = 0;
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = addr & (64u - 1u);
  if (mis) {
    size_t pro = (64u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_set_f32(dst, value, pro);
    dst += pro; n -= pro;
  }

  __m512 v = _mm512_set1_ps(value);
  for (; i + 16 <= n; i += 16) _mm512_stream_ps(dst + i, v);
  ie_scalar_set_f32(dst + i, value, n - i);
#else
  (void)dst; (void)value; (void)n;
#endif
}

/**
 * @brief Streaming store set (AVX2 if available).
 */
static void ie_nt_set_f32_avx2(float *dst, float value, size_t n) {
#if IE_X86 && defined(__AVX2__)
  size_t i = 0;
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = addr & (32u - 1u);
  if (mis) {
    size_t pro = (32u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_set_f32(dst, value, pro);
    dst += pro; n -= pro;
  }

  __m256 v = _mm256_set1_ps(value);
  for (; i + 8 <= n; i += 8) _mm256_stream_ps(dst + i, v);
  ie_scalar_set_f32(dst + i, value, n - i);
#else
  (void)dst; (void)value; (void)n;
#endif
}

/* ----------------------------- public kernels ----------------------------- */

void ie_stream_copy_f32(float *dst, const float *src, size_t n,
                        const struct ie_stream_policy *p, double nt_ratio) {
  if (!dst || !src || !p) return;
  const size_t bytes = n * sizeof(float);
  const bool want_nt = ie_stream_should_use_nt(bytes, p) && (ie_clamp01(nt_ratio) > 0.0);

  if (!want_nt) {
    ie_scalar_copy_f32(dst, src, n);
    return;
  }

#if IE_X86 && defined(__AVX512F__)
  ie_nt_copy_f32_avx512(dst, src, n);
#elif IE_X86 && defined(__AVX2__)
  ie_nt_copy_f32_avx2(dst, src, n);
#else
  /* No streaming stores available at compile time. */
  ie_scalar_copy_f32(dst, src, n);
#endif
}

void ie_stream_memset_f32(float *dst, float value, size_t n,
                          const struct ie_stream_policy *p, double nt_ratio) {
  if (!dst || !p) return;
  const size_t bytes = n * sizeof(float);
  const bool want_nt = ie_stream_should_use_nt(bytes, p) && (ie_clamp01(nt_ratio) > 0.0);

  if (!want_nt) {
    ie_scalar_set_f32(dst, value, n);
    return;
  }

#if IE_X86 && defined(__AVX512F__)
  ie_nt_set_f32_avx512(dst, value, n);
#elif IE_X86 && defined(__AVX2__)
  ie_nt_set_f32_avx2(dst, value, n);
#else
  ie_scalar_set_f32(dst, value, n);
#endif
}
