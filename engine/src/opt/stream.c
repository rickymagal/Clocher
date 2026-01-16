/* ============================================================================
 * File: engine/src/opt/stream.c
 * ============================================================================
 */
/**
 * @file stream.c
 * @brief Prefetch and non-temporal (streaming) heuristics and helpers.
 *
 * @details
 * This module implements:
 *  - Best-effort cache size discovery on Linux via sysfs:
 *      "/sys/devices/system/cpu/cpu0/cache/index<N>/size"
 *    with size strings like "32K" or "12M".
 *  - Policy initialization and environment overrides (documented in ie_stream.h).
 *  - Prefetch helpers (T0 and NTA) with graceful no-op on non-x86.
 *  - Optional non-temporal copy/memset helpers for float arrays, using AVX-512F
 *    or AVX2 when available, with safe scalar fallback.
 *
 * Runtime dispatch:
 *  - Feature detection is performed once via ie_cpu_features_detect() and cached
 *    with pthread_once().
 *  - AVX-512F / AVX2 implementations are compiled using function-targeting
 *    (__attribute__((target(...)))) so the translation unit can be built with
 *    baseline flags while still containing optimized helpers.
 *
 * Environment variables (interpreted here):
 *  - IE_STREAM_PREFETCH        Override prefetch lead distance (bytes).
 *  - IE_STREAM_NT_THRESHOLD    Override non-temporal threshold (bytes).
 *  - IE_STREAM_FORCE_NT        "1" => always NT, "0" => never NT, else heuristic.
 */

#define _POSIX_C_SOURCE 200809L

#include "ie_stream.h"
#include "ie_cpu.h"

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  #define IE_X86 1
#else
  #define IE_X86 0
#endif

#if IE_X86
  #if defined(__GNUC__) || defined(__clang__)
    #define IE_TARGET_AVX2    __attribute__((target("avx2,fma,sse3")))
    #define IE_TARGET_AVX512F __attribute__((target("avx512f")))
  #else
    #define IE_TARGET_AVX2
    #define IE_TARGET_AVX512F
  #endif
#endif

/* --------------------------------- helpers -------------------------------- */

/**
 * @brief Parse a sysfs cache size file containing values like "32K\n" or "12M\n".
 *
 * @details
 * Sysfs cache size strings commonly encode units as K/M. This helper parses the
 * numeric prefix and optional unit suffix, returning bytes.
 *
 * If the file cannot be opened/read or parsing fails, returns 0.
 *
 * @param path NUL-terminated file path.
 * @return Cache size in bytes (>=0), or 0 if unavailable/unreadable.
 */
static size_t ie_read_cache_size_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;

  char buf[64] = {0};
  if (!fgets(buf, sizeof buf, f)) {
    fclose(f);
    return 0;
  }
  fclose(f);

  char *p = buf;
  while (*p && (*p == ' ' || *p == '\t')) p++;

  long long v = 0;
  char unit = 0;
  if (sscanf(p, "%lld%c", &v, &unit) < 1) return 0;
  if (v <= 0) return 0;

  if (unit == 'M' || unit == 'm') return (size_t)v * 1024ull * 1024ull;
  if (unit == 'K' || unit == 'k') return (size_t)v * 1024ull;
  return (size_t)v;
}

/**
 * @brief Best-effort cache discovery on Linux sysfs.
 *
 * @details
 * Vendor sysfs layouts vary; we read a small fixed set of potential index paths
 * and rank the discovered sizes. The mapping "largest => L3, second => L2,
 * third => L1" is a rough approximation but works reasonably well for tuning
 * thresholds without hard dependencies.
 *
 * If nothing is found, outputs remain zero.
 *
 * @param l1 Out: L1D size in bytes (best-effort).
 * @param l2 Out: L2 size in bytes (best-effort).
 * @param l3 Out: L3/LLC size in bytes (best-effort).
 */
static void ie_detect_caches_linux(size_t *l1, size_t *l2, size_t *l3) {
  if (l1) *l1 = 0;
  if (l2) *l2 = 0;
  if (l3) *l3 = 0;

  const char *paths[] = {
    "/sys/devices/system/cpu/cpu0/cache/index0/size",
    "/sys/devices/system/cpu/cpu0/cache/index1/size",
    "/sys/devices/system/cpu/cpu0/cache/index2/size",
    "/sys/devices/system/cpu/cpu0/cache/index3/size",
    "/sys/devices/system/cpu/cpu0/cache/index4/size",
  };

  size_t found[5] = {0};
  for (int i = 0; i < 5; ++i) found[i] = ie_read_cache_size_file(paths[i]);

  /* Sort descending (N=5 selection sort). */
  for (int i = 0; i < 5; ++i) {
    for (int j = i + 1; j < 5; ++j) {
      if (found[j] > found[i]) {
        size_t t = found[i];
        found[i] = found[j];
        found[j] = t;
      }
    }
  }

  if (l3) *l3 = found[0];
  if (l2) *l2 = found[1];
  if (l1) *l1 = found[2];
}

/**
 * @brief Clamp a double to [0.0, 1.0].
 *
 * @param x Input value.
 * @return Clamped value in [0,1].
 */
static double ie_clamp01(double x) {
  if (x < 0.0) return 0.0;
  if (x > 1.0) return 1.0;
  return x;
}

/* -------------------------- cached CPU feature flags ----------------------- */

#if IE_X86
static pthread_once_t g_ie_stream_cpu_once = PTHREAD_ONCE_INIT;
static struct ie_cpu_features g_ie_stream_cpu = {0};

static void ie_stream_detect_cpu_once(void) {
  ie_cpu_features_detect(&g_ie_stream_cpu);
}

static const struct ie_cpu_features *ie_stream_cpu_features(void) {
  (void)pthread_once(&g_ie_stream_cpu_once, ie_stream_detect_cpu_once);
  return &g_ie_stream_cpu;
}
#endif

/* ------------------------------- policy init ------------------------------ */

/**
 * @brief Initialize an ie_stream_policy with detected cache sizes and defaults.
 *
 * @details
 * Cache detection is best-effort (Linux sysfs). If detection fails, conservative
 * defaults are used:
 *  - L1D: 32 KiB
 *  - L2:  1 MiB
 *  - L3:  8 MiB
 *
 * Defaults:
 *  - prefetch_distance: 6 cache lines
 *  - nt_threshold_bytes: 2x LLC
 *
 * Env overrides:
 *  - IE_STREAM_PREFETCH
 *  - IE_STREAM_NT_THRESHOLD
 *  - IE_STREAM_FORCE_NT
 *
 * @param p Output policy to initialize.
 * @return 0 on success, or -EINVAL on invalid arguments.
 */
int ie_stream_policy_init(struct ie_stream_policy *p) {
  if (!p) return -EINVAL;
  memset(p, 0, sizeof *p);

  size_t l1 = 0, l2 = 0, l3 = 0;
  ie_detect_caches_linux(&l1, &l2, &l3);

  p->l1_bytes = l1;
  p->l2_bytes = l2;
  p->l3_bytes = l3;

  if (p->l1_bytes == 0) p->l1_bytes = 32u * 1024u;
  if (p->l2_bytes == 0) p->l2_bytes = 1u * 1024u * 1024u;
  if (p->l3_bytes == 0) p->l3_bytes = 8u * 1024u * 1024u;

  p->prefetch_distance = 6u * IE_CACHELINE_BYTES;
  p->nt_threshold_bytes = 2u * p->l3_bytes;

  p->prefetch_hint = 0;
  p->enable_prefetch = true;
  p->force_nt_on = false;
  p->force_nt_off = false;

  const char *s = NULL;

  s = getenv("IE_STREAM_PREFETCH");
  if (s && *s) {
    size_t v = (size_t)strtoull(s, NULL, 10);
    if (v > 0) p->prefetch_distance = v;
  }

  s = getenv("IE_STREAM_NT_THRESHOLD");
  if (s && *s) {
    size_t v = (size_t)strtoull(s, NULL, 10);
    if (v > 0) p->nt_threshold_bytes = v;
  }

  /* IE_STREAM_FORCE_NT:
   *  - "1" => always use non-temporal stores
   *  - "0" => never use non-temporal stores
   *  - otherwise => ignore (keep heuristic)
   */
  s = getenv("IE_STREAM_FORCE_NT");
  if (s && *s) {
    int v = atoi(s);
    if (v == 1) {
      p->force_nt_on = true;
      p->force_nt_off = false;
    } else if (v == 0) {
      p->force_nt_on = false;
      p->force_nt_off = true;
    }
  }

  return 0;
}

/* -------------------------- decision / recommendation --------------------- */

/**
 * @brief Decide whether a memory operation of @p bytes should use non-temporal stores.
 *
 * @details
 * The decision uses the policy threshold unless forced on/off via
 * IE_STREAM_FORCE_NT.
 *
 * @param bytes Size of the memory operation in bytes.
 * @param p Policy (must be non-NULL).
 * @return true if non-temporal stores are recommended, else false.
 */
bool ie_stream_should_use_nt(size_t bytes, const struct ie_stream_policy *p) {
  if (!p) return false;
  if (p->force_nt_on) return true;
  if (p->force_nt_off) return false;
  return bytes >= p->nt_threshold_bytes;
}

/**
 * @brief Recommend a prefetch lead distance for a given stride.
 *
 * @details
 * This clamps the configured distance to a practical range:
 *  - at least one cache line,
 *  - at most half of L1,
 *  - at least stride_bytes when stride_bytes is provided.
 *
 * @param p Policy (may be NULL).
 * @param stride_bytes Access stride in bytes (0 => assume cacheline stride).
 * @return Recommended prefetch lead distance in bytes.
 */
size_t ie_stream_recommend_prefetch_distance(const struct ie_stream_policy *p,
                                             size_t stride_bytes) {
  if (!p) return 4u * IE_CACHELINE_BYTES;

  size_t lead = p->prefetch_distance ? p->prefetch_distance
                                     : (6u * IE_CACHELINE_BYTES);

  if (lead < IE_CACHELINE_BYTES) lead = IE_CACHELINE_BYTES;
  if (lead > p->l1_bytes / 2u) lead = p->l1_bytes / 2u;

  if (stride_bytes > 0 && lead < stride_bytes) lead = stride_bytes;

  return lead;
}

/* -------------------------------- prefetch -------------------------------- */

/**
 * @brief Issue a temporal (T0) prefetch at base+distance (best-effort).
 *
 * @details
 * On non-x86 builds this function is a no-op.
 *
 * @param base Base pointer.
 * @param distance Byte offset from base to prefetch.
 */
void ie_stream_prefetch_t0(const void *base, size_t distance) {
  const char *p = (const char *)base + distance;
#if IE_X86
  _mm_prefetch(p, _MM_HINT_T0);
#else
  (void)p;
#endif
}

/**
 * @brief Issue a non-temporal (NTA) prefetch at base+distance (best-effort).
 *
 * @details
 * On non-x86 builds this function is a no-op.
 *
 * @param base Base pointer.
 * @param distance Byte offset from base to prefetch.
 */
void ie_stream_prefetch_nta(const void *base, size_t distance) {
  const char *p = (const char *)base + distance;
#if IE_X86
  _mm_prefetch(p, _MM_HINT_NTA);
#else
  (void)p;
#endif
}

/**
 * @brief Prefetch a range with temporal (T0) hint.
 *
 * @details
 * Walks the byte range [base, base+bytes) and issues prefetches at
 * (current + distance), stepping by stride (or cacheline if stride==0).
 *
 * @param base Base pointer of the range.
 * @param bytes Number of bytes to cover.
 * @param stride Step size in bytes (0 => cacheline).
 * @param distance Prefetch lead distance in bytes.
 */
void ie_stream_prefetch_range_t0(const void *base, size_t bytes,
                                 size_t stride, size_t distance) {
  const char *c = (const char *)base;
  size_t step = (stride > 0 ? stride : IE_CACHELINE_BYTES);
  for (size_t off = 0; off < bytes; off += step) {
    ie_stream_prefetch_t0(c + off, distance);
  }
}

/**
 * @brief Prefetch a range with non-temporal (NTA) hint.
 *
 * @details
 * Walks the byte range [base, base+bytes) and issues prefetches at
 * (current + distance), stepping by stride (or cacheline if stride==0).
 *
 * @param base Base pointer of the range.
 * @param bytes Number of bytes to cover.
 * @param stride Step size in bytes (0 => cacheline).
 * @param distance Prefetch lead distance in bytes.
 */
void ie_stream_prefetch_range_nta(const void *base, size_t bytes,
                                  size_t stride, size_t distance) {
  const char *c = (const char *)base;
  size_t step = (stride > 0 ? stride : IE_CACHELINE_BYTES);
  for (size_t off = 0; off < bytes; off += step) {
    ie_stream_prefetch_nta(c + off, distance);
  }
}

/* ---------------------------- streaming kernels --------------------------- */

/**
 * @brief Scalar copy fallback for float arrays.
 *
 * @param dst Destination float array.
 * @param src Source float array.
 * @param n Number of float elements.
 */
static inline void ie_scalar_copy_f32(float *dst, const float *src, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

/**
 * @brief Scalar set fallback for float arrays.
 *
 * @param dst Destination float array.
 * @param v Value to write.
 * @param n Number of float elements.
 */
static inline void ie_scalar_set_f32(float *dst, float v, size_t n) {
  for (size_t i = 0; i < n; ++i) dst[i] = v;
}

#if IE_X86

/**
 * @brief Streaming store copy using AVX-512F (function-targeted).
 *
 * @param dst Destination float array.
 * @param src Source float array.
 * @param n Number of float elements.
 */
static IE_TARGET_AVX512F void ie_nt_copy_f32_avx512(float *dst, const float *src,
                                                    size_t n) {
  size_t i = 0;

  /* Prologue until destination aligns to 64B (best for 512-bit streams). */
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = (size_t)(addr & (64u - 1u));
  if (mis) {
    size_t pro = (64u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_copy_f32(dst, src, pro);
    dst += pro;
    src += pro;
    n -= pro;
  }

  for (; i + 16 <= n; i += 16) {
    __m512 v = _mm512_loadu_ps(src + i);
    _mm512_stream_ps(dst + i, v);
  }
  ie_scalar_copy_f32(dst + i, src + i, n - i);
  _mm_sfence();
}

/**
 * @brief Streaming store copy using AVX2 (function-targeted).
 *
 * @param dst Destination float array.
 * @param src Source float array.
 * @param n Number of float elements.
 */
static IE_TARGET_AVX2 void ie_nt_copy_f32_avx2(float *dst, const float *src,
                                               size_t n) {
  size_t i = 0;

  /* Align to 32B for 256-bit streams. */
  uintptr_t addr = (uintptr_t)dst;
  size_t mis = (size_t)(addr & (32u - 1u));
  if (mis) {
    size_t pro = (32u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_copy_f32(dst, src, pro);
    dst += pro;
    src += pro;
    n -= pro;
  }

  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    _mm256_stream_ps(dst + i, v);
  }
  ie_scalar_copy_f32(dst + i, src + i, n - i);
  _mm_sfence();
}

/**
 * @brief Streaming store set using AVX-512F (function-targeted).
 *
 * @param dst Destination float array.
 * @param value Value to write.
 * @param n Number of float elements.
 */
static IE_TARGET_AVX512F void ie_nt_set_f32_avx512(float *dst, float value,
                                                   size_t n) {
  size_t i = 0;

  uintptr_t addr = (uintptr_t)dst;
  size_t mis = (size_t)(addr & (64u - 1u));
  if (mis) {
    size_t pro = (64u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_set_f32(dst, value, pro);
    dst += pro;
    n -= pro;
  }

  __m512 v = _mm512_set1_ps(value);
  for (; i + 16 <= n; i += 16) _mm512_stream_ps(dst + i, v);
  ie_scalar_set_f32(dst + i, value, n - i);
  _mm_sfence();
}

/**
 * @brief Streaming store set using AVX2 (function-targeted).
 *
 * @param dst Destination float array.
 * @param value Value to write.
 * @param n Number of float elements.
 */
static IE_TARGET_AVX2 void ie_nt_set_f32_avx2(float *dst, float value, size_t n) {
  size_t i = 0;

  uintptr_t addr = (uintptr_t)dst;
  size_t mis = (size_t)(addr & (32u - 1u));
  if (mis) {
    size_t pro = (32u - mis) / sizeof(float);
    if (pro > n) pro = n;
    ie_scalar_set_f32(dst, value, pro);
    dst += pro;
    n -= pro;
  }

  __m256 v = _mm256_set1_ps(value);
  for (; i + 8 <= n; i += 8) _mm256_stream_ps(dst + i, v);
  ie_scalar_set_f32(dst + i, value, n - i);
  _mm_sfence();
}

#endif /* IE_X86 */

/* ----------------------------- public kernels ----------------------------- */

/**
 * @brief Copy a float array, optionally using non-temporal stores.
 *
 * @details
 * The function uses policy thresholds plus @p nt_ratio to decide whether to use
 * non-temporal streaming stores. If @p nt_ratio is 0, the function forces a
 * scalar copy even if the policy would otherwise stream.
 *
 * @param dst Destination float array.
 * @param src Source float array.
 * @param n Number of float elements.
 * @param p Stream policy (must be non-NULL).
 * @param nt_ratio Ratio in [0,1]; 0 disables NT, >0 enables if policy allows.
 */
void ie_stream_copy_f32(float *dst, const float *src, size_t n,
                        const struct ie_stream_policy *p, double nt_ratio) {
  if (!dst || !src || !p) return;

  const size_t bytes = n * sizeof(float);
  const bool want_nt =
      ie_stream_should_use_nt(bytes, p) && (ie_clamp01(nt_ratio) > 0.0);

  if (!want_nt) {
    ie_scalar_copy_f32(dst, src, n);
    return;
  }

#if IE_X86
  const struct ie_cpu_features *f = ie_stream_cpu_features();

  /* Field names are expected to match ie_cpu_features_detect() output. */
  if (f && f->avx512f) {
    ie_nt_copy_f32_avx512(dst, src, n);
    return;
  }
  if (f && f->avx2) {
    ie_nt_copy_f32_avx2(dst, src, n);
    return;
  }
#endif

  ie_scalar_copy_f32(dst, src, n);
}

/**
 * @brief Set a float array, optionally using non-temporal stores.
 *
 * @details
 * The function uses policy thresholds plus @p nt_ratio to decide whether to use
 * non-temporal streaming stores. If @p nt_ratio is 0, the function forces a
 * scalar set even if the policy would otherwise stream.
 *
 * @param dst Destination float array.
 * @param value Value to write.
 * @param n Number of float elements.
 * @param p Stream policy (must be non-NULL).
 * @param nt_ratio Ratio in [0,1]; 0 disables NT, >0 enables if policy allows.
 */
void ie_stream_memset_f32(float *dst, float value, size_t n,
                          const struct ie_stream_policy *p, double nt_ratio) {
  if (!dst || !p) return;

  const size_t bytes = n * sizeof(float);
  const bool want_nt =
      ie_stream_should_use_nt(bytes, p) && (ie_clamp01(nt_ratio) > 0.0);

  if (!want_nt) {
    ie_scalar_set_f32(dst, value, n);
    return;
  }

#if IE_X86
  const struct ie_cpu_features *f = ie_stream_cpu_features();

  /* Field names are expected to match ie_cpu_features_detect() output. */
  if (f && f->avx512f) {
    ie_nt_set_f32_avx512(dst, value, n);
    return;
  }
  if (f && f->avx2) {
    ie_nt_set_f32_avx2(dst, value, n);
    return;
  }
#endif

  ie_scalar_set_f32(dst, value, n);
}
