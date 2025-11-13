/**
 * @file ie_stream.h
 * @brief Portable wrappers for controlled prefetching and non-temporal (streaming) stores.
 *
 * This header provides:
 *  - Thin wrappers around _mm_prefetch with cache-level hints.
 *  - Heuristics for choosing prefetch distances and when to use non-temporal stores.
 *  - High-level streaming copy/set helpers that mix regular and streaming stores via a ratio.
 *
 * Design notes (safety & portability):
 *  - All intrinsics are guarded by architecture macros. On non-x86 or when a feature is
 *    unavailable at compile time, safe scalar fallbacks are used.
 *  - Streaming stores require attention to alignment for best results. The helpers handle
 *    a small scalar "prologue/epilogue" to avoid alignment faults and still benefit from
 *    wide stores on the bulk region when available.
 *  - Prefetching uses byte distances relative to the base pointer. Distances should be
 *    chosen as multiples of the cache line size (commonly 64 bytes on x86).
 *
 * Environment variables (optional):
 *  - IE_STREAM_PREFETCH=<bytes>          : override recommended prefetch distance.
 *  - IE_STREAM_NT_THRESHOLD=<bytes>      : size threshold above which non-temporal stores are preferred.
 *  - IE_STREAM_FORCE_NT=0/1              : force-disable or force-enable streaming stores.
 */

#ifndef IE_STREAM_H
#define IE_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Conventional cache line size used for distance heuristics (bytes). */
#ifndef IE_CACHELINE_BYTES
#define IE_CACHELINE_BYTES 64u
#endif

/** @brief Policy controlling prefetch and streaming behavior. */
struct ie_stream_policy {
  /** Detected or user-provided L1 data cache size in bytes (0 if unknown). */
  size_t l1_bytes;
  /** Detected or user-provided L2 cache size in bytes (0 if unknown). */
  size_t l2_bytes;
  /** Detected or user-provided LLC/L3 cache size in bytes (0 if unknown). */
  size_t l3_bytes;

  /** Prefetch lead distance in bytes (how far ahead to prefetch). */
  size_t prefetch_distance;
  /** If payload bytes >= this threshold, prefer non-temporal stores (unless forced off). */
  size_t nt_threshold_bytes;

  /** Prefetch hint to pass to _mm_prefetch (0=T0, 1=T1, 2=T2, 3=NTA). */
  int prefetch_hint;

  /** Enable software prefetching when true. */
  bool enable_prefetch;
  /** Force-enable streaming stores (overrides threshold). */
  bool force_nt_on;
  /** Force-disable streaming stores. */
  bool force_nt_off;
};

/**
 * @brief Initialize policy with detected cache sizes and sane defaults.
 *
 * The function attempts to read cache sizes from Linux sysfs and then applies
 * internal heuristics to set @ref prefetch_distance and @ref nt_threshold_bytes.
 * Environment variables listed in the file header override the computed values.
 *
 * @param[out] p  Non-null policy to be initialized.
 * @return 0 on success; negative value on error (e.g., invalid pointer).
 */
int ie_stream_policy_init(struct ie_stream_policy *p);

/**
 * @brief Decide if non-temporal stores should be used for a given transfer size.
 *
 * The decision observes @ref force_nt_on and @ref force_nt_off first; otherwise
 * it compares @p bytes with @ref nt_threshold_bytes.
 *
 * @param bytes  Total transfer size in bytes.
 * @param p      Non-null policy pointer.
 * @return true if non-temporal stores are recommended.
 */
bool ie_stream_should_use_nt(size_t bytes, const struct ie_stream_policy *p);

/**
 * @brief Recommend a prefetch distance in bytes for the given stride.
 *
 * The recommendation considers the policy prefetch distance, stride size, and
 * attempts to keep the lead within a few cache lines.
 *
 * @param p             Non-null policy pointer.
 * @param stride_bytes  Positive stride size in bytes.
 * @return Distance in bytes to pass to the prefetch helpers.
 */
size_t ie_stream_recommend_prefetch_distance(const struct ie_stream_policy *p,
                                             size_t stride_bytes);

/**
 * @brief Issue a T0 (cache-hot) prefetch @p distance bytes ahead of @p base.
 *
 * @param base      Base address (non-null).
 * @param distance  Lead distance in bytes (can be zero).
 */
void ie_stream_prefetch_t0(const void *base, size_t distance);

/**
 * @brief Issue a non-temporal (NTA) prefetch @p distance bytes ahead of @p base.
 *
 * @param base      Base address (non-null).
 * @param distance  Lead distance in bytes (can be zero).
 */
void ie_stream_prefetch_nta(const void *base, size_t distance);

/**
 * @brief Prefetch a range using T0 hint at a given stride and lead distance.
 *
 * @param base      Range base (non-null).
 * @param bytes     Range size in bytes.
 * @param stride    Step in bytes between prefetches (>= IE_CACHELINE_BYTES recommended).
 * @param distance  Lead distance in bytes (see @ref ie_stream_recommend_prefetch_distance).
 */
void ie_stream_prefetch_range_t0(const void *base, size_t bytes,
                                 size_t stride, size_t distance);

/**
 * @brief Prefetch a range using NTA hint at a given stride and lead distance.
 *
 * @param base      Range base (non-null).
 * @param bytes     Range size in bytes.
 * @param stride    Step in bytes between prefetches.
 * @param distance  Lead distance in bytes.
 */
void ie_stream_prefetch_range_nta(const void *base, size_t bytes,
                                  size_t stride, size_t distance);

/**
 * @brief Copy @p n floats from @p src to @p dst using optional non-temporal stores.
 *
 * If @ref ie_stream_should_use_nt decides for streaming, the function uses wide
 * non-temporal stores when available (AVX2/AVX-512). The @p nt_ratio allows mixing
 * regular and streaming stores (e.g., 0.5 means half of the bulk uses streaming).
 * For small tails or when features are unavailable, a scalar fallback is used.
 *
 * @param dst       Destination pointer (non-null).
 * @param src       Source pointer (non-null).
 * @param n         Number of float elements to copy.
 * @param p         Non-null policy pointer.
 * @param nt_ratio  Fraction in [0,1]; 0=never stream, 1=always stream (when allowed).
 */
void ie_stream_copy_f32(float *dst, const float *src, size_t n,
                        const struct ie_stream_policy *p, double nt_ratio);

/**
 * @brief Set @p n floats at @p dst to @p value using optional non-temporal stores.
 *
 * @param dst       Destination pointer (non-null).
 * @param value     Value to set.
 * @param n         Number of float elements to set.
 * @param p         Non-null policy pointer.
 * @param nt_ratio  Fraction in [0,1] controlling the mix of streaming stores.
 */
void ie_stream_memset_f32(float *dst, float value, size_t n,
                          const struct ie_stream_policy *p, double nt_ratio);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_STREAM_H */
