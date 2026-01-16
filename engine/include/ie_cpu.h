/**
 * @file ie_cpu.h
 * @brief Runtime CPU feature detection utilities.
 *
 * @details
 * This header defines a small, dependency-free API to detect ISA capabilities
 * relevant to CPU inference kernels at runtime (e.g., selecting AVX2 GEMV).
 *
 * The implementation is best-effort:
 *  - On x86/x86_64: uses CPUID and XGETBV to confirm both HW support and OS-
 *    enabled register state (XMM/YMM/ZMM).
 *  - On non-x86: reports all capabilities as false.
 *
 * The detected flags are intended for:
 *  - Choosing kernel variants (generic vs AVX2 vs AVX-512).
 *  - Logging/instrumentation to confirm which fast paths are available.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Set of x86 SIMD capabilities used by the inference engine.
 *
 * @note
 * `avx512_bf16` implies `avx512f` + OS ZMM state enabled, but the struct keeps
 * each flag explicit so callers can emit clear logs.
 */
typedef struct ie_cpu_features {
  /** @brief OS+HW support for AVX (XMM/YMM enabled). */
  bool avx;
  /** @brief OS+HW support for AVX2 (requires AVX + CPUID leaf 7 AVX2). */
  bool avx2;
  /** @brief HW support for FMA (CPUID leaf 1 FMA). */
  bool fma;

  /** @brief OS+HW support for AVX-512 Foundation. */
  bool avx512f;
  /** @brief OS+HW support for AVX-512 DQ. */
  bool avx512dq;
  /** @brief OS+HW support for AVX-512 BW. */
  bool avx512bw;
  /** @brief OS+HW support for AVX-512 VL. */
  bool avx512vl;

  /**
   * @brief OS+HW support for AVX-512 BF16.
   *
   * @details
   * Reported via CPUID leaf 7 subleaf 1 (EAX bit 5) with ZMM state enabled.
   */
  bool avx512_bf16;
} ie_cpu_features_t;

/**
 * @brief Detect CPU features at runtime.
 *
 * @param[out] out Receives the detected capabilities.
 * @return true if detection ran (even if all flags are false), false on invalid
 *         arguments.
 */
bool ie_cpu_detect(ie_cpu_features_t *out);

/**
 * @brief Alias for @ref ie_cpu_detect for clearer call sites.
 *
 * Some call sites prefer "features_detect" naming (e.g., instrumentation).
 *
 * @param[out] out Receives the detected capabilities.
 * @return See @ref ie_cpu_detect.
 */
bool ie_cpu_features_detect(ie_cpu_features_t *out);

/**
 * @brief Format a feature set as a compact, stable string.
 *
 * Example:
 * `avx=1 avx2=1 fma=1 avx512f=0 avx512dq=0 avx512bw=0 avx512vl=0 avx512_bf16=0`
 *
 * @param feats Feature set (may be NULL, treated as all zeros).
 * @param buf Output buffer.
 * @param cap Capacity of @p buf in bytes.
 * @return Number of bytes written (excluding NUL), or 0 on invalid buffer.
 */
size_t ie_cpu_features_to_string(const ie_cpu_features_t *feats, char *buf, size_t cap);

#ifdef __cplusplus
} /* extern "C" */
#endif
