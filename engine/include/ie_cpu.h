/**
 * @file ie_cpu.h
 * @brief CPU feature detection API for runtime kernel dispatch.
 *
 * This header exposes a tiny, dependency-free interface to detect
 * a subset of x86 CPU features (AVX2 and FMA). On non-x86 platforms,
 * detection succeeds but all features are reported as `false`.
 */

#ifndef IE_CPU_H
#define IE_CPU_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU feature set used for kernel selection.
 */
typedef struct {
  bool avx2;  /**< @brief AVX2 support present at runtime. */
  bool fma;   /**< @brief FMA (Fused Multiply-Add) support present. */
} ie_cpu_features_t;

/**
 * @brief Detect CPU features on the current machine.
 *
 * The function is best-effort: it never crashes on unsupported platforms.
 * On non-x86 builds it returns `true` and fills all flags as `false`.
 *
 * @param[out] out  Pointer to a #ie_cpu_features_t structure to fill.
 * @return `true` if the function executed; `false` only if @p out is NULL.
 */
bool ie_cpu_detect(ie_cpu_features_t *out);

#ifdef __cplusplus
}
#endif

#endif /* IE_CPU_H */
