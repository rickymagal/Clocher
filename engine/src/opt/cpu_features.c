/**
 * @file cpu_features.c
 * @brief Runtime CPU feature detection via CPUID/XGETBV (x86) with safe fallbacks.
 *
 * @details
 * This module performs best-effort feature detection with no external deps:
 *  - AVX2 and FMA (CPUID leaf 1 + leaf 7 and XGETBV validation).
 *  - Optional AVX-512 BF16 capability (CPUID leaf 7 subleaf 1) if present in the ABI.
 *
 * It is intended to drive runtime kernel selection and logging.
 *
 * Environment:
 *  - Set `IE_DEBUG_KERNELS=1` to print a one-line CPU feature summary.
 */

#include "ie_cpu.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif

#include <stdatomic.h>
#include <strings.h> /* strcasecmp */

/* -------------------------------------------------------------------------- */
/* Local helpers                                                               */
/* -------------------------------------------------------------------------- */

/**
 * @brief Return 1 if the environment variable @p name is set to a truthy value.
 *
 * Truthy values (case-insensitive): "1", "true", "yes", "on".
 *
 * @param name Environment variable name.
 * @return 1 if truthy, 0 otherwise.
 */
static int ie_env_flag(const char *name) {
  const char *s = getenv(name);
  if (!s || !*s) return 0;
  return (!strcasecmp(s, "1") || !strcasecmp(s, "true") ||
          !strcasecmp(s, "yes") || !strcasecmp(s, "on"));
}

/**
 * @brief Query CPUID with @p leaf and @p subleaf.
 *
 * On non-x86 builds, outputs are zeroed.
 *
 * @param leaf    CPUID leaf.
 * @param subleaf CPUID subleaf (ECX input).
 * @param eax     Output EAX.
 * @param ebx     Output EBX.
 * @param ecx     Output ECX.
 * @param edx     Output EDX.
 */
static void ie_cpuid_ex(unsigned leaf, unsigned subleaf,
                        unsigned *eax, unsigned *ebx,
                        unsigned *ecx, unsigned *edx) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  #if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, (int)leaf, (int)subleaf);
    *eax = (unsigned)regs[0];
    *ebx = (unsigned)regs[1];
    *ecx = (unsigned)regs[2];
    *edx = (unsigned)regs[3];
  #else
    unsigned a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    *eax = a; *ebx = b; *ecx = c; *edx = d;
  #endif
#else
  (void)leaf; (void)subleaf;
  *eax = *ebx = *ecx = *edx = 0u;
#endif
}

/**
 * @brief Read XCR0 via XGETBV to validate OS support for XMM/YMM state.
 *
 * On non-x86 builds, returns 0.
 *
 * @return Lower 32 bits of XCR0.
 */
static unsigned ie_xgetbv_xcr0(void) {
#if (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
  unsigned a, d;
  __asm__ volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(a), "=d"(d) : "c"(0));
  return a;
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  return (unsigned)_xgetbv(0);
#else
  return 0u;
#endif
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                  */
/* -------------------------------------------------------------------------- */

bool ie_cpu_features_detect(ie_cpu_features_t *out) {
  if (!out) return false;

  out->avx2 = false;
  out->fma  = false;

  /* If your ie_cpu_features_t includes this field, we fill it. */
#if defined(__GNUC__) || defined(_MSC_VER)
  /* Some builds may not include avx512_bf16 in the struct; keep this guarded
   * at the source level by only referencing the member name used in your ABI.
   */
  out->avx512_bf16 = false;
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  unsigned a=0,b=0,c=0,d=0;

  /* Leaf 0: max leaf availability */
  ie_cpuid_ex(0, 0, &a, &b, &c, &d);
  if (a == 0) {
    return true; /* CPUID unsupported or unavailable */
  }

  /* Leaf 1: baseline features (AVX, OSXSAVE, FMA) */
  ie_cpuid_ex(1, 0, &a, &b, &c, &d);
  const int osxsave = (int)((c >> 27) & 1u);
  const int avx     = (int)((c >> 28) & 1u);
  const int fma     = (int)((c >> 12) & 1u);

  /* Validate OS has enabled XMM/YMM via XGETBV */
  unsigned xcr0 = osxsave ? ie_xgetbv_xcr0() : 0u;
  const int ymm_ok = ((xcr0 & 0x6u) == 0x6u);

  /* Leaf 7 subleaf 0: extended features (AVX2) */
  ie_cpuid_ex(7, 0, &a, &b, &c, &d);
  const int avx2 = (int)((b >> 5) & 1u);

  out->avx2 = (avx && ymm_ok && avx2);
  out->fma  = (fma != 0);

  /* Leaf 7 subleaf 1: AVX512_BF16 is in EAX bit 5 on supporting CPUs.
   * We still store it even if the OS might not enable ZMM state; this flag is
   * primarily used as a capability indicator for "do we even try BF16 kernels".
   */
  ie_cpuid_ex(7, 1, &a, &b, &c, &d);
  {
    const int avx512bf16 = (int)((a >> 5) & 1u);
    out->avx512_bf16 = (avx512bf16 != 0);
  }
#endif

  if (ie_env_flag("IE_DEBUG_KERNELS")) {
    char buf[256];
    (void)ie_cpu_features_to_string(out, buf, sizeof buf);
    fprintf(stderr, "[cpu] %s\n", buf);
  }

  return true;
}

/**
 * @brief Serialize CPU features into a stable one-line string.
 *
 * @param f    Feature struct.
 * @param buf  Output buffer.
 * @param cap  Output capacity in bytes.
 * @return Number of bytes written (excluding NUL), clamped to `cap-1`.
 */
size_t ie_cpu_features_to_string(const ie_cpu_features_t *f, char *buf, size_t cap) {
  if (!buf || cap == 0) return 0;
  if (!f) {
    buf[0] = 0;
    return 0;
  }

  const int n = snprintf(buf, cap,
                         "avx2=%d fma=%d avx512_bf16=%d",
                         f->avx2 ? 1 : 0,
                         f->fma ? 1 : 0,
                         f->avx512_bf16 ? 1 : 0);
  if (n < 0) {
    buf[0] = 0;
    return 0;
  }

  /* Clamp without signedness warnings under -Werror. */
  return ((size_t)n >= cap) ? (cap - 1) : (size_t)n;
}
