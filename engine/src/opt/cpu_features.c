/**
 * @file cpu_features.c
 * @brief Best-effort CPU feature detection for AVX2 and FMA at runtime.
 *
 * This module avoids external dependencies and uses CPUID/XGETBV directly.
 * On non-x86 platforms all feature flags are reported as false.
 */

#include "ie_cpu.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif

/**
 * @brief Query CPUID with the given leaf.
 *
 * On non-x86 builds, the outputs are set to zero.
 *
 * @param leaf  CPUID leaf to query.
 * @param eax   Output EAX register value.
 * @param ebx   Output EBX register value.
 * @param ecx   Output ECX register value.
 * @param edx   Output EDX register value.
 */
static void cpuid_ex(unsigned leaf,
                     unsigned *eax, unsigned *ebx,
                     unsigned *ecx, unsigned *edx) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  #if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, (int)leaf);
    *eax = (unsigned)regs[0];
    *ebx = (unsigned)regs[1];
    *ecx = (unsigned)regs[2];
    *edx = (unsigned)regs[3];
  #else
    unsigned a, b, c, d;
    __cpuid(leaf, a, b, c, d);
    *eax = a; *ebx = b; *ecx = c; *edx = d;
  #endif
#else
  (void)leaf; *eax = *ebx = *ecx = *edx = 0;
#endif
}

/**
 * @brief Read XCR0 via XGETBV to validate OS support for YMM state.
 *
 * On non-x86 builds, returns 0.
 *
 * @return Lower 32 bits of XCR0.
 */
static unsigned xgetbv_xcr0(void) {
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

bool ie_cpu_detect(ie_cpu_features_t *out) {
  if (!out) return false;
  out->avx2 = false;
  out->fma  = false;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  unsigned a=0,b=0,c=0,d=0;

  /* Leaf 0: availability */
  cpuid_ex(0, &a, &b, &c, &d);
  if (a == 0) return true; /* CPUID unsupported or unavailable */

  /* Leaf 1: baseline features (AVX, OSXSAVE, FMA) */
  cpuid_ex(1, &a, &b, &c, &d);
  const int osxsave = (int)((c >> 27) & 1);
  const int avx     = (int)((c >> 28) & 1);
  const int fma     = (int)((c >> 12) & 1);

  /* Check OS has enabled XMM/YMM via XGETBV */
  unsigned xcr0 = osxsave ? xgetbv_xcr0() : 0u;
  const int ymm_ok = ((xcr0 & 0x6u) == 0x6u);

  /* Leaf 7: extended features (AVX2) */
  cpuid_ex(7, &a, &b, &c, &d);
  const int avx2 = (int)((b >> 5) & 1);

  out->avx2 = (avx && ymm_ok && avx2);
  out->fma  = (fma != 0);
#else
  /* Non-x86: keep safe defaults (false). */
#endif
  return true;
}
