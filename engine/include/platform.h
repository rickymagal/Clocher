/**
 * @file platform.h
 * @brief Platform feature detection (baseline stubs).
 */
#ifndef IE_PLATFORM_H
#define IE_PLATFORM_H
#ifdef __x86_64__
  #define IE_X86 1
#else
  #define IE_X86 0
#endif
#endif /* IE_PLATFORM_H */
