/**
 * @file ie_device_ze.cpp
 * @brief oneAPI Level Zero backend (skeleton implementation).
 *
 * This TU compiles when IE_WITH_ZE=1 and provides runtime probes that
 * currently return "unavailable". The goal is to let the engine wire the
 * selection logic now and fill in the kernel path later without breaking
 * the build or the CPU/CUDA flows.
 *
 * To implement for real:
 *  - Initialize zeInit with ZE_INIT_FLAG_GPU_ONLY (or 0).
 *  - Enumerate drivers/devices, pick a GPU.
 *  - Create context, command queue/list, module (SPIR-V), kernel.
 *  - Upload buffers, set arguments, dispatch, copy back, teardown.
 */

#include "ie_device_ze.h"

#if defined(IE_WITH_ZE) && (IE_WITH_ZE+0)==1

/* In a full implementation we would include:
 * #include <level_zero/ze_api.h>
 * and link with -lze_loader.
 * For now we keep the skeleton minimal.
 */

int ie_ze_is_available(void) {
  /* TODO: call zeInit and enumerate devices. */
  return IE_ZE_UNAVAILABLE;
}

int ie_ze_gemv_f32(const float *W,
                   const float *x,
                   float *y,
                   int rows,
                   int cols,
                   int ldw)
{
  (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)ldw;
  /* TODO: actual L0 kernel path. */
  return IE_ZE_UNAVAILABLE;
}

#else /* IE_WITH_ZE != 1 */

int ie_ze_is_available(void) {
  return IE_ZE_UNAVAILABLE;
}

int ie_ze_gemv_f32(const float *W,
                   const float *x,
                   float *y,
                   int rows,
                   int cols,
                   int ldw)
{
  (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)ldw;
  return IE_ZE_UNAVAILABLE;
}

#endif /* IE_WITH_ZE */
