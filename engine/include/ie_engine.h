/* ============================================================================
 * File: engine/include/ie_engine.h
 * ============================================================================
 */
/**
 * @file ie_engine.h
 * @brief Public engine façade (compatibility shim).
 *
 * This header intentionally re-exports the stable API declared in
 * `ie_api.h` without introducing new typedefs or structs. Doing so avoids
 * duplicate definitions of:
 *   - ie_status_t
 *   - ie_metrics / ie_metrics_t
 *   - ie_engine_params / ie_engine_params_t
 *   - ie_engine_t
 * and keeps all function prototypes single-sourced.
 *
 * @note If you previously had a standalone `ie_engine.h` defining its own
 *       enums/structs, remove it and use this file to prevent redefinition
 *       errors during compilation and linking.
 *
 * @defgroup ie_engine Public Engine API (compat)
 * @details See the symbols documented in `ie_api.h`. The key calls are:
 * - ::ie_engine_create
 * - ::ie_engine_generate
 * - ::ie_engine_metrics
 * - ::ie_engine_destroy
 * @{
 */

#ifndef IE_ENGINE_H_
#define IE_ENGINE_H_

#include <stddef.h>
#include <stdint.h>

/* The single source of truth for public types and functions. */
#include "ie_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* No additional declarations here on purpose. All public types and functions
 * are provided by ie_api.h. Keeping this file as a thin façade guarantees that
 * projects including either "ie_engine.h" or "ie_api.h" see the exact same API.
 */

/**
 * @brief Placeholder doxygen anchor for the engine handle.
 *
 * Use ::ie_engine_t as declared in `ie_api.h`.
 */

/**
 * @brief Placeholder doxygen anchor for engine parameters.
 *
 * Use ::ie_engine_params_t as declared in `ie_api.h`.
 */

/**
 * @brief Placeholder doxygen anchor for engine metrics.
 *
 * Use ::ie_metrics_t as declared in `ie_api.h`.
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_ENGINE_H_ */

/** @} end of ie_engine group */
