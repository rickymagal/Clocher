/**
 * @file util_logging.h
 * @brief Minimal logging interface (printf-style), used by CLI and tests.
 */

#ifndef UTIL_LOGGING_H
#define UTIL_LOGGING_H

#include <stdarg.h>

/**
 * @brief Log an informational message (printf-style).
 *
 * @param fmt Format string.
 * @param ... Arguments.
 */
void ie_log_info(const char *fmt, ...);

/**
 * @brief Log a warning message (printf-style).
 *
 * @param fmt Format string.
 * @param ... Arguments.
 */
void ie_log_warn(const char *fmt, ...);

/**
 * @brief Log an error message (printf-style).
 *
 * @param fmt Format string.
 * @param ... Arguments.
 */
void ie_log_error(const char *fmt, ...);

#endif /* UTIL_LOGGING_H */
