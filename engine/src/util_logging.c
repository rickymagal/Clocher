/**
 * @file util_logging.c
 * @brief Minimal logging helpers (baseline).
 *
 * Provides leveled logging to stderr. Levels:
 *   0 = quiet, 1 = info, 2 = debug.
 *
 * This module is intentionally tiny and dependency-free.
 */

#include <stdio.h>
#include <stdarg.h>

/** @brief Global log level (0=quiet, 1=info, 2=debug). */
static int g_log_level = 1;

/**
 * @brief Set the global log verbosity level.
 * @param lvl 0=quiet, 1=info, 2=debug.
 */
void ie_log_set_level(int lvl) { g_log_level = lvl; }

/**
 * @brief Emit an informational log line to stderr if level >= 1.
 * @param fmt printf-style format string.
 * @param ... Variadic arguments.
 */
void ie_log_info(const char *fmt, ...) {
  if (g_log_level < 1) return;
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[info] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

/**
 * @brief Emit a debug log line to stderr if level >= 2.
 * @param fmt printf-style format string.
 * @param ... Variadic arguments.
 */
void ie_log_debug(const char *fmt, ...) {
  if (g_log_level < 2) return;
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[debug] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}
