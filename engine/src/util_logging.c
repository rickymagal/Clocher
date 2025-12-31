/**
 * @file util_logging.c
 * @brief Minimal logging helpers (stderr) used across the engine.
 *
 * This project intentionally avoids third-party dependencies. This module
 * provides a small, stable logging surface that other components can link
 * against without pulling in any extra libraries.
 */

#include "util_logging.h"

#include <stdarg.h>
#include <stdio.h>

static void ie_vlog_(const char *tag, const char *fmt, va_list ap) {
  if (!fmt) return;

  if (tag && tag[0]) {
    fputs(tag, stderr);
    fputs(": ", stderr);
  }

  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
}

void ie_log_info(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  ie_vlog_("info", fmt, ap);
  va_end(ap);
}

void ie_log_warn(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  ie_vlog_("warn", fmt, ap);
  va_end(ap);
}

void ie_log_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  ie_vlog_("error", fmt, ap);
  va_end(ap);
}
