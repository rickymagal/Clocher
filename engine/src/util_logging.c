/**
 * @file util_logging.c
 * @brief Minimal logging helpers (baseline).
 */
#include <stdio.h>
#include <stdarg.h>

static int g_log_level = 1; /* 0=quiet, 1=info, 2=debug */

void ie_log_set_level(int lvl) { g_log_level = lvl; }

void ie_log_info(const char *fmt, ...) {
  if (g_log_level < 1) return;
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[info] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

void ie_log_debug(const char *fmt, ...) {
  if (g_log_level < 2) return;
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[debug] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}
