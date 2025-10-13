/**
 * @file util_metrics.c
 * @brief Metrics helpers (baseline).
 */
#include <string.h>
#include "ie_metrics.h"

void ie_metrics_zero(ie_metrics_t *m) {
  memset(m, 0, sizeof(*m));
}
