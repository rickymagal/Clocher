/**
 * @file util_metrics.c
 * @brief Metrics helpers (baseline).
 *
 * This module exists to host utility routines related to metrics snapshots.
 * Currently only zeroing is needed; more helpers may be added later.
 */

#include <string.h>
#include "ie_metrics.h"

/**
 * @brief Zero-initialize a metrics struct.
 * @param m Metrics struct to clear.
 */
void ie_metrics_zero(ie_metrics_t *m) {
  memset(m, 0, sizeof(*m));
}
