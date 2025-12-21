/* File: engine/src/opt/numa_probe.c
 * -----------------------------------------------------------------------------
 * @file numa_probe.c
 * @brief Detect NUMA nodes via sysfs without linking libnuma.
 *
 * @details
 * On Linux, this implementation reads `/sys/devices/system/node/online` and
 * parses range-lists such as:
 *   - "0"
 *   - "0-1"
 *   - "0-3,8-9"
 * to infer a conservative node count as `(max_id + 1)`.
 *
 * On non-Linux platforms (or on failure), it returns 1.
 */

#define _POSIX_C_SOURCE 200809L

#include "ie_numa.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Parse a NUMA node range-list and return `(max_id + 1)`.
 *
 * @details
 * The input is expected to be a NUL-terminated ASCII string containing one or
 * more comma-separated ranges. Each range is either a single integer `N` or a
 * closed interval `A-B` where `A` and `B` are non-negative integers and
 * `A <= B`. Examples:
 *   - "0"            -> returns 1
 *   - "0-1"          -> returns 2
 *   - "0-3,8-9"      -> returns 10 (since max id is 9)
 *
 * Parsing notes:
 *  - If the string cannot be fully parsed, the function returns `(max_id + 1)`
 *    for any successfully parsed ranges encountered before the error.
 *  - If no valid range is found at all, the function returns 1.
 *
 * @param s NUL-terminated range-list string to parse (non-NULL).
 * @return `(max_id + 1)` where `max_id` is the largest parsed id; at least 1.
 */
static int parse_range_list(const char *s) {
  int max_id = 0;
  const char *p = s;

  while (*p) {
    int a = 0;
    int b = 0;
    char *end = NULL;

    a = (int)strtol(p, &end, 10);
    if (end == p) break;
    p = end;

    if (*p == '-') {
      ++p;
      b = (int)strtol(p, &end, 10);
      if (end == p) break;
      p = end;
    } else {
      b = a;
    }

    if (b > max_id) max_id = b;

    if (*p == ',') ++p;
  }

  return max_id + 1;
}

/**
 * @brief Detect the number of NUMA nodes.
 *
 * @details
 * On Linux, reads `/sys/devices/system/node/online` and parses it. If the file
 * cannot be opened, read, or yields no usable data, returns 1.
 *
 * On non-Linux systems, always returns 1.
 *
 * @return Detected NUMA node count (>= 1).
 */
int ie_numa_detect_nodes(void) {
#ifdef __linux__
  FILE *f = fopen("/sys/devices/system/node/online", "r");
  if (!f) return 1;

  char buf[256] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);

  if (n == 0) return 1;
  buf[n] = '\0';

  int nodes = parse_range_list(buf);
  return (nodes > 0) ? nodes : 1;
#else
  return 1;
#endif
}
