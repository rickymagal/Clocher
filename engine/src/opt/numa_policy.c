/* File: engine/src/opt/numa_policy.c
 * -----------------------------------------------------------------------------
 * @file numa_policy.c
 * @brief NUMA-aware policy initialization and helper decisions.
 *
 * @details
 * This module provides a small policy object used by NUMA-aware subsystems.
 * It:
 *  - obtains socket count via ie_topology when available,
 *  - obtains NUMA node count via sysfs-based detection (no libnuma),
 *  - reads env knobs to decide whether NUMA is enabled and whether to replicate
 *    hot blobs per socket,
 *  - provides helper decisions used by the loader and hot replication logic.
 *
 * Environment variables:
 *  - IE_NUMA_ENABLE            (default: 1)
 *  - IE_NUMA_REPLICATE_HOT     (default: 1)
 *  - IE_NUMA_MAX_REPLICAS      (default: detected_sockets, clamped to [1, 64])
 *  - IE_NUMA_HOT_MIN_BYTES     (default: 33554432 = 32 MiB)
 */

#define _POSIX_C_SOURCE 200809L

#include "numa_policy.h"

#include "ie_numa.h"
#include "ie_topology.h"
#include "util_logging.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef IE_LOG_WARN
  #include <stdio.h>
  #define IE_LOG_WARN(...) do { fprintf(stderr, "[warn] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif

/* ------------------------------ env parsing -------------------------------- */

/**
 * @brief Case-insensitive ASCII equality helper.
 *
 * @details
 * This helper compares two NUL-terminated strings using ASCII case folding.
 * It is intentionally minimal and does not attempt locale-sensitive behavior.
 *
 * @param a NUL-terminated string (may be NULL).
 * @param b NUL-terminated string (may be NULL).
 * @return 1 if equal (case-insensitive), otherwise 0.
 */
static int str_ieq(const char *a, const char *b) {
  if (!a || !b) return 0;
  while (*a && *b) {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    ++a;
    ++b;
  }
  return (*a == '\0' && *b == '\0');
}

/**
 * @brief Read a boolean-like environment variable.
 *
 * @details
 * Accepted false values (case-insensitive): "0", "false", "no", "off".
 * Any other non-empty value is treated as true.
 *
 * @param name Environment variable name.
 * @param default_value Value used when the variable is unset/empty.
 * @return 0 or 1.
 */
static int env_flag_get(const char *name, int default_value) {
  const char *v = getenv(name);
  if (!v || !*v) return default_value;

  if (str_ieq(v, "0") || str_ieq(v, "false") || str_ieq(v, "no") || str_ieq(v, "off")) return 0;
  return 1;
}

/**
 * @brief Read an integer environment variable with clamping.
 *
 * @details
 * If the variable is unset/empty or unparsable, @p def is returned.
 * Otherwise, the parsed value is clamped to [min_v, max_v].
 *
 * @param name Env var name.
 * @param def Default if unset or unparsable.
 * @param min_v Minimum allowed value.
 * @param max_v Maximum allowed value.
 * @return Clamped integer value.
 */
static int env_int_get(const char *name, int def, int min_v, int max_v) {
  const char *v = getenv(name);
  if (!v || !*v) return def;

  char *end = NULL;
  long x = strtol(v, &end, 10);
  if (end == v) return def;

  if (x < (long)min_v) x = (long)min_v;
  if (x > (long)max_v) x = (long)max_v;
  return (int)x;
}

/**
 * @brief Read a size_t environment variable with clamping.
 *
 * @details
 * If the variable is unset/empty or unparsable, @p def is returned.
 * Otherwise, the parsed value is clamped to [min_v, max_v].
 *
 * @param name Env var name.
 * @param def Default if unset or unparsable.
 * @param min_v Minimum allowed value.
 * @param max_v Maximum allowed value.
 * @return Clamped size_t value.
 */
static size_t env_sizet_get(const char *name, size_t def, size_t min_v, size_t max_v) {
  const char *v = getenv(name);
  if (!v || !*v) return def;

  char *end = NULL;
  unsigned long long x = strtoull(v, &end, 10);
  if (end == v) return def;

  if (x < (unsigned long long)min_v) x = (unsigned long long)min_v;
  if (x > (unsigned long long)max_v) x = (unsigned long long)max_v;
  return (size_t)x;
}

/* ------------------------------- public API -------------------------------- */

/**
 * @brief Initialize a NUMA policy snapshot from topology detection and env knobs.
 *
 * @details
 * This function is meant to be called during startup to capture a stable
 * policy snapshot. It does not allocate memory and has no side effects beyond
 * reading environment variables.
 *
 * Socket count is sourced from @p topo (if provided), using ie_topology_sockets().
 * Node count is sourced from ie_numa_detect_nodes() (sysfs-based on Linux).
 *
 * @param out  Output policy object (non-NULL).
 * @param topo Optional topology handle (may be NULL).
 * @return 0 on success, negative errno-like value on failure.
 */
int ie_numa_policy_init(ie_numa_policy_t *out, const ie_topology_t *topo) {
  if (!out) return -EINVAL;
  memset(out, 0, sizeof(*out));

  int sockets = 1;
  if (topo) {
    sockets = ie_topology_sockets(topo);
    if (sockets <= 0) sockets = 1;
  }

  out->detected_nodes = ie_numa_detect_nodes();
  if (out->detected_nodes <= 0) out->detected_nodes = 1;

  out->detected_sockets = sockets;

  out->enable_numa   = (env_flag_get("IE_NUMA_ENABLE", 1) != 0);
  out->replicate_hot = (env_flag_get("IE_NUMA_REPLICATE_HOT", 1) != 0);

  out->max_replicas = sockets;
  if (out->max_replicas < 1) out->max_replicas = 1;
  if (out->max_replicas > 64) out->max_replicas = 64;
  out->max_replicas = env_int_get("IE_NUMA_MAX_REPLICAS", out->max_replicas, 1, 64);

  /* Default: 32 MiB, clamped to 1 TiB. */
  out->hot_min_bytes = env_sizet_get("IE_NUMA_HOT_MIN_BYTES",
                                     (size_t)33554432u,
                                     (size_t)0u,
                                     (size_t)1099511627776ull);

  return 0;
}

/**
 * @brief Decide if a blob of @p bytes should be replicated per socket.
 *
 * @details
 * Replication is intended for large/hot blobs where remote NUMA access would
 * otherwise dominate runtime. This function only provides a policy decision;
 * actual replication and placement are handled elsewhere.
 *
 * @param p     Policy object (non-NULL).
 * @param bytes Blob size in bytes.
 * @return true if replication should happen, false otherwise.
 */
bool ie_numa_policy_should_replicate_hot(const ie_numa_policy_t *p, size_t bytes) {
  if (!p) return false;
  if (!p->enable_numa) return false;
  if (!p->replicate_hot) return false;
  if (p->detected_sockets <= 1) return false;
  if (bytes < p->hot_min_bytes) return false;
  return true;
}

/**
 * @brief Compute the replica count to build under this policy.
 *
 * @details
 * This returns min(detected_sockets, max_replicas), clamped to at least 1.
 * Callers can use this as the number of per-socket replicas to allocate.
 *
 * @param p Policy object (non-NULL).
 * @return Number of replicas to create (>= 1).
 */
int ie_numa_policy_replica_count(const ie_numa_policy_t *p) {
  if (!p) return 1;
  int s = (p->detected_sockets > 0) ? p->detected_sockets : 1;
  int m = (p->max_replicas > 0) ? p->max_replicas : 1;
  return (s < m) ? s : m;
}
