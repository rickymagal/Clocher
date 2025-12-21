/* File: engine/include/opt/numa_policy.h
 * -----------------------------------------------------------------------------
 * @file numa_policy.h
 * @brief Runtime NUMA policy knobs used by the loader and hot-weights replication logic.
 *
 * @details
 * This header defines a small policy/config snapshot that:
 *  - captures detected topology facts (sockets, NUMA nodes),
 *  - reads environment-based knobs,
 *  - provides helper decisions (e.g., whether to replicate a hot blob).
 *
 * Design goals:
 *  - Lightweight: no libnuma dependency.
 *  - Stable ABI surface: only plain C types and a forward-declared topology handle.
 *  - Deterministic behavior: env vars define policy knobs; topology supplies facts.
 *
 * Environment variables (consumed by ie_numa_policy_init):
 *  - IE_NUMA_ENABLE            (default: 1)
 *  - IE_NUMA_REPLICATE_HOT     (default: 1)
 *  - IE_NUMA_MAX_REPLICAS      (default: detected_sockets, clamped to [1, 64])
 *  - IE_NUMA_HOT_MIN_BYTES     (default: 33554432 = 32 MiB)
 */

#ifndef IE_OPT_NUMA_POLICY_H
#define IE_OPT_NUMA_POLICY_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration to avoid a hard include dependency on ie_topology.h. */
struct ie_topology;
typedef struct ie_topology ie_topology_t;

/**
 * @struct ie_numa_policy_t
 * @brief Policy/config snapshot for NUMA-aware behavior.
 *
 * @details
 * This object is a "snapshot" meant to be initialized once at startup, then
 * passed around by value or const pointer to modules that need to decide:
 *  - whether NUMA-aware logic should run,
 *  - whether large/hot blobs should be replicated per socket,
 *  - how many replicas are allowed.
 *
 * All fields are intentionally simple and stable.
 */
typedef struct ie_numa_policy_t {
  int    detected_nodes;    /**< NUMA node count (>= 1). */
  int    detected_sockets;  /**< Socket count (>= 1). */

  bool   enable_numa;       /**< Master enable switch for NUMA-aware behavior. */
  bool   replicate_hot;     /**< Allow per-socket replication of hot blobs. */

  int    max_replicas;      /**< Upper bound on replicas to create (>= 1). */
  size_t hot_min_bytes;     /**< Minimum blob size for replication to be worth it. */
} ie_numa_policy_t;

/**
 * @brief Initialize a NUMA policy snapshot from topology detection and env knobs.
 *
 * @details
 * This function populates detected topology facts and reads the environment
 * variables listed in the module documentation. It does not allocate memory.
 *
 * If @p topo is NULL (or provides invalid socket information), sockets default
 * to 1. NUMA nodes are detected via ie_numa_detect_nodes() (sysfs-based on
 * Linux), defaulting to 1 on failure.
 *
 * @param out  Output policy object (non-NULL).
 * @param topo Optional topology handle (may be NULL).
 * @return 0 on success, negative errno-like value on failure (e.g., -EINVAL).
 */
int ie_numa_policy_init(ie_numa_policy_t *out, const ie_topology_t *topo);

/**
 * @brief Decide if a blob of @p bytes should be replicated per socket.
 *
 * @details
 * The decision is based on:
 *  - enable_numa flag,
 *  - replicate_hot flag,
 *  - detected_sockets > 1,
 *  - bytes >= hot_min_bytes.
 *
 * @param p     Policy object (non-NULL).
 * @param bytes Blob size in bytes.
 * @return true if replication should happen, false otherwise.
 */
bool ie_numa_policy_should_replicate_hot(const ie_numa_policy_t *p, size_t bytes);

/**
 * @brief Compute the replica count to build under this policy.
 *
 * @details
 * Typically this returns min(detected_sockets, max_replicas), clamped to at
 * least 1.
 *
 * @param p Policy object (non-NULL).
 * @return Number of replicas to create (>= 1).
 */
int ie_numa_policy_replica_count(const ie_numa_policy_t *p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_OPT_NUMA_POLICY_H */

