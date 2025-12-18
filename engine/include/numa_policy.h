#ifndef IE_OPT_NUMA_POLICY_H
#define IE_OPT_NUMA_POLICY_H

/**
 * @file numa_policy.h
 * @brief Runtime NUMA policy knobs used by the loader and hot-weights replication logic.
 *
 * This header defines a small "policy" object that:
 *  - captures detected topology facts (sockets, NUMA nodes),
 *  - reads environment-based knobs,
 *  - provides helper decisions (e.g., whether to replicate a hot blob).
 *
 * It is intentionally lightweight and does not depend on libnuma.
 */

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
 * Environment variables:
 *  - IE_NUMA_ENABLE            (default: 1)
 *  - IE_NUMA_REPLICATE_HOT     (default: 1)
 *  - IE_NUMA_MAX_REPLICAS      (default: detected_sockets, clamped to [1, 64])
 *  - IE_NUMA_HOT_MIN_BYTES     (default: 33554432 = 32 MiB)
 *
 * @param out  Output policy object (non-NULL).
 * @param topo Optional topology handle (may be NULL).
 * @return 0 on success, negative errno-like on failure.
 */
int ie_numa_policy_init(ie_numa_policy_t *out, const ie_topology_t *topo);

/**
 * @brief Decide if a blob of @p bytes should be replicated per socket.
 *
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
 * Typically this is min(detected_sockets, max_replicas), and at least 1.
 *
 * @param p Policy object (non-NULL).
 * @return Number of replicas to create (>= 1).
 */
int ie_numa_policy_replica_count(const ie_numa_policy_t *p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_OPT_NUMA_POLICY_H */

