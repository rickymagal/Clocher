#ifndef IE_NUMA_POLICY_H
#define IE_NUMA_POLICY_H

/**
 * @file numa_policy.h
 * @brief NUMA-aware policy snapshot used by subsystems at runtime.
 *
 * @details
 * This policy layer does not perform allocation or placement by itself.
 * It only captures:
 *  - detected sockets (from topology when available),
 *  - detected NUMA nodes (sysfs-based),
 *  - env-driven knobs to enable NUMA behaviors,
 *  - decisions about whether to replicate "hot" blobs.
 */

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ie_topology ie_topology_t;

/**
 * @struct ie_numa_policy_t
 * @brief Immutable policy snapshot (initialized once at startup).
 */
typedef struct ie_numa_policy_t {
  int detected_nodes;
  int detected_sockets;

  int enable_numa;
  int replicate_hot;
  int max_replicas;

  size_t hot_min_bytes;
} ie_numa_policy_t;

/**
 * @brief Initialize a NUMA policy snapshot.
 *
 * @param out Output policy (non-NULL).
 * @param topo Optional topology handle (may be NULL).
 * @return 0 on success, negative errno-like value on failure.
 */
int ie_numa_policy_init(ie_numa_policy_t* out, const ie_topology_t* topo);

/**
 * @brief Decide whether a blob should be replicated per socket.
 *
 * @param p Policy (non-NULL).
 * @param bytes Blob size in bytes.
 * @return true if replication should happen.
 */
bool ie_numa_policy_should_replicate_hot(const ie_numa_policy_t* p, size_t bytes);

/**
 * @brief Compute how many replicas to create under this policy.
 *
 * @param p Policy (non-NULL).
 * @return Replica count (>= 1).
 */
int ie_numa_policy_replica_count(const ie_numa_policy_t* p);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_NUMA_POLICY_H */

