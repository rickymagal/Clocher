/* ============================================================================
 * File: engine/include/ie_topology.h
 * ============================================================================
 * Lightweight CPU topology discovery (sockets and NUMA nodes).
 *
 * The implementation prefers Linux sysfs:
 *   - "/sys/devices/system/cpu/cpu<N>/topology/physical_package_id" to determine
 *     the physical socket (package) id per CPU.
 *   - "/sys/devices/system/node/node<N>/cpulist" to determine NUMA node mapping.
 *
 * If sysfs is unavailable or incomplete, a safe fallback is used:
 *   - 1 socket, 1 NUMA node, all online CPUs belong to both.
 *
 * This header exposes a small, allocation-owning handle and pure getters that
 * never fail after successful initialization.
 * ========================================================================== */

#ifndef IE_TOPOLOGY_H_
#define IE_TOPOLOGY_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle for discovered CPU topology. */
typedef struct ie_topology ie_topology_t;

/**
 * @brief Initialize and build a topology handle.
 *
 * This probes the host to discover:
 *  - total online CPUs
 *  - socket id for each CPU
 *  - NUMA node id for each CPU
 * A best-effort approach is used; missing information is replaced by a
 * consistent single-socket, single-node fallback.
 *
 * @param[out] topo  Output pointer to receive a newly allocated handle (non-NULL).
 * @return 0 on success; negative errno-like value on failure (e.g., -12 for OOM).
 */
int ie_topology_init(ie_topology_t **topo);

/**
 * @brief Free a previously created topology handle.
 *
 * It is safe to pass NULL; no action will be taken.
 *
 * @param topo Topology handle returned by ie_topology_init.
 */
void ie_topology_destroy(ie_topology_t *topo);

/**
 * @brief Return the number of online CPUs observed during discovery.
 *
 * @param topo Topology handle (non-NULL).
 * @return Number of CPUs (>= 1).
 */
int ie_topology_cpus(const ie_topology_t *topo);

/**
 * @brief Return the number of sockets (physical packages).
 *
 * @param topo Topology handle (non-NULL).
 * @return Number of sockets (>= 1).
 */
int ie_topology_sockets(const ie_topology_t *topo);

/**
 * @brief Return the number of NUMA nodes discovered.
 *
 * @param topo Topology handle (non-NULL).
 * @return Number of NUMA nodes (>= 1).
 */
int ie_topology_nodes(const ie_topology_t *topo);

/**
 * @brief Map a CPU id to its socket id.
 *
 * If the CPU id is out of range, returns -1.
 *
 * @param topo Topology handle (non-NULL).
 * @param cpu  Zero-based CPU id.
 * @return Socket id (>= 0), or -1 if invalid.
 */
int ie_topology_cpu_to_socket(const ie_topology_t *topo, int cpu);

/**
 * @brief Map a CPU id to its NUMA node id.
 *
 * If the CPU id is out of range, returns -1.
 *
 * @param topo Topology handle (non-NULL).
 * @param cpu  Zero-based CPU id.
 * @return NUMA node id (>= 0), or -1 if invalid.
 */
int ie_topology_cpu_to_node(const ie_topology_t *topo, int cpu);

/**
 * @brief Return a representative CPU id that belongs to the given socket.
 *
 * This is typically the smallest CPU id observed in that socket during
 * discovery. Useful for pinning one thread per socket.
 *
 * @param topo   Topology handle (non-NULL).
 * @param socket Socket id (0 .. ie_topology_sockets()-1).
 * @return CPU id (>= 0) on success; -1 if @p socket is invalid or empty.
 */
int ie_topology_first_cpu_on_socket(const ie_topology_t *topo, int socket);

/**
 * @brief Bind the current thread/process to a representative CPU of a socket.
 *
 * This is a thin helper: it picks ie_topology_first_cpu_on_socket(@p topo, @p socket)
 * and calls sched_setaffinity() for the current thread/process.
 *
 * @param topo   Topology handle (non-NULL).
 * @param socket Socket id (0 .. ie_topology_sockets()-1).
 * @return 0 on success; negative errno-like value on failure.
 */
int ie_topology_bind_thread_to_socket(const ie_topology_t *topo, int socket);

#ifdef __cplusplus
}
#endif

#endif /* IE_TOPOLOGY_H_ */
