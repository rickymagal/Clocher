/**
 * @file ie_numa.h
 * @brief Lightweight NUMA probing via sysfs (Linux only), no external deps.
 *
 * This header declares a minimal NUMA-detection helper that reads
 * `/sys/devices/system/node/online` to infer the number of NUMA nodes.
 * On non-Linux platforms, the helper degrades gracefully to a single node.
 */

#ifndef IE_NUMA_H_
#define IE_NUMA_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Detect the number of NUMA nodes available on the system.
 *
 * On Linux, this function reads `/sys/devices/system/node/online` and parses
 * the CPU-node range list (e.g., "0", "0-1", "0-3,8-9") to determine the
 * highest node id, returning `max_id + 1`. If the file is not present or
 * cannot be read, the function returns 1.
 *
 * On non-Linux systems, the function always returns 1.
 *
 * @return Detected NUMA node count (>= 1). Returns 1 on error or non-Linux OS.
 */
int ie_numa_detect_nodes(void);

#ifdef __cplusplus
}
#endif

#endif /* IE_NUMA_H_ */
