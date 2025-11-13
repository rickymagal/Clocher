/* ============================================================================
 * File: engine/src/opt/topology.c
 * ============================================================================
 * Implementation of CPU topology discovery (sockets and NUMA nodes).
 * See: engine/include/ie_topology.h
 * ========================================================================== */

#define _GNU_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include "ie_topology.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sched.h>   /* CPU_SET/CPU_ZERO, sched_setaffinity */

/* ------------------------------ Data model -------------------------------- */

/** Internal representation. */
struct ie_topology {
  int n_cpus;
  int n_sockets;
  int n_nodes;
  int *cpu_to_socket;     /* length n_cpus, -1 if unknown */
  int *cpu_to_node;       /* length n_cpus, -1 if unknown */
  int *first_cpu_socket;  /* length n_sockets, -1 if empty (should not happen) */
};

/* ------------------------------ Utilities --------------------------------- */

/**
 * @brief Return number of online CPUs using sysconf.
 *
 * @return CPUs (>=1) on success; 1 as a conservative fallback.
 */
static int get_online_cpus(void) {
  long onln = sysconf(_SC_NPROCESSORS_ONLN);
  if (onln <= 0) return 1;
  if (onln > INT_MAX) onln = INT_MAX;
  return (int)onln;
}

/**
 * @brief Read an integer from a text file.
 *
 * Accepts trailing newline. Ignores leading whitespace.
 *
 * @param path File path.
 * @param out  Output integer.
 * @return 0 on success; negative errno-like on error.
 */
static int read_int_file(const char *path, int *out) {
  FILE *f = fopen(path, "r");
  if (!f) return -errno ? -errno : -1;
  char buf[64];
  if (!fgets(buf, sizeof(buf), f)) {
    int e = ferror(f) ? -errno : -1;
    fclose(f);
    return e ? e : -1;
  }
  fclose(f);
  char *end = NULL;
  long v = strtol(buf, &end, 10);
  if (end == buf) return -EINVAL;
  *out = (int)v;
  return 0;
}

/**
 * @brief Apply a Linux cpulist string (e.g., "0-3,8,10-11") to mark node for CPUs.
 *
 * @param s            cpulist string (NUL-terminated).
 * @param node_id      Node id to assign.
 * @param cpu2node     Array of length cap; will be set to node_id for listed CPUs.
 * @param cap          Array capacity (number of CPUs).
 */
static void apply_cpulist_to_node(const char *s, int node_id, int *cpu2node, int cap) {
  const char *p = s ? s : "";
  while (*p) {
    /* skip separators and whitespace */
    while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' || *p == '\r') ++p;
    if (!*p) break;
    char *end = NULL;
    long a = strtol(p, &end, 10);
    if (end == p) { /* skip invalid token */
      while (*p && *p != ',' && *p != '\n' && *p != '\r') ++p;
      continue;
    }
    p = end;
    long b = a;
    if (*p == '-') {
      ++p;
      b = strtol(p, &end, 10);
      if (end == p) b = a; /* treat lone '-' weirdness as single */
      p = end;
    }
    if (a < 0) a = 0;
    if (b < 0) b = 0;
    if (a > b) { long t = a; a = b; b = t; }
    for (long i = a; i <= b; ++i) {
      if (i >= 0 && i < cap) cpu2node[(int)i] = node_id;
    }
  }
}

/**
 * @brief Discover sockets using per-CPU physical_package_id in sysfs.
 *
 * Probes "/sys/devices/system/cpu/cpu<N>/topology/physical_package_id"
 * for N = 0..n_cpus-1.
 *
 * @param n_cpus       Number of CPUs to probe.
 * @param cpu2socket   Output array length n_cpus (initialized to -1 before call).
 * @return Count of distinct sockets (>=0). 0 means not found.
 */
static int discover_sockets_sysfs(int n_cpus, int *cpu2socket) {
  int max_socket_id = -1;
  for (int cpu = 0; cpu < n_cpus; ++cpu) {
    char path[256];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    int sid = -1;
    if (read_int_file(path, &sid) == 0 && sid >= 0) {
      cpu2socket[cpu] = sid;
      if (sid > max_socket_id) max_socket_id = sid;
    }
  }
  if (max_socket_id < 0) return 0;
  return max_socket_id + 1;
}

/**
 * @brief Discover NUMA nodes by parsing node cpulist files in sysfs.
 *
 * Probes "/sys/devices/system/node/node<N>/cpulist" for existing node
 * directories and applies their ranges to the cpu->node map.
 *
 * Missing directories or files are tolerated. Unknown entries are ignored.
 *
 * @param n_cpus    Number of CPUs to consider.
 * @param cpu2node  Output array length n_cpus (initialized to -1 before call).
 * @return Count of nodes discovered (>=0). 0 means not found.
 */
static int discover_nodes_sysfs(int n_cpus, int *cpu2node) {
  const char *root = "/sys/devices/system/node";
  DIR *d = opendir(root);
  if (!d) return 0;

  int max_node_id = -1;
  struct dirent *de;
  while ((de = readdir(d)) != NULL) {
    if (strncmp(de->d_name, "node", 4) != 0) continue;
    char *end = NULL;
    long nid = strtol(de->d_name + 4, &end, 10);
    if (end == de->d_name + 4 || nid < 0 || nid > INT_MAX) continue;

    char path[512];
    snprintf(path, sizeof(path), "%s/%s/cpulist", root, de->d_name);

    FILE *f = fopen(path, "r");
    if (!f) continue;

    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    apply_cpulist_to_node(buf, (int)nid, cpu2node, n_cpus);
    if ((int)nid > max_node_id) max_node_id = (int)nid;
  }
  closedir(d);

  if (max_node_id < 0) return 0;
  /* Normalize: nodes are 0..max_node_id, but some may be sparse; still fine. */
  return max_node_id + 1;
}

/**
 * @brief Compute a small representative CPU per socket.
 *
 * @param n_cpus        Number of CPUs.
 * @param n_sockets     Number of sockets.
 * @param cpu2socket    Array length n_cpus.
 * @param out_first     Output length n_sockets; filled with first CPU id or -1.
 */
static void compute_first_cpu_per_socket(int n_cpus, int n_sockets,
                                         const int *cpu2socket, int *out_first) {
  for (int s = 0; s < n_sockets; ++s) out_first[s] = -1;
  for (int cpu = 0; cpu < n_cpus; ++cpu) {
    int s = cpu2socket[cpu];
    if (s >= 0 && s < n_sockets) {
      if (out_first[s] == -1 || cpu < out_first[s]) out_first[s] = cpu;
    }
  }
}

/* ------------------------------ Public API -------------------------------- */

int ie_topology_init(ie_topology_t **out_topo) {
  if (!out_topo) return -EINVAL;

  int n_cpus = get_online_cpus();

  ie_topology_t *t = (ie_topology_t *)calloc(1, sizeof(*t));
  if (!t) return -ENOMEM;

  t->n_cpus = n_cpus;

  t->cpu_to_socket = (int *)malloc(sizeof(int) * n_cpus);
  t->cpu_to_node   = (int *)malloc(sizeof(int) * n_cpus);
  if (!t->cpu_to_socket || !t->cpu_to_node) {
    free(t->cpu_to_socket);
    free(t->cpu_to_node);
    free(t);
    return -ENOMEM;
  }
  for (int i = 0; i < n_cpus; ++i) {
    t->cpu_to_socket[i] = -1;
    t->cpu_to_node[i] = -1;
  }

  /* Best-effort discovery */
  int sockets = discover_sockets_sysfs(n_cpus, t->cpu_to_socket);
  int nodes   = discover_nodes_sysfs(n_cpus, t->cpu_to_node);

  if (sockets <= 0) {
    sockets = 1;
    for (int i = 0; i < n_cpus; ++i) t->cpu_to_socket[i] = 0;
  }
  if (nodes <= 0) {
    nodes = 1;
    for (int i = 0; i < n_cpus; ++i) t->cpu_to_node[i] = 0;
  }
  t->n_sockets = sockets;
  t->n_nodes   = nodes;

  t->first_cpu_socket = (int *)malloc(sizeof(int) * sockets);
  if (!t->first_cpu_socket) {
    free(t->cpu_to_socket);
    free(t->cpu_to_node);
    free(t);
    return -ENOMEM;
  }
  compute_first_cpu_per_socket(n_cpus, sockets, t->cpu_to_socket, t->first_cpu_socket);

  *out_topo = t;
  return 0;
}

/**
 * @brief Free topology handle created by ie_topology_init().
 *
 * @param topo Topology handle (nullable).
 */
void ie_topology_destroy(ie_topology_t *topo) {
  if (!topo) return;
  free(topo->cpu_to_socket);
  free(topo->cpu_to_node);
  free(topo->first_cpu_socket);
  free(topo);
}

/**
 * @brief Get number of CPUs.
 * @param topo Handle (non-NULL).
 * @return CPU count (>=1).
 */
int ie_topology_cpus(const ie_topology_t *topo) {
  return topo ? topo->n_cpus : 0;
}

/**
 * @brief Get number of sockets.
 * @param topo Handle (non-NULL).
 * @return Socket count (>=1).
 */
int ie_topology_sockets(const ie_topology_t *topo) {
  return topo ? topo->n_sockets : 0;
}

/**
 * @brief Get number of NUMA nodes.
 * @param topo Handle (non-NULL).
 * @return Node count (>=1).
 */
int ie_topology_nodes(const ie_topology_t *topo) {
  return topo ? topo->n_nodes : 0;
}

/**
 * @brief CPU -> socket mapping.
 * @param topo Handle (non-NULL).
 * @param cpu  Zero-based CPU id.
 * @return Socket id (>=0) or -1 if invalid.
 */
int ie_topology_cpu_to_socket(const ie_topology_t *topo, int cpu) {
  if (!topo || cpu < 0 || cpu >= topo->n_cpus) return -1;
  return topo->cpu_to_socket[cpu];
}

/**
 * @brief CPU -> NUMA node mapping.
 * @param topo Handle (non-NULL).
 * @param cpu  Zero-based CPU id.
 * @return Node id (>=0) or -1 if invalid.
 */
int ie_topology_cpu_to_node(const ie_topology_t *topo, int cpu) {
  if (!topo || cpu < 0 || cpu >= topo->n_cpus) return -1;
  return topo->cpu_to_node[cpu];
}

/**
 * @brief First (smallest id) CPU present on a given socket.
 * @param topo   Handle (non-NULL).
 * @param socket Socket id.
 * @return CPU id (>=0) or -1 on invalid socket.
 */
int ie_topology_first_cpu_on_socket(const ie_topology_t *topo, int socket) {
  if (!topo || socket < 0 || socket >= topo->n_sockets) return -1;
  return topo->first_cpu_socket[socket];
}

/**
 * @brief Bind the current thread/process to a representative CPU of a socket.
 *
 * Picks ie_topology_first_cpu_on_socket() and applies sched_setaffinity().
 *
 * @param topo   Handle (non-NULL).
 * @param socket Socket id (0..n_sockets-1).
 * @return 0 on success; negative errno-like on failure.
 */
int ie_topology_bind_thread_to_socket(const ie_topology_t *topo, int socket) {
  if (!topo) return -EINVAL;
  if (socket < 0 || socket >= topo->n_sockets) return -EINVAL;

  int cpu = ie_topology_first_cpu_on_socket(topo, socket);
  if (cpu < 0) return -EINVAL;

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET((unsigned)cpu, &set);

  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    return -errno ? -errno : -1;
  }
  return 0;
}

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
