/* ============================================================================
 * File: tests/c/test_topology.c
 * ============================================================================
 * Simple smoke test for the topology API.
 * The test only prints information and returns 0 on success.
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include "ie_topology.h"

/**
 * @brief Dump a short human-readable view of the topology.
 *
 * @param t Initialized topology handle.
 */
static void dump_topology(const ie_topology_t *t) {
  const int n_cpu     = ie_topology_cpus(t);
  const int n_socket  = ie_topology_sockets(t);
  const int n_node    = ie_topology_nodes(t);

  printf("topology: cpus=%d sockets=%d nodes=%d\n", n_cpu, n_socket, n_node);

  for (int s = 0; s < n_socket; ++s) {
    int c0 = ie_topology_first_cpu_on_socket(t, s);
    printf("  socket %d: first-cpu=%d\n", s, c0);
  }

  for (int i = 0; i < n_cpu; ++i) {
    int sid = ie_topology_cpu_to_socket(t, i);
    int nid = ie_topology_cpu_to_node(t, i);
    printf("  cpu %d -> socket %d, node %d\n", i, sid, nid);
  }
}

/**
 * @brief Main test entry: build topology, print, and basic assertions.
 *
 * @return 0 on success; non-zero on failure.
 */
int main(void) {
  ie_topology_t *t = NULL;
  int rc = ie_topology_init(&t);
  if (rc != 0 || !t) {
    fprintf(stderr, "ie_topology_init failed: %d\n", rc);
    return 1;
  }

  /* Basic invariants */
  if (ie_topology_cpus(t) <= 0) {
    fprintf(stderr, "invalid cpu count\n");
    ie_topology_destroy(t);
    return 2;
  }
  if (ie_topology_sockets(t) <= 0) {
    fprintf(stderr, "invalid socket count\n");
    ie_topology_destroy(t);
    return 3;
  }
  if (ie_topology_nodes(t) <= 0) {
    fprintf(stderr, "invalid node count\n");
    ie_topology_destroy(t);
    return 4;
  }

  dump_topology(t);

  ie_topology_destroy(t);
  puts("ok test_topology");
  return 0;
}
