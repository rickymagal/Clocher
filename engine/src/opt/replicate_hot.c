/* ============================================================================
 * File: engine/src/opt/replicate_hot.c
 * ============================================================================
 */
/**
 * @file replicate_hot.c
 * @brief Replication of hot weights per socket using first-touch placement.
 *
 * This unit depends on ie_topology.h for socket enumeration and thread
 * binding. It does not define any thread-binding symbol by itself; it calls
 * ie_topology_bind_thread_to_socket() from topology.c.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__linux__) || defined(__APPLE__)
  #include <sys/mman.h> /* posix_madvise / MADV_* if present */
#endif

#include "ie_topology.h"
#include "util_logging.h"

#ifndef IE_LIKELY
#  define IE_LIKELY(x)   __builtin_expect(!!(x), 1)
#endif
#ifndef IE_UNLIKELY
#  define IE_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

/* ----- logging fallbacks (in case util_logging.h provides none) ----------- */
#ifndef IE_LOG_INFO
#  define IE_LOG_INFO(...)  do { fprintf(stderr, "[info] "  __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif
#ifndef IE_LOG_WARN
#  define IE_LOG_WARN(...)  do { fprintf(stderr, "[warn] "  __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif
#ifndef IE_LOG_ERROR
#  define IE_LOG_ERROR(...) do { fprintf(stderr, "[error] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif

/**
 * @struct ie_hot_replica_t
 * @brief A single per-socket replica.
 *
 * The memory is owned by the replica set and must be freed via
 * ie_hot_replicas_free().
 */
typedef struct ie_hot_replica_t {
  void*  ptr;        /**< Base address of the replica, or NULL if missing. */
  size_t size;       /**< Size in bytes. */
  int    socket_id;  /**< Socket that owns this replica. */
} ie_hot_replica_t;

/**
 * @struct ie_hot_replicas_t
 * @brief Container of all per-socket replicas.
 *
 * The array length equals the number of sockets used to build the set.
 */
typedef struct ie_hot_replicas_t {
  int               n_sockets;  /**< Number of sockets. */
  ie_hot_replica_t* r;          /**< Array length n_sockets. */
} ie_hot_replicas_t;

/* ------------------------------ local helpers ------------------------------ */
/**
 * @brief First-touch a writable buffer to bias NUMA page placement.
 *
 * Writes one byte per cache line across the region to ensure resident pages
 * are backed by the NUMA node of the current CPU affinity.
 *
 * @param p   Pointer to writable memory (non-NULL).
 * @param sz  Size in bytes.
 */
static void first_touch_write(void* p, size_t sz) {
  volatile uint8_t* b = (volatile uint8_t*)p;
  const size_t step = 64;
  for (size_t i = 0; i < sz; i += step) b[i] = 0u;
  if (sz) b[sz - 1] = 0u;
}

/**
 * @brief Best-effort hint to the OS that the buffer will be needed soon.
 *
 * Uses POSIX readahead/madvise when available. If the platform does not expose
 * a declaration for posix_madvise (or the specific advice constant), this
 * function degrades to a no-op.
 *
 * @param p   Pointer to memory (non-NULL).
 * @param sz  Size in bytes.
 */
static void advise_willneed(void* p, size_t sz) {
  (void)p; (void)sz;
#if defined(POSIX_MADV_WILLNEED)
  (void)posix_madvise(p, sz, POSIX_MADV_WILLNEED);
#elif defined(__linux__) && defined(MADV_WILLNEED)
  /* Fallback to Linux-specific madvise if available in headers. */
  (void)madvise(p, sz, MADV_WILLNEED);
#else
  /* No advisory available on this platform. */
  (void)p; (void)sz;
#endif
}

/* ------------------------------- public API -------------------------------- */
/**
 * @brief Create per-socket replicas of @p blob using first-touch placement.
 *
 * For each socket:
 *   1) bind to that socket via ie_topology_bind_thread_to_socket();
 *   2) allocate replica memory and first-touch it;
 *   3) memcpy() the source blob;
 *   4) hint the kernel that pages will be needed (best-effort).
 *
 * @param topo       Topology handle (non-NULL recommended).
 * @param n_sockets  Sockets to build (if <= 0, derived from @p topo; if topo
 *                   is NULL or invalid, defaults to 1).
 * @param blob       Source blob (non-NULL).
 * @param blob_size  Size of @p blob in bytes (> 0).
 * @param out        Output pointer to receive the replicas object (non-NULL).
 * @return 0 on success; negative errno-like on failure.
 */
int ie_hot_replicate_build(const ie_topology_t* topo,
                           int n_sockets,
                           const void* blob,
                           size_t blob_size,
                           ie_hot_replicas_t** out)
{
  if (!out || !blob || blob_size == 0) return -EINVAL;

  int S = n_sockets;
  if (S <= 0 && topo) S = ie_topology_sockets(topo);
  if (S <= 0) S = 1;

  ie_hot_replicas_t* set = (ie_hot_replicas_t*)calloc(1, sizeof(*set));
  if (!set) return -ENOMEM;

  set->n_sockets = S;
  set->r = (ie_hot_replica_t*)calloc((size_t)S, sizeof(*set->r));
  if (!set->r) { free(set); return -ENOMEM; }

  int rc = 0;
  for (int s = 0; s < S; ++s) {
    /* Bind for first-touch locality (best-effort). */
    int brc = ie_topology_bind_thread_to_socket(topo, s);
    if (brc != 0) IE_LOG_WARN("binding to socket %d failed (%d)", s, brc);

    void* buf = malloc(blob_size);
    if (!buf) { rc = -ENOMEM; goto fail; }

    first_touch_write(buf, blob_size);
    memcpy(buf, blob, blob_size);
    advise_willneed(buf, blob_size);

    set->r[s].ptr       = buf;
    set->r[s].size      = blob_size;
    set->r[s].socket_id = s;
  }

  *out = set;
  return 0;

fail:
  if (set) {
    if (set->r) {
      for (int s = 0; s < S; ++s) free(set->r[s].ptr);
      free(set->r);
    }
    free(set);
  }
  return rc ? rc : -ENOMEM;
}

/**
 * @brief Free replicas previously built by ie_hot_replicate_build().
 *
 * Releases all memory owned by the replica set.
 *
 * @param reps  Replicas object (NULL is allowed).
 */
void ie_hot_replicas_free(ie_hot_replicas_t* reps) {
  if (!reps) return;
  if (reps->r) {
    for (int s = 0; s < reps->n_sockets; ++s) free(reps->r[s].ptr);
    free(reps->r);
  }
  free(reps);
}

/**
 * @brief Retrieve the replica pointer for a socket.
 *
 * @param reps       Replicas object (non-NULL).
 * @param socket_id  Socket id (>= 0).
 * @param size_out   Optional size return; may be NULL.
 * @return Pointer to replica memory or NULL on error.
 */
void* ie_hot_replica_for_socket(const ie_hot_replicas_t* reps,
                                int socket_id,
                                size_t* size_out)
{
  if (!reps || socket_id < 0 || socket_id >= reps->n_sockets) return NULL;
  if (size_out) *size_out = reps->r[socket_id].size;
  return reps->r[socket_id].ptr;
}

/**
 * @brief Convenience: memcpy() into a given socket’s replica.
 *
 * Copies exactly the size of the existing replica buffer for @p socket_id.
 *
 * @param reps       Replicas object (non-NULL).
 * @param socket_id  Target socket id (>= 0).
 * @param src        Source bytes (non-NULL). Must be at least replica size.
 * @return 0 on success; -EINVAL on bad args; -ENOBUFS on size mismatch.
 */
int ie_hot_replica_memcpy(ie_hot_replicas_t* reps, int socket_id, const void* src) {
  if (!reps || !src || socket_id < 0 || socket_id >= reps->n_sockets) return -EINVAL;
  void* dst = reps->r[socket_id].ptr;
  if (!dst) return -EINVAL;
  if (reps->r[socket_id].size == 0) return -ENOBUFS;
  memcpy(dst, src, reps->r[socket_id].size);
  return 0;

}
