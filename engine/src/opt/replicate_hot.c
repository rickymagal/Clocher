/* ============================================================================
 * File: engine/src/opt/replicate_hot.c
 * ============================================================================
 */
/**
 * @file replicate_hot.c
 * @brief Replication of hot weight blobs per socket using first-touch placement.
 *
 * This module implements a small utility to create per-socket replicas of a
 * contiguous "hot" blob so that the inference kernel can read local memory
 * when threads are socket-pinned.
 *
 * Step-3 (Runtime reconstruction path) integration:
 *  - This unit now supports two build modes:
 *      1) Copy mode: replicate an existing source blob via memcpy().
 *      2) Fill mode: allocate per-socket memory, first-touch it, then let a
 *         caller-provided callback *populate* the replica in-place.
 *
 * Fill mode exists to avoid a common NUMA pitfall:
 *  - If you reconstruct/assemble the hot blob once on socket 0 and then memcpy
 *    it to other sockets, you pay cross-socket bandwidth and potentially pull
 *    pages into the wrong node before first-touch.
 *
 * The module depends on ie_topology.h for socket enumeration and thread
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

/**
 * @typedef ie_hot_replica_fill_fn
 * @brief Callback type for fill-mode replica construction.
 *
 * The callback is invoked once per socket after:
 *  - the current thread has been (best-effort) bound to that socket,
 *  - memory has been allocated and first-touched for locality.
 *
 * The callback must fully initialize the provided destination buffer.
 *
 * @param socket_id Socket id being built.
 * @param dst Destination buffer for the replica.
 * @param dst_size Destination buffer size in bytes.
 * @param user User context pointer provided to the builder.
 * @return 0 on success, negative errno-like on failure.
 */
typedef int (*ie_hot_replica_fill_fn)(int socket_id, void* dst, size_t dst_size, void* user);

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
  (void)madvise(p, sz, MADV_WILLNEED);
#else
  (void)p; (void)sz;
#endif
}

/**
 * @brief Best-effort hint to the OS that sequential access is expected.
 *
 * This can help some kernels choose better readahead / paging heuristics.
 *
 * @param p Pointer to memory.
 * @param sz Size in bytes.
 */
static void advise_sequential(void* p, size_t sz) {
  (void)p; (void)sz;
#if defined(POSIX_MADV_SEQUENTIAL)
  (void)posix_madvise(p, sz, POSIX_MADV_SEQUENTIAL);
#elif defined(__linux__) && defined(MADV_SEQUENTIAL)
  (void)madvise(p, sz, MADV_SEQUENTIAL);
#else
  (void)p; (void)sz;
#endif
}

/**
 * @brief Allocate a replica buffer and establish locality via first-touch.
 *
 * @param blob_size Size in bytes (> 0).
 * @return Pointer to allocated memory, or NULL on failure.
 */
static void* alloc_replica_first_touch(size_t blob_size) {
  void* buf = malloc(blob_size);
  if (!buf) return NULL;
  first_touch_write(buf, blob_size);
  return buf;
}

/**
 * @brief Free all allocations owned by a replica set (internal helper).
 *
 * @param reps Replica set (may be NULL).
 */
static void free_replicas_internal(ie_hot_replicas_t* reps) {
  if (!reps) return;
  if (reps->r) {
    for (int s = 0; s < reps->n_sockets; ++s) {
      free(reps->r[s].ptr);
      reps->r[s].ptr = NULL;
      reps->r[s].size = 0;
      reps->r[s].socket_id = 0;
    }
    free(reps->r);
    reps->r = NULL;
  }
  free(reps);
}

/* ------------------------------- public API -------------------------------- */

/**
 * @brief Create per-socket replicas of @p blob using first-touch placement (copy mode).
 *
 * For each socket:
 *   1) bind to that socket via ie_topology_bind_thread_to_socket();
 *   2) allocate replica memory and first-touch it;
 *   3) memcpy() the source blob;
 *   4) hint the kernel that pages will be needed (best-effort).
 *
 * This is a good default when the source blob is small enough or already hot,
 * and when you do not have a better per-socket fill path.
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
    int brc = ie_topology_bind_thread_to_socket(topo, s);
    if (brc != 0) IE_LOG_WARN("binding to socket %d failed (%d)", s, brc);

    void* buf = alloc_replica_first_touch(blob_size);
    if (!buf) { rc = -ENOMEM; goto fail; }

    memcpy(buf, blob, blob_size);
    advise_willneed(buf, blob_size);
    advise_sequential(buf, blob_size);

    set->r[s].ptr       = buf;
    set->r[s].size      = blob_size;
    set->r[s].socket_id = s;
  }

  *out = set;
  return 0;

fail:
  free_replicas_internal(set);
  return rc ? rc : -ENOMEM;
}

/**
 * @brief Create per-socket replicas using a caller-provided fill callback (fill mode).
 *
 * This is the preferred mode for Step-3 persistent reconstruction:
 * you can reconstruct/assemble the "hot weights blob" directly into the local
 * per-socket replica after first-touch, avoiding cross-socket memcpy traffic.
 *
 * For each socket:
 *   1) bind to that socket via ie_topology_bind_thread_to_socket();
 *   2) allocate replica memory and first-touch it;
 *   3) call @p fill(socket_id, replica_ptr, replica_size, user);
 *   4) hint the kernel that pages will be needed (best-effort).
 *
 * @param topo       Topology handle (non-NULL recommended).
 * @param n_sockets  Sockets to build (if <= 0, derived from @p topo; if topo
 *                   is NULL or invalid, defaults to 1).
 * @param replica_size Size in bytes of each replica (> 0).
 * @param fill       Callback to populate each per-socket replica (non-NULL).
 * @param user       Opaque user pointer passed to @p fill.
 * @param out        Output pointer to receive the replicas object (non-NULL).
 * @return 0 on success; negative errno-like on failure.
 */
int ie_hot_replicate_build_fill(const ie_topology_t* topo,
                                int n_sockets,
                                size_t replica_size,
                                ie_hot_replica_fill_fn fill,
                                void* user,
                                ie_hot_replicas_t** out)
{
  if (!out || !fill || replica_size == 0) return -EINVAL;

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
    int brc = ie_topology_bind_thread_to_socket(topo, s);
    if (brc != 0) IE_LOG_WARN("binding to socket %d failed (%d)", s, brc);

    void* buf = alloc_replica_first_touch(replica_size);
    if (!buf) { rc = -ENOMEM; goto fail; }

    rc = fill(s, buf, replica_size, user);
    if (rc != 0) {
      IE_LOG_ERROR("fill callback failed for socket %d (%d)", s, rc);
      free(buf);
      goto fail;
    }

    advise_willneed(buf, replica_size);
    advise_sequential(buf, replica_size);

    set->r[s].ptr       = buf;
    set->r[s].size      = replica_size;
    set->r[s].socket_id = s;
  }

  *out = set;
  return 0;

fail:
  free_replicas_internal(set);
  return rc ? rc : -ENOMEM;
}

/**
 * @brief Free replicas previously built by ie_hot_replicate_build() or ie_hot_replicate_build_fill().
 *
 * Releases all memory owned by the replica set.
 *
 * @param reps Replicas object (NULL is allowed).
 */
void ie_hot_replicas_free(ie_hot_replicas_t* reps) {
  free_replicas_internal(reps);
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
 * @return 0 on success; -EINVAL on bad args; -ENOBUFS if replica size is zero.
 */
int ie_hot_replica_memcpy(ie_hot_replicas_t* reps, int socket_id, const void* src) {
  if (!reps || !src || socket_id < 0 || socket_id >= reps->n_sockets) return -EINVAL;
  void* dst = reps->r[socket_id].ptr;
  if (!dst) return -EINVAL;
  if (reps->r[socket_id].size == 0) return -ENOBUFS;
  memcpy(dst, src, reps->r[socket_id].size);
  return 0;
}
