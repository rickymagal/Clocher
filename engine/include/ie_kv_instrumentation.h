/* ========================================================================== */
/* File: engine/include/ie_kv_instrumentation.h                               */
/* ========================================================================== */
/**
 * @file ie_kv_instrumentation.h
 * @brief Lightweight KV cache reuse counters (process-local, single-threaded).
 *
 * @details
 * These counters are intended to answer one practical question:
 *   "How much previously-computed K/V data did we reuse while generating?"
 *
 * The engine may call ::ie_kv_add_hits() with the number of cached K/V entries
 * consumed by an attention operation, and ::ie_kv_add_misses() with the number
 * of newly-produced K/V entries (typically 1 per token per layer).
 *
 * The counters are process-local and not thread-safe.
 */

#ifndef IE_KV_INSTRUMENTATION_H
#define IE_KV_INSTRUMENTATION_H

#include <stdint.h>

/**
 * @brief Deprecated legacy hook (no-op).
 *
 * @details
 * Older builds used this hook during token generation. The current
 * instrumentation counts reuse explicitly via ie_kv_add_hits/ie_kv_add_misses.
 * This function is kept only to avoid breaking downstream callers.
 *
 * @param token Unused.
 */
void ie_kv_on_token(uint32_t token);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Begin (or reset) KV instrumentation for the current round.
 *
 * @return 0 on success.
 */
int ie_kv_begin_round(void);

/**
 * @brief Add to the KV "hit" counter.
 *
 * @param n Number of reused (cached) K/V entries.
 */
void ie_kv_add_hits(uint64_t n);

/**
 * @brief Add to the KV "miss" counter.
 *
 * @param n Number of newly-produced K/V entries.
 */
void ie_kv_add_misses(uint64_t n);

/**
 * @brief Finish a round and return the accumulated counters.
 *
 * @param total_tokens Number of tokens in the round, used only as a fallback.
 * @param out_hits     Output hit count (must be non-NULL).
 * @param out_misses   Output miss count (must be non-NULL).
 */
void ie_kv_finish_round(uint64_t total_tokens,
                        uint64_t *out_hits,
                        uint64_t *out_misses);

/**
 * @brief Deprecated legacy hook (no-op).
 *
 * @details
 * Older builds used this hook during token generation. The current
 * instrumentation counts reuse explicitly via ie_kv_add_hits/ie_kv_add_misses.
 * This function is kept only to avoid breaking downstream callers.
 *
 * @param token Unused.
 */
void ie_kv_on_token(uint32_t token);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_KV_INSTRUMENTATION_H */

/* ========================================================================== */
/* End of file                                                                */
/* ========================================================================== */
