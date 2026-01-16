/**
 * @file ie_cli_extras.h
 * @brief Small, self-contained CLI extras shared across benchmark-style entrypoints.
 *
 * This header intentionally defines only the "extras" flags that are common across
 * harness entrypoints (batching, prompts file, rounds, and optional expected-token
 * verification). It does not impose a particular argument parser implementation.
 *
 * Expected tokens file format (one entry per line):
 *   <prompt_id><whitespace><token0,token1,token2,...>
 *
 * - Lines starting with '#' and empty lines are ignored.
 * - prompt_id can be decimal or 0x-prefixed hexadecimal.
 * - prompt_id is a stable 64-bit FNV-1a hash of the prompt bytes (UTF-8).
 *
 * The intent is to allow a strict harness to verify that an engine build is stable
 * (token IDs are identical) across runs and across environments.
 */

#ifndef IE_CLI_EXTRAS_H
#define IE_CLI_EXTRAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Common "extras" CLI configuration used by inference entrypoints.
 */
typedef struct ie_cli_extras {
  const char *prompts_file;         /**< Prompts file path (1 prompt per line). */
  int batch;                        /**< Batch size (if supported by the entrypoint). */
  const char *prefetch;             /**< Prefetch policy string (on|off|auto|N). */
  int warmup_tokens;                /**< Warmup token count (0 disables warmup). */
  int aggregate;                    /**< If nonzero, run all prompts from prompts_file. */
  int rounds;                       /**< Number of rounds over the prompt set (>=1). */

  const char *expected_tokens_file; /**< Optional expected tokens file (golden output). */

  int report_tokens;                /**< If nonzero, include token IDs in the JSON report. */
  size_t report_tokens_max;         /**< Max tokens to report per prompt/round (0 keeps all). */
} ie_cli_extras_t;

/**
 * @brief Initialize an @ref ie_cli_extras_t with safe defaults.
 * @param e Extras struct to initialize.
 */
static inline void ie_cli_extras_init(ie_cli_extras_t *e) {
  if (!e) return;
  e->prompts_file = NULL;
  e->batch = 1;
  e->prefetch = "auto";
  e->warmup_tokens = 1;
  e->aggregate = 0;
  e->rounds = 1;

  e->expected_tokens_file = NULL;
  e->report_tokens = 1;
  e->report_tokens_max = 0;
}

#ifdef __cplusplus
}
#endif

#endif /* IE_CLI_EXTRAS_H */
