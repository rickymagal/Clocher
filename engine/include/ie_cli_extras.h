/**
 * @file ie_cli_extras.h
 * @brief Optional CLI extras for Step 6: batching & warmup flags.
 */
#ifndef IE_CLI_EXTRAS_H
#define IE_CLI_EXTRAS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Parsed extras for I/O & batching. */
typedef struct ie_cli_extras {
  size_t batch;        /**< Microbatch size: number of prompts processed per slice. */
  size_t prefetch;     /**< Number of background tokenization workers. */
  size_t warmup;       /**< Number of warmup tokens to generate before timing. */
  const char *prompts_file; /**< Optional file path for standardized prompts. */
} ie_cli_extras_t;

/**
 * @brief Populate defaults for CLI extras.
 *
 * @param e Output struct to initialize.
 */
static inline void ie_cli_extras_defaults(ie_cli_extras_t *e) {
  if (!e) return;
  e->batch = 1;
  e->prefetch = 0;  /* 0 => auto = max(1, threads/2) inside main */
  e->warmup = 0;
  e->prompts_file = NULL;
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IE_CLI_EXTRAS_H */
