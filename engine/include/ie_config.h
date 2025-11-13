/* ============================================================================
 * File: engine/include/ie_config.h
 * ============================================================================
 */
/**
 * @file ie_config.h
 * @brief Tiny, header-only helpers for engine configuration defaults and env overrides.
 *
 * This header centralizes:
 *  - Canonical names for environment variables used by the CLI/harness.
 *  - Safe getters for integers and strings from the environment.
 *  - A small struct (::ie_config_env_t) that aggregates common env-driven knobs.
 *
 * It is header-only by design to avoid adding a new compilation unit.
 * All functions are `static inline` and can be included from multiple TUs.
 */

#ifndef IE_CONFIG_H_
#define IE_CONFIG_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- Canonical environment variable names -------------------------------- */

/** Require IEBIN presence (strict mode) when set to "1". */
#define IE_ENV_REQUIRE_MODEL   "IE_REQUIRE_MODEL"
/** Bytes to touch per token during the measured window (0 disables). */
#define IE_ENV_BYTES_PER_TOKEN "IE_BYTES_PER_TOKEN"
/** Stride in bytes for the work-touch loop. */
#define IE_ENV_STRIDE_BYTES    "IE_STRIDE_BYTES"
/** When non-zero, prevents the compiler from eliding the touch accumulator. */
#define IE_ENV_VERIFY_TOUCH    "IE_VERIFY_TOUCH"
/** Precision label override (fp32|bf16|fp16|int8w|int4|int4w). */
#define IE_ENV_PRECISION       "IE_PRECISION"
/** Compatibility alias for precision; lower precedence than IE_ENV_PRECISION. */
#define IE_ENV_PRECISION_ALT   "PRECISION"

/* ---- Canonical string constants for precision labels --------------------- */

#ifndef IE_PREC_FP32
#  define IE_PREC_FP32  "fp32"
#endif
#ifndef IE_PREC_BF16
#  define IE_PREC_BF16  "bf16"
#endif
#ifndef IE_PREC_FP16
#  define IE_PREC_FP16  "fp16"
#endif
#ifndef IE_PREC_INT8W
#  define IE_PREC_INT8W "int8w"
#endif
#ifndef IE_PREC_INT4W
#  define IE_PREC_INT4W "int4w"
#endif
#ifndef IE_PREC_INT4
#  define IE_PREC_INT4  "int4"
#endif

/* ---- Small configuration aggregate -------------------------------------- */

/**
 * @struct ie_config_env_t
 * @brief Snapshot of environment-driven configuration knobs used by the CLI.
 */
typedef struct ie_config_env {
  int      require_model;     /**< 1 to enforce IEBIN presence, 0 otherwise. */
  size_t   bytes_per_token;   /**< Work-touch bytes per token (0 disables). */
  size_t   stride_bytes;      /**< Work-touch stride in bytes (>=1). */
  int      verify_touch;      /**< Side-effect barrier toggle for touch loop. */
  const char *precision;      /**< Precision label; never NULL (defaults to "fp32"). */
} ie_config_env_t;

/* ---- Helpers ------------------------------------------------------------- */

/**
 * @brief Return an environment variable value or a default when unset/empty.
 *
 * @param name Environment variable name.
 * @param defv Default string when unset/empty.
 * @return Pointer to environment string or @p defv (never NULL).
 */
static inline const char *ie_cfg_env_str(const char *name, const char *defv) {
  const char *s = getenv(name);
  return (s && *s) ? s : defv;
}

/**
 * @brief Parse an environment variable as long; fall back to a default on error.
 *
 * Accepts base-10 only. Overflow/underflow are handled by `strtol` clamping.
 *
 * @param name Environment variable name.
 * @param defv Default value when unset/invalid.
 * @return Parsed value or @p defv.
 */
static inline long ie_cfg_env_long(const char *name, long defv) {
  const char *s = getenv(name);
  if (!s || !*s) return defv;
  char *end = NULL;
  long v = strtol(s, &end, 10);
  return (end && *end == '\0') ? v : defv;
}

/**
 * @brief Load a snapshot of environment-driven configuration into @p out.
 *
 * Resolution order for precision is:
 *  1) `IE_PRECISION` if set/non-empty
 *  2) `PRECISION` if set/non-empty
 *  3) "fp32" otherwise
 *
 * @param out Output struct (must be non-NULL).
 */
static inline void ie_cfg_load_env(ie_config_env_t *out) {
  if (!out) return;
  out->require_model  = (int)ie_cfg_env_long(IE_ENV_REQUIRE_MODEL, 0);
  long bpt            = ie_cfg_env_long(IE_ENV_BYTES_PER_TOKEN, 0);
  long stride         = ie_cfg_env_long(IE_ENV_STRIDE_BYTES, 256);
  out->bytes_per_token= (bpt > 0 ? (size_t)bpt : 0u);
  out->stride_bytes   = (stride > 0 ? (size_t)stride : 1u);
  out->verify_touch   = (int)ie_cfg_env_long(IE_ENV_VERIFY_TOUCH, 0);
  const char *p = ie_cfg_env_str(IE_ENV_PRECISION,
                   ie_cfg_env_str(IE_ENV_PRECISION_ALT, IE_PREC_FP32));
  out->precision = p;
}

#endif /* IE_CONFIG_H_ */
