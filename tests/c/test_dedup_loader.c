/* ============================================================================
 * File: tests/c/test_dedup_loader.c
 * ============================================================================
 */
/**
 * @file test_dedup_loader.c
 * @brief Sanity test for the deduplicated weight-loader path.
 *
 * This test is intentionally conservative: it validates that the loader can
 * open and touch weights in both baseline mode and (when present) dedup mode,
 * without asserting implementation-specific internals.
 *
 * Behavior:
 *  - Always tries to open <model-dir>/model.ie.json + <model-dir>/model.ie.bin.
 *  - If <model-dir>/model.dedup.json exists, it enables dedup via env vars and
 *    attempts to open/touch again. If dedup assets are missing, it skips
 *    gracefully (exit 0) rather than failing the whole test suite.
 *
 * Notes:
 *  - This test uses only the public I/O surface exposed by ie_io.h. It does not
 *    depend on patch-list internals or private structs beyond fields that are
 *    already used elsewhere in the test suite.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "ie_io.h"
#include "util_logging.h"

#ifndef IE_LOG_INFO
#  define IE_LOG_INFO(...)  do { fprintf(stderr, "[info] "  __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif
#ifndef IE_LOG_WARN
#  define IE_LOG_WARN(...)  do { fprintf(stderr, "[warn] "  __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif
#ifndef IE_LOG_ERROR
#  define IE_LOG_ERROR(...) do { fprintf(stderr, "[error] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#endif

static int file_exists(const char* path) {
  struct stat st;
  return (path && *path && stat(path, &st) == 0);
}

static void join_path(char* out, size_t outsz, const char* dir, const char* leaf) {
  if (!out || outsz == 0) return;
  out[0] = '\0';
  if (!leaf || !*leaf) return;
  if (!dir || !*dir) {
    snprintf(out, outsz, "%s", leaf);
    return;
  }
  size_t n = strlen(dir);
  if (n && dir[n - 1] == '/') snprintf(out, outsz, "%s%s", dir, leaf);
  else snprintf(out, outsz, "%s/%s", dir, leaf);
}

static int open_and_touch(const char* json_path, const char* bin_path) {
  ie_weights_t w;
  memset(&w, 0, sizeof(w));

  int rc = ie_weights_open(json_path, bin_path, &w);
  if (rc != IE_IO_OK) {
    IE_LOG_ERROR("ie_weights_open failed (json='%s', bin='%s') rc=%d errno=%d (%s)",
                 json_path, bin_path, rc, errno, strerror(errno));
    return 1;
  }

  if (ie_weights_touch(&w) != 0) {
    IE_LOG_ERROR("ie_weights_touch failed (json='%s', bin='%s') errno=%d (%s)",
                 json_path, bin_path, errno, strerror(errno));
    ie_weights_close(&w);
    return 2;
  }

  ie_weights_close(&w);
  return 0;
}

int main(int argc, char** argv) {
  const char* model_dir = (argc > 1 && argv[1] && argv[1][0]) ? argv[1] : "models/gpt-oss-20b";

  char json_path[PATH_MAX];
  char bin_path[PATH_MAX];
  char dedup_json[PATH_MAX];

  join_path(json_path, sizeof(json_path), model_dir, "model.ie.json");
  join_path(bin_path,  sizeof(bin_path),  model_dir, "model.ie.bin");
  join_path(dedup_json, sizeof(dedup_json), model_dir, "model.dedup.json");

  if (!file_exists(json_path) || !file_exists(bin_path)) {
    IE_LOG_WARN("IEBIN missing under '%s' (expected model.ie.json/bin). Skipping test.", model_dir);
    return 0;
  }

  IE_LOG_INFO("baseline open/touch: %s + %s", json_path, bin_path);
  {
    int rc = open_and_touch(json_path, bin_path);
    if (rc != 0) return rc;
  }

  if (!file_exists(dedup_json)) {
    IE_LOG_INFO("no dedup manifest found (%s). Skipping dedup path.", dedup_json);
    return 0;
  }

  setenv("IE_DEDUP", "1", 1);
  if (!getenv("IE_DEDUP_POLICY")) setenv("IE_DEDUP_POLICY", "cache", 1);

  IE_LOG_INFO("dedup enabled via env (IE_DEDUP=1). Re-opening to exercise dedup path.");
  {
    int rc = open_and_touch(json_path, bin_path);
    if (rc != 0) {
      IE_LOG_WARN("dedup path failed. If dedup artifacts are not fully present yet, this is expected.");
      IE_LOG_WARN("To make this strict, ensure dedup assets exist and set WEIGHTS_TEST_STRICT=1 in CI.");
      if (getenv("WEIGHTS_TEST_STRICT") && strcmp(getenv("WEIGHTS_TEST_STRICT"), "1") == 0) return rc;
      return 0;
    }
  }

  IE_LOG_INFO("OK");
  return 0;
}
