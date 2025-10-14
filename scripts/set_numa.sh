#!/usr/bin/env bash
# NUMA helper for running the inference engine with a chosen policy.
# Usage:
#   scripts/set_numa.sh compact   -- <cmd> [args...]
#   scripts/set_numa.sh interleave -- <cmd> [args...]
#   scripts/set_numa.sh node:0    -- <cmd> [args...]

set -euo pipefail

MODE="${1:-compact}"
shift || true
if [[ "${1:-}" != "--" ]]; then
  echo "usage: $0 {compact|interleave|node:X} -- <command> [args...]" >&2
  exit 2
fi
shift

if ! command -v numactl >/dev/null 2>&1; then
  echo "[warn] numactl not found; running without NUMA policy." >&2
  exec "$@"
fi

case "$MODE" in
  compact|auto)
    exec numactl --cpunodebind=0 --membind=0 "$@"
    ;;
  interleave)
    exec numactl --interleave=all "$@"
    ;;
  node:*)
    NODE="${MODE#node:}"
    exec numactl --cpunodebind="${NODE}" --membind="${NODE}" "$@"
    ;;
  *)
    echo "invalid NUMA mode: $MODE" >&2
    exit 2
    ;;
esac
