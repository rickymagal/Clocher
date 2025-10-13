#!/usr/bin/env bash
set -euo pipefail
BIN="${1:-build/inference-engine}"
if ! command -v perf >/dev/null; then echo "perf not installed (skipping)"; exit 0; fi
perf record -g -- "$BIN" "Warmup profile run"
perf script > out.perf
if command -v flamegraph >/dev/null; then
  flamegraph out.perf > flamegraph.svg
  echo "flamegraph.svg generated"
else
  echo "FlameGraph not found; kept perf.data and out.perf"
fi
