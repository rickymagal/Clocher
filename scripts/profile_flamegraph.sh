#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# @file profile_flamegraph.sh
# @brief Run Linux perf and generate an SVG flamegraph for the given binary.
#
# Usage:
#   scripts/profile_flamegraph.sh build/inference-engine "your prompt"
#
# Requirements:
#   - Linux 'perf' (perf_event) installed and permitted
#   - FlameGraph scripts available (we auto-download to .dist/ if missing)
#
# Output:
#   - perf.data
#   - flamegraph.svg
# -----------------------------------------------------------------------------
set -euo pipefail

BIN=${1:-build/inference-engine}
PROMPT=${2:-"profile run"}

if [[ ! -x "$BIN" ]]; then
  echo "[error] binary not found or not executable: $BIN"
  exit 1
fi

# Ensure helper dir
DIST=".dist"
mkdir -p "$DIST"

# Fetch FlameGraph if missing
if [[ ! -d "$DIST/FlameGraph" ]]; then
  echo "[info] fetching FlameGraph to $DIST/FlameGraph"
  git clone --depth=1 https://github.com/brendangregg/FlameGraph.git "$DIST/FlameGraph" >/dev/null 2>&1 || {
    echo "[error] failed to fetch FlameGraph (no git or offline)"; exit 1; }
fi

# Clean old outputs
rm -f perf.data out.perf script.stacks flamegraph.svg

# Record with perf (user-space only)
echo "[info] perf record..."
perf record -F 999 -g -- "$BIN" "$PROMPT" >/dev/null

# Script to folded stacks
echo "[info] perf script..."
perf script > out.perf

# Collapse and render
echo "[info] generate flamegraph.svg"
"$DIST/FlameGraph/stackcollapse-perf.pl" out.perf > script.stacks
"$DIST/FlameGraph/flamegraph.pl" script.stacks > flamegraph.svg

echo "[ok] flamegraph.svg generated"
