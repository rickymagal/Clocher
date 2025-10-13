#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/inference-engine"
[[ -x "$BIN" ]] || { echo "Binary missing. Run: make build"; exit 1; }
python3 "$ROOT/benchmarks/harness.py"
