#!/usr/bin/env bash
# scripts/profile_flamegraph.sh — minimal, env-first FlameGraph driver
set -euo pipefail

BIN="${1:-build/inference-engine}"
PROMPT="${2:-profiling prompt with 64+ tokens}"

# ---- required external tools (env can point to non-exec .pl; we wrap with perl) ----
SC_PATH="${STACKCOLLAPSE:-}"
FG_PATH="${FLAMEGRAPH:-}"

if [[ -z "$SC_PATH" || ! -e "$SC_PATH" ]]; then
  # try PATH fallbacks
  SC_PATH="$(command -v stackcollapse-perf.pl || command -v stackcollapse-perf || true)"
fi
if [[ -z "$FG_PATH" || ! -e "$FG_PATH" ]]; then
  FG_PATH="$(command -v flamegraph.pl || command -v flamegraph || true)"
fi

[[ -n "$SC_PATH" && -e "$SC_PATH" ]] || { echo "error: FlameGraph stackcollapse utility not found."; exit 3; }
[[ -n "$FG_PATH" && -e "$FG_PATH" ]] || { echo "error: FlameGraph renderer not found."; exit 3; }

SC_CMD=( perl "$SC_PATH" ); [[ -x "$SC_PATH" ]] && SC_CMD=( "$SC_PATH" )
FG_CMD=( perl "$FG_PATH" ); [[ -x "$FG_PATH" ]] && FG_CMD=( "$FG_PATH" )

# ---- knobs (env overridable) ----
ROUNDS="${ROUNDS:-60}"
MAX_NEW="${MAX_NEW:-65536}"
FREQ="${FREQ:-1200}"
CALLGRAPH="${CALLGRAPH:-fp}"   # fp|dwarf

# Optional CLI hints via env
ENGINE_CLI=()
[[ -n "${THREADS:-}"      ]] && ENGINE_CLI+=( --threads "$THREADS" )
[[ -n "${PRECISION:-}"    ]] && ENGINE_CLI+=( --precision "$PRECISION" )
[[ -n "${PRETRANSPOSE:-}" ]] && ENGINE_CLI+=( --pretranspose "$PRETRANSPOSE" )
[[ -n "${AFFINITY:-}"     ]] && ENGINE_CLI+=( --affinity "$AFFINITY" )
[[ -n "${PREFETCH:-}"     ]] && ENGINE_CLI+=( --prefetch "$PREFETCH" )
[[ -n "${WARMUP:-}"       ]] && ENGINE_CLI+=( --warmup "$WARMUP" )
[[ -n "${BATCH:-}"        ]] && ENGINE_CLI+=( --batch "$BATCH" )

if [[ -n "${PROMPTS_FILE:-}" && -f "${PROMPTS_FILE}" ]]; then
  ENGINE_CLI+=( --prompts-file "$PROMPTS_FILE" )
else
  ENGINE_CLI+=( --prompt "$PROMPT" )
fi
ENGINE_CLI+=( --max-new "$MAX_NEW" )

[[ -x "$BIN" ]] || { echo "error: $BIN not executable"; exit 2; }
command -v perf >/dev/null 2>&1 || { echo "error: perf not found"; exit 2; }

echo "[profile] generating flamegraph..."
echo "[tools] stackcollapse: ${SC_CMD[*]}"
echo "[tools] flamegraph   : ${FG_CMD[*]}"
rm -f perf.data script.stacks flamegraph.svg

record_once () {
  perf record --append -o perf.data -F "$FREQ" --call-graph "$CALLGRAPH" -- \
    "$BIN" "${ENGINE_CLI[@]}" >/dev/null 2>&1 || true
}

for ((i=1;i<=ROUNDS;i++)); do record_once; done

if [[ ! -s perf.data && "$CALLGRAPH" == "dwarf" ]]; then
  echo "[fallback] empty with dwarf; retrying fp..."
  CALLGRAPH="fp"
  for ((i=1;i<=ROUNDS;i++)); do record_once; done
fi

if [[ ! -s perf.data ]]; then
  echo "[fallback] escalating to sudo perf…"
  for ((i=1;i<=ROUNDS;i++)); do
    sudo perf record --append -o perf.data -F "$FREQ" --call-graph "$CALLGRAPH" -- \
      "$BIN" "${ENGINE_CLI[@]}" >/dev/null 2>&1 || true
  done
fi

[[ -s perf.data ]] || { echo "ERROR: perf.data is empty"; exit 5; }

perf script | "${SC_CMD[@]}" > script.stacks
"${FG_CMD[@]}" script.stacks > flamegraph.svg
echo "[ok] flamegraph.svg"
