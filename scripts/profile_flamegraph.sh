#!/usr/bin/env bash
# Perf → stackcollapse → flamegraph, placed under benchmarks/reports/<stamp>/
set -euo pipefail

BIN="${1:-build/inference-engine}"
LABEL="${2:-profiling run}"

# Workload knobs
PROMPTS_FILE="${PROMPTS_FILE:-benchmarks/prompts_10.txt}"
BATCH="${BATCH:-16}"
WARMUP="${WARMUP:-8}"
PREFETCH_RAW="${PREFETCH:-auto}"
MAX_NEW="${MAX_NEW:-65536}"
THREADS="${THREADS:-4}"
PRECISION="${PRECISION:-fp32}"
PRETRANSPOSE="${PRETRANSPOSE:-none}"
AFFINITY="${AFFINITY:-auto}"

# Perf knobs
ROUNDS="${ROUNDS:-120}"
FREQ="${FREQ:-1200}"
CALLGRAPH="${CALLGRAPH:-fp}"   # fp|dwarf|lbr (if supported)

# Tool discovery
STACKCOLLAPSE="${STACKCOLLAPSE:-}"
FLAMEGRAPH="${FLAMEGRAPH:-}"

if [[ -z "${STACKCOLLAPSE}" ]]; then
  for c in scripts/FlameGraph/stackcollapse-perf.pl /usr/bin/stackcollapse-perf.pl /usr/bin/stackcollapse-perf; do
    [[ -x "$c" ]] && STACKCOLLAPSE="$c" && break
  done
fi
if [[ -z "${FLAMEGRAPH}" ]]; then
  for c in scripts/FlameGraph/flamegraph.pl /usr/bin/flamegraph.pl /usr/bin/flamegraph; do
    [[ -x "$c" ]] && FLAMEGRAPH="$c" && break
  done
fi

echo "[profile] generating flamegraph..."
if [[ ! -x "${STACKCOLLAPSE:-/nope}" || ! -x "${FLAMEGRAPH:-/nope}" ]]; then
  echo "error: FlameGraph utilities not found." >&2
  echo "       STACKCOLLAPSE='${STACKCOLLAPSE:-unset}' FLAMEGRAPH='${FLAMEGRAPH:-unset}'" >&2
  exit 3
fi
echo "[tools] stackcollapse: ${STACKCOLLAPSE}"
echo "[tools] flamegraph   : ${FLAMEGRAPH}"

# Normalize prefetch for new CLI
norm="$(printf '%s' "$PREFETCH_RAW" | tr '[:upper:]' '[:lower:]')"
case "$norm" in
  0|off)   PREFETCH="off" ;;
  1|on)    PREFETCH="on"  ;;
  2|auto|autotune) PREFETCH="auto" ;;
  *)       PREFETCH="auto" ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="benchmarks/reports/${STAMP}"
mkdir -p "$OUTDIR"

# Ensure frame pointers help: suggest but do not enforce
if [[ -n "${CFLAGS:-}" && "${CFLAGS}" != *"-fno-omit-frame-pointer"* ]]; then
  echo "[warn] consider compiling with -fno-omit-frame-pointer for better fp callgraphs" >&2
fi

# Compose command once
base_cmd=( "$BIN" --max-new "$MAX_NEW" --threads "$THREADS"
           --precision "$PRECISION" --affinity "$AFFINITY"
           --prefetch "$PREFETCH" --warmup "$WARMUP" )
if [[ -f "$PROMPTS_FILE" ]]; then
  base_cmd+=( --prompts-file "$PROMPTS_FILE" --batch "$BATCH" )
else
  base_cmd+=( --prompt "$LABEL" )
fi

# Record multiple iterations in a single perf session
echo "[info] perf record (rounds=${ROUNDS}, freq=${FREQ}Hz, callgraph=${CALLGRAPH})…"
perf_cmd=( perf record -F "$FREQ" --call-graph="$CALLGRAPH" -o perf.data -- )
# shellcheck disable=SC2016
driver=( bash -c 'for i in $(seq 1 '"$ROUNDS"'); do "$@" >/dev/null || exit 1; done' -- )
"${perf_cmd[@]}" "${driver[@]}" "${base_cmd[@]}"

# Convert → SVG
perf script | "$STACKCOLLAPSE" | "$FLAMEGRAPH" > flamegraph.svg || {
  echo "ERROR: stack collapse/flamegraph failed." >&2
  exit 5
}

# Stash artifacts
mv -f perf.data "$OUTDIR"/perf.data
mv -f flamegraph.svg "$OUTDIR"/flamegraph.svg

# Write params.json for docs
cat >"$OUTDIR/params.json" <<JSON
{
  "threads": ${THREADS},
  "precision": "${PRECISION}",
  "pretranspose": "${PRETRANSPOSE}",
  "affinity": "${AFFINITY}",
  "batch": ${BATCH},
  "prefetch": "${PREFETCH}",
  "warmup": ${WARMUP},
  "max_new": ${MAX_NEW}
}
JSON

echo "[ok] report dir: ${OUTDIR}"
