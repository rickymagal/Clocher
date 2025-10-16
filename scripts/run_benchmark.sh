#!/usr/bin/env bash
set -euo pipefail

# Location of the binary and reports folder
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/inference-engine"
REPORTS="$ROOT/benchmarks/reports"

mkdir -p "$REPORTS"

# Timestamped output directory
TS="$(date -u +%Y%m%d_%H%M%S)"
OUTDIR="$REPORTS/$TS"
mkdir -p "$OUTDIR"

# Defaults (can be overridden via env)
: "${PROMPTS_FILE:=$ROOT/benchmarks/prompts_10.txt}"
: "${MAX_NEW:=32}"
: "${THREADS:=4}"
: "${PRECISION:=fp32}"
: "${AFFINITY:=auto}"
: "${BATCH:=4}"
: "${PREFETCH:=2}"
: "${WARMUP:=16}"

SAMPLES_CSV="$OUTDIR/samples.csv"
SUMMARY_JSON="$OUTDIR/summary.json"
PARAMS_JSON="$OUTDIR/params.json"

# Header with parameter context on the first commented line
echo "# threads=$THREADS precision=$PRECISION pretranspose=none affinity=$AFFINITY" > "$SAMPLES_CSV"
echo "tokens_generated,wall_time_s,tps_true,latency_p50_ms,latency_p95_ms,cmdline" >> "$SAMPLES_CSV"

CMD="$BIN --prompts-file \"$PROMPTS_FILE\" --max-new $MAX_NEW --threads $THREADS --precision $PRECISION --affinity $AFFINITY --batch $BATCH --prefetch $PREFETCH --warmup $WARMUP"

# One run; append to CSV
OUT="$($BIN --prompts-file "$PROMPTS_FILE" --max-new "$MAX_NEW" --threads "$THREADS" --precision "$PRECISION" --affinity "$AFFINITY" --batch "$BATCH" --prefetch "$PREFETCH" --warmup "$WARMUP")"
TOKENS="$(echo "$OUT" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["tokens_generated"])')"
WALL="$(echo "$OUT" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["wall_time_s"])')"
TPS="$(echo "$OUT" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["tps_true"])')"
P50="$(echo "$OUT" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["latency_p50_ms"])')"
P95="$(echo "$OUT" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j["latency_p95_ms"])')"
echo "$TOKENS,$WALL,$TPS,$P50,$P95,$CMD" >> "$SAMPLES_CSV"

# Params captured for the doc generator
cat > "$PARAMS_JSON" <<EOF
{
  "threads": $THREADS,
  "precision": "$PRECISION",
  "pretranspose": "none",
  "affinity": "$AFFINITY",
  "batch": $BATCH,
  "prefetch": $PREFETCH,
  "warmup": $WARMUP,
  "prompts_file": "$(realpath "$PROMPTS_FILE")"
}
EOF

# Simple summary (leave room for doc script to compute richer stats)
cat > "$SUMMARY_JSON" <<EOF
{
  "tps_true": $TPS,
  "latency_p50_ms": $P50,
  "latency_p95_ms": $P95,
  "cmdline": "$CMD"
}
EOF

echo "[ok] wrote: $SAMPLES_CSV"
echo "[ok] wrote: $SUMMARY_JSON"
echo "[ok] wrote: $PARAMS_JSON"
