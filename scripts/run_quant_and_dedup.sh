#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_quant_and_dedup.sh --model-dir <path> --tokenizer <path> [options]

Options:
  --model-dir <path>     Model directory (required)
  --tokenizer <path>     Tokenizer json (required)
  --threads <n>          Threads (default: 8)
  --rounds <n>           Rounds (default: 1)
  --warmup <n>           Warmup (default: 1)
  --prompt <text>        Prompt (default: Hello.)
  --max-new <n>          Max new tokens (default: 512)
  --seed <n>             Seed (default: 1)
  --precision <p>        bf16|int4 (default: int4)
  --dedup <0|1>          Enable dedup env var (default: 1)
  --strict <0|1>         Enable strict dedup env var (default: 1)
  --verify-touch <0|1>   Verify touch (default: 1)
  --bytes-per-token <n>  IE_BYTES_PER_TOKEN (default: 67108864)
  --stride-bytes <n>     IE_STRIDE_BYTES (default: 256)
  -h, --help             Show this help

Notes:
  - This script DOES NOT rebuild artifacts. Run:
      bash scripts/rebuild_dedup_and_quant.sh --model-dir <model-dir>
    first, and make sure these exist:
      <model-dir>/model.ie.compat.json
      <model-dir>/model.q4.bin
      <model-dir>/model.dedup.json
USAGE
}

die() {
  echo "error: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_DIR=""
TOKENIZER=""

THREADS="8"
ROUNDS="1"
WARMUP="1"
PROMPT="Hello."
MAX_NEW="512"
SEED="1"
PRECISION="int4"

IE_DEDUP="1"
IE_DEDUP_STRICT="1"
IE_VERIFY_TOUCH="1"
IE_BYTES_PER_TOKEN="67108864"
IE_STRIDE_BYTES="256"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --tokenizer) TOKENIZER="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --dedup) IE_DEDUP="$2"; shift 2 ;;
    --strict) IE_DEDUP_STRICT="$2"; shift 2 ;;
    --verify-touch) IE_VERIFY_TOUCH="$2"; shift 2 ;;
    --bytes-per-token) IE_BYTES_PER_TOKEN="$2"; shift 2 ;;
    --stride-bytes) IE_STRIDE_BYTES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "$MODEL_DIR" ]] || die "--model-dir is required"
[[ -n "$TOKENIZER" ]] || die "--tokenizer is required"

MODEL_DIR_ABS="$(cd "$MODEL_DIR" && pwd)"
ENGINE="$REPO_DIR/build/inference-engine"

[[ -x "$ENGINE" ]] || die "engine not found/executable: $ENGINE"

if [[ "$PRECISION" == "int4" ]]; then
  [[ -f "$MODEL_DIR_ABS/model.ie.compat.json" ]] || die "missing: $MODEL_DIR_ABS/model.ie.compat.json"
  [[ -f "$MODEL_DIR_ABS/model.q4.bin" ]] || die "missing: $MODEL_DIR_ABS/model.q4.bin"
fi

if [[ "$IE_DEDUP" == "1" ]]; then
  [[ -f "$MODEL_DIR_ABS/model.dedup.json" ]] || die "missing: $MODEL_DIR_ABS/model.dedup.json"
fi

exec env \
  IE_DEDUP="$IE_DEDUP" \
  IE_DEDUP_STRICT="$IE_DEDUP_STRICT" \
  IE_VERIFY_TOUCH="$IE_VERIFY_TOUCH" \
  IE_BYTES_PER_TOKEN="$IE_BYTES_PER_TOKEN" \
  IE_STRIDE_BYTES="$IE_STRIDE_BYTES" \
  "$ENGINE" \
    --model-dir "$MODEL_DIR_ABS" \
    --tokenizer "$TOKENIZER" \
    --precision "$PRECISION" \
    --threads "$THREADS" \
    --rounds "$ROUNDS" \
    --warmup "$WARMUP" \
    --prompt "$PROMPT" \
    --max-new "$MAX_NEW" \
    --greedy \
    --seed "$SEED"
