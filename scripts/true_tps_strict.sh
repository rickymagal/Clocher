# File: scripts/true_tps_strict.sh
#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   ENGINE_BIN, DEVICE, MODEL_DIR, PROMPTS
# Optional env (with defaults):
#   THREADS, PRECISION, BATCH, PREFETCH, PRETRANSPOSE, AFFINITY, MAX_NEW
#   IE_REQUIRE_MODEL, IE_BYTES_PER_TOKEN, IE_STRIDE_BYTES, IE_VERIFY_TOUCH
#   RUNS, ROUNDS, PUSHGATEWAY_URL, REPORT_ROOT

: "${ENGINE_BIN:?set ENGINE_BIN=/abs/path/to/engine}"
: "${DEVICE:?set DEVICE=cpu|cuda}"
: "${MODEL_DIR:?set MODEL_DIR=/abs/path/to/model}"
: "${PROMPTS:?set PROMPTS=/abs/path/to/prompts.txt}"

THREADS="${THREADS:-$(nproc)}"
PRECISION="${PRECISION:-fp32}"
BATCH="${BATCH:-1}"
PREFETCH="${PREFETCH:-auto}"
PRETRANSPOSE="${PRETRANSPOSE:-all}"
AFFINITY="${AFFINITY:-auto}"
MAX_NEW="${MAX_NEW:-128}"

IE_REQUIRE_MODEL="${IE_REQUIRE_MODEL:-1}"
IE_BYTES_PER_TOKEN="${IE_BYTES_PER_TOKEN:-67108864}"
IE_STRIDE_BYTES="${IE_STRIDE_BYTES:-256}"
IE_VERIFY_TOUCH="${IE_VERIFY_TOUCH:-1}"

RUNS="${RUNS:-3}"
ROUNDS="${ROUNDS:-$RUNS}"

# Precision passthrough (accepts int4)
case "${PRECISION}" in
  fp32|bf16|int8|int4)
    CLI_PREC="${PRECISION}"
    ;;
  *)
    CLI_PREC="fp32"
    echo "WARN: unknown PRECISION '${PRECISION}', defaulting CLI to fp32"
    ;;
esac

CMD=(
  "${ENGINE_BIN}"
  --device "${DEVICE}"
  --model-dir "${MODEL_DIR}"
  --prompts-file "${PROMPTS}"
  --threads "${THREADS}"
  --precision "${CLI_PREC}"
  --batch "${BATCH}"
  --prefetch "${PREFETCH}"
  --pretranspose "${PRETRANSPOSE}"
  --affinity "${AFFINITY}"
  --max-new "${MAX_NEW}"
  --rounds "${ROUNDS}"
)

export IE_REQUIRE_MODEL IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH
export PUSHGATEWAY_URL REPORT_ROOT

exec "${CMD[@]}"
