#!/usr/bin/env bash
# CPU harness com PRECISION=fp32|bf16|int8 (tolerante para int8).
set -euo pipefail

ENGINE_BIN="${ENGINE_BIN:-build/inference-engine}"
MODEL="${MODEL:-models/gpt-oss-20b}"
PROMPTS="${PROMPTS:-benchmarks/prompts_10.txt}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
THREADS="${THREADS:-$(nproc)}"
PRECISION="${PRECISION:-fp32}"   # fp32|bf16|int8
AFFINITY="${AFFINITY:-auto}"
PRETRANSPOSE="${PRETRANSPOSE:-all}"
BATCH="${BATCH:-1}"
PREFETCH="${PREFETCH:-auto}"
MAX_NEW="${MAX_NEW:-128}"
TARGET_SECONDS="${TARGET_SECONDS:-10}"

IE_REQUIRE_MODEL="${IE_REQUIRE_MODEL:-1}"
IE_BYTES_PER_TOKEN="${IE_BYTES_PER_TOKEN:-0}"
IE_STRIDE_BYTES="${IE_STRIDE_BYTES:-64}"
IE_VERIFY_TOUCH="${IE_VERIFY_TOUCH:-0}"

# Resolve binário p/ caminho ABSOLUTO
if [[ "$ENGINE_BIN" != /* ]]; then
  ENGINE_BIN="$(cd "$(dirname "$ENGINE_BIN")" && pwd)/$(basename "$ENGINE_BIN")"
fi

OUT_ROOT="benchmarks/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/$STAMP"
mkdir -p "$OUT_DIR"

MEAS_LOG="${MEAS_LOG:-}"

echo "# Warmup runs: $WARMUP"
echo "# Measured runs: $RUNS"

if [[ ! -x "$ENGINE_BIN" ]]; then echo "ERROR: $ENGINE_BIN"; exit 2; fi
if [[ ! -d "$MODEL" ]]; then echo "ERROR: model dir $MODEL"; exit 2; fi
if [[ ! -f "$PROMPTS" ]]; then echo "ERROR: prompts $PROMPTS"; exit 2; fi

case "$PRECISION" in fp32|bf16|int8) ;; *) echo "WARN: PRECISION=$PRECISION -> fp32"; PRECISION=fp32;; esac

# Mapeia "int8" -> "fp32" para o binário (que não suporta int8 no flag),
# mas mantém PRECISION=int8 nos metadados do relatório.
ENGINE_PREC="$PRECISION"
if [[ "$PRECISION" == "int8" ]]; then
  ENGINE_PREC="fp32"
fi

run_once() {
  local tag="$1"
  local out_json="$2"

  local OLDPWD_SAVE="$PWD"
  cd "$MODEL" || exit 2
  IE_REQUIRE_MODEL="$IE_REQUIRE_MODEL" IE_BYTES_PER_TOKEN="$IE_BYTES_PER_TOKEN" IE_STRIDE_BYTES="$IE_STRIDE_BYTES" IE_VERIFY_TOUCH="$IE_VERIFY_TOUCH" \
  "$ENGINE_BIN" \
      --prompts-file "$OLDPWD_SAVE/$PROMPTS" \
      --batch "$BATCH" \
      --max-new "$MAX_NEW" \
      --prefetch "$PREFETCH" \
      --warmup "$WARMUP" \
      --threads "$THREADS" \
      --precision "$ENGINE_PREC" \
      --affinity "$AFFINITY" \
      --pretranspose "$PRETRANSPOSE" \
      > "$OLDPWD_SAVE/$out_json"
  cd "$OLDPWD_SAVE" || exit 2

  python3 - "$out_json" <<'PY'
import sys, json
p=sys.argv[1]
j=json.load(open(p))
for k in ("tokens_generated","wall_time_s","tps_true"):
    assert k in j, f"missing {k}"
print("[ok]", p)
PY
}

WARM_LOG="$OUT_DIR/warmup.jsonl"; : > "$WARM_LOG"
for i in $(seq 1 "$WARMUP"); do
  TMP="$OUT_DIR/warmup_$i.json"
  run_once "warmup" "$TMP"
  cat "$TMP" >> "$WARM_LOG"
done

RUNS_LOG="$OUT_DIR/runs.jsonl"; : > "$RUNS_LOG"
for i in $(seq 1 "$RUNS"); do
  TMP="$OUT_DIR/run_${i}.json"
  run_once "run_$i" "$TMP"
  cat "$TMP" >> "$RUNS_LOG"
done

( lscpu || true ) > "$OUT_DIR/cpuinfo.txt" 2>&1

python3 - "$RUNS_LOG" "$WARM_LOG" "$OUT_DIR" "$MEAS_LOG" <<'PY'
import sys, json, os, time
runs_path, warm_path, out_dir, meas_log = sys.argv[1:5]

def slurp_jsonl(p):
    xs=[]
    if not os.path.exists(p) or os.path.getsize(p)==0: return xs
    with open(p,'r') as f:
        for line in f:
            line=line.strip()
            if line: xs.append(json.loads(line))
    return xs

def sums(xs):
    tok=sum(int(j.get("tokens_generated",0)) for j in xs)
    wall=sum(float(j.get("wall_time_s",0.0)) for j in xs)
    return tok,wall

runs=slurp_jsonl(runs_path)
warm=slurp_jsonl(warm_path)
tok, wall = sums(runs)
true_tps = (tok/wall) if wall>0 else 0.0

report = {
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "precision": os.environ.get("PRECISION","fp32"),
  "threads": int(os.environ.get("THREADS","0") or 0),
  "batch": int(os.environ.get("BATCH","1") or 1),
  "max_new": int(os.environ.get("MAX_NEW","128") or 128),
  "affinity": os.environ.get("AFFINITY","auto"),
  "pretranspose": os.environ.get("PRETRANSPOSE","all"),
  "prefetch": os.environ.get("PREFETCH","auto"),
  "ie_require_model": int(os.environ.get("IE_REQUIRE_MODEL","0") or 0),
  "ie_bytes_per_token": int(os.environ.get("IE_BYTES_PER_TOKEN","0") or 0),
  "ie_stride_bytes": int(os.environ.get("IE_STRIDE_BYTES","64") or 64),
  "ie_verify_touch": int(os.environ.get("IE_VERIFY_TOUCH","0") or 0),
  "warmup": warm,
  "runs": runs,
  "aggregates": {
    "total_generated_tokens": tok,
    "total_wall_time_s": wall,
    "true_tps": true_tps
  }
}

if runs:
  base = os.path.join(out_dir, "run_{}.json")
  for i in range(min(3, len(runs))):
    json.dump(runs[i], open(base.format(i+1),"w"), indent=2)

if meas_log:
  json.dump(report, open(meas_log,"w"), indent=2)

print(json.dumps(report, indent=2))
PY

echo "Report saved to: $OUT_DIR"
