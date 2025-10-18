#!/usr/bin/env bash
# Robust per-prompt benchmark harness (no sed JSON parsing).
# Gera: benchmarks/reports/<STAMP>/{run_*.json,run_*.err,samples.csv,summary.json,params.json}

set -euo pipefail

# ---------- Config via env ----------
BENCH_PROMPTS="${BENCH_PROMPTS:-benchmarks/prompts_10.txt}"
BENCH_BATCH="${BENCH_BATCH:-16}"
BENCH_WARMUP="${BENCH_WARMUP:-4}"
BENCH_PREFETCH="${BENCH_PREFETCH:-on}"        # on|off
MAX_NEW="${MAX_NEW:-32}"
THREADS="${THREADS:-4}"
PRECISION="${PRECISION:-fp32}"                # fp32|bf16|...
PRETRANSPOSE="${PRETRANSPOSE:-none}"          # none|auto|...
AFFINITY="${AFFINITY:-auto}"                  # auto|none|pin
ROUNDS="${ROUNDS:-3}"
BIN="${BIN:-build/inference-engine}"
OUT_DIR_BASE="benchmarks/reports"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR_BASE}/${STAMP}"

mkdir -p "${OUT_DIR}"

# ---------- Sanity ----------
if ! [ -x "${BIN}" ]; then
  echo "[err] binary ${BIN} not found"; exit 2
fi
if ! [ -f "${BENCH_PROMPTS}" ]; then
  echo "[err] prompts file ${BENCH_PROMPTS} not found"; exit 2
fi

# ---------- Load prompts ----------
mapfile -t PROMPTS < <(awk 'NF{print}' "${BENCH_PROMPTS}")

# ---------- Single run ----------
call_one() {
  local idx="$1"
  local prompt="$2"
  local run_json="${OUT_DIR}/run_${idx}.json"
  local run_err="${OUT_DIR}/run_${idx}.err"

  local cmd=( "${BIN}" --prompt "${prompt}"
              --max-new "${MAX_NEW}"
              --threads "${THREADS}"
              --precision "${PRECISION}"
              --affinity "${AFFINITY}"
              --batch "${BENCH_BATCH}"
              --prefetch "${BENCH_PREFETCH}"
              --warmup "${BENCH_WARMUP}" )

  printf "[dbg] %s\n" "${cmd[*]}" > "${run_err}"
  if ! "${cmd[@]}" > "${run_json}" 2>> "${run_err}"; then
    echo "[warn] child run ${idx} failed (see ${run_err})"
  fi
}

# ---------- Execute ----------
run_idx=0
for r in $(seq 1 "${ROUNDS}"); do
  for p in "${PROMPTS[@]}"; do
    run_idx=$((run_idx + 1))
    call_one "${run_idx}" "${p}"
  done
done

# ---------- Aggregate with Python ----------
SAMPLES_CSV="${OUT_DIR}/samples.csv"
SUMMARY_JSON="${OUT_DIR}/summary.json"
PARAMS_JSON="${OUT_DIR}/params.json"

python3 - "$OUT_DIR" "$THREADS" "$PRECISION" "$PRETRANSPOSE" "$AFFINITY" <<'PY'
import sys, json, glob, statistics, math, os
out_dir, threads, precision, pretranspose, affinity = sys.argv[1:6]

runs = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
rows = []
t_tokens = 0
t_wall = 0.0
p50_vals = []
p95_vals = []

def safe_get(d, k, default):
    v = d.get(k, default)
    # normalize numbers
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except Exception:
        return default

for f in runs:
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read().strip()
            if not text:
                raise ValueError("empty")
            data = json.loads(text)
    except Exception:
        # malformed or empty -> zero row
        rows.append((0, 0.0, 0.0, 0.0, 0.0, "via-run_benchmark.sh"))
        continue

    tok = int(data.get("tokens_generated", 0) or 0)
    wall = float(data.get("wall_time_s", 0.0) or 0.0)
    tps  = float(data.get("tps_true", 0.0) or 0.0)
    p50  = float(data.get("latency_p50_ms", 0.0) or 0.0)
    p95  = float(data.get("latency_p95_ms", 0.0) or 0.0)
    cmd  = data.get("cmdline", "via-run_benchmark.sh")

    rows.append((tok, wall, tps, p50, p95, cmd))
    t_tokens += tok
    t_wall   += wall
    p50_vals.append(p50)
    p95_vals.append(p95)

# write samples.csv
with open(os.path.join(out_dir, "samples.csv"), "w", encoding="utf-8") as csv:
    csv.write(f"# threads={threads} precision={precision} pretranspose={pretranspose} affinity={affinity}\n")
    csv.write("tokens_generated,wall_time_s,tps_true,latency_p50_ms,latency_p95_ms,cmdline\n")
    for tok, wall, tps, p50, p95, cmd in rows:
        csv.write(f"{tok},{wall:.6f},{tps:.6f},{p50:.3f},{p95:.3f},{cmd}\n")

# aggregate
agg_tps = (t_tokens / t_wall) if t_wall > 0 else 0.0
def quantile(vs, q):
    if not vs:
        return 0.0
    vs = sorted(vs)
    i = int(round((len(vs)-1)*q))
    i = max(0, min(i, len(vs)-1))
    return float(vs[i])

agg_p50 = quantile(p50_vals, 0.50)
agg_p95 = quantile(p95_vals, 0.95)

with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "tokens_generated": t_tokens,
        "wall_time_s": round(t_wall, 6),
        "tps_true": float(f"{agg_tps:.6f}"),
        "latency_p50_ms": float(f"{agg_p50:.3f}"),
        "latency_p95_ms": float(f"{agg_p95:.3f}")
    }, fh)

with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as fh:
    json.dump({
        "threads": int(threads),
        "precision": precision,
        "pretranspose": pretranspose,
        "affinity": affinity
    }, fh)

print(f"[ok] wrote: {os.path.join(out_dir, 'samples.csv')}")
print(f"[ok] wrote: {os.path.join(out_dir, 'summary.json')}")
print(f"[ok] wrote: {os.path.join(out_dir, 'params.json')}")
PY

echo "${OUT_DIR}"
