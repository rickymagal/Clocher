#!/usr/bin/env bash
set -euo pipefail

# -------- User params ---------------------------------------------------------
ENGINE_BIN="${ENGINE_BIN:-build/inference-engine}"
MODEL="${MODEL:-models/gpt-oss-20b}"
PROMPTS="${PROMPTS:-benchmarks/prompts_10.txt}"

RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"

THREADS="${THREADS:-$(nproc)}"
AFFINITY="${AFFINITY:-auto}"         # auto|compact|scatter
PRECISION="${PRECISION:-fp32}"       # fp32|bf16|int8|fp16 (if supported)
PRETRANSPOSE="${PRETRANSPOSE:-all}"  # none|woh|wxh|all
BATCH="${BATCH:-1}"
PREFETCH="${PREFETCH:-auto}"         # on|off|auto|N
WARMUP_TOKENS="${WARMUP_TOKENS:-64}" # used with --warmup
MAX_NEW="${MAX_NEW:-0}"              # 0 = do not pass; >0 passes --max-new N

REPORT_ROOT="${REPORT_ROOT:-benchmarks/reports}"
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-}"

# -------- Resolve absolute paths ---------------------------------------------
abspath_py() { python3 - "$1" <<'PY'
import os, sys; print(os.path.abspath(sys.argv[1]))
PY
}
ENGINE_BIN_ABS="$(abspath_py "${ENGINE_BIN}")"
MODEL_ABS="$(abspath_py "${MODEL}")"
PROMPTS_ABS="$(abspath_py "${PROMPTS}")"

# -------- Sanity checks ------------------------------------------------------
[[ -x "${ENGINE_BIN_ABS}" ]] || { echo "error: binary not found: ${ENGINE_BIN_ABS}" >&2; exit 2; }
[[ -d "${MODEL_ABS}" ]] || { echo "error: model dir not found: ${MODEL_ABS}" >&2; exit 2; }
[[ -f "${PROMPTS_ABS}" ]] || { echo "error: prompts file not found: ${PROMPTS_ABS}" >&2; exit 2; }
[[ -f "${MODEL_ABS}/model.ie.bin" && -f "${MODEL_ABS}/model.ie.json" && -f "${MODEL_ABS}/vocab.json" ]] || {
  echo "error: expected model.ie.bin, model.ie.json, vocab.json in ${MODEL_ABS}" >&2; exit 2; }

# -------- Paths --------------------------------------------------------------
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${REPORT_ROOT}/${STAMP}"
mkdir -p "${RUN_DIR}"
CPUINFO="${RUN_DIR}/cpuinfo.txt"
MEAS_LOG="${RUN_DIR}/runs.jsonl"
WARMUP_LOG="${RUN_DIR}/warmup.jsonl"

# -------- HW snapshot --------------------------------------------------------
{
  echo "timestamp_utc=${STAMP}"
  echo "threads=${THREADS}"
  echo "affinity=${AFFINITY}"
  echo "precision=${PRECISION}"
  echo "pretranspose=${PRETRANSPOSE}"
  echo "batch=${BATCH}"
  echo "prefetch=${PREFETCH}"
  echo "warmup_tokens=${WARMUP_TOKENS}"
  echo "max_new=${MAX_NEW}"
  echo "model_path=${MODEL_ABS}"
  echo
  uname -a || true
  echo
  lscpu || true
  echo
  numactl -H || true
} > "${CPUINFO}"

# -------- Runner -------------------------------------------------------------
run_engine() {
  if [[ "${MAX_NEW}" != "0" ]]; then
    ( cd "${MODEL_ABS}" && "${ENGINE_BIN_ABS}" \
        --prompts-file "${PROMPTS_ABS}" \
        --threads "${THREADS}" \
        --precision "${PRECISION}" \
        --affinity "${AFFINITY}" \
        --pretranspose "${PRETRANSPOSE}" \
        --batch "${BATCH}" \
        --prefetch "${PREFETCH}" \
        --max-new "${MAX_NEW}" )
  else
    ( cd "${MODEL_ABS}" && "${ENGINE_BIN_ABS}" \
        --prompts-file "${PROMPTS_ABS}" \
        --threads "${THREADS}" \
        --precision "${PRECISION}" \
        --affinity "${AFFINITY}" \
        --pretranspose "${PRETRANSPOSE}" \
        --batch "${BATCH}" \
        --prefetch "${PREFETCH}" )
  fi
}

# -------- Warmup -------------------------------------------------------------
if [[ "${WARMUP}" -gt 0 ]]; then
  echo "# Warmup runs: ${WARMUP}"
  for _ in $(seq 1 "${WARMUP}"); do
    if [[ "${MAX_NEW}" != "0" ]]; then
      ( cd "${MODEL_ABS}" && "${ENGINE_BIN_ABS}" \
          --prompts-file "${PROMPTS_ABS}" \
          --threads "${THREADS}" \
          --precision "${PRECISION}" \
          --affinity "${AFFINITY}" \
          --pretranspose "${PRETRANSPOSE}" \
          --batch "${BATCH}" \
          --prefetch "${PREFETCH}" \
          --warmup "${WARMUP_TOKENS}" \
          --max-new "${MAX_NEW}" ) 2>/dev/null | tee -a "${WARMUP_LOG}" >/dev/null || true
    else
      ( cd "${MODEL_ABS}" && "${ENGINE_BIN_ABS}" \
          --prompts-file "${PROMPTS_ABS}" \
          --threads "${THREADS}" \
          --precision "${PRECISION}" \
          --affinity "${AFFINITY}" \
          --pretranspose "${PRETRANSPOSE}" \
          --batch "${BATCH}" \
          --prefetch "${PREFETCH}" \
          --warmup "${WARMUP_TOKENS}" ) 2>/dev/null | tee -a "${WARMUP_LOG}" >/dev/null || true
    fi
  done
fi

# -------- Measured runs ------------------------------------------------------
echo "# Measured runs: ${RUNS}"
for i in $(seq 1 "${RUNS}"); do
  OUT="${RUN_DIR}/run_${i}.json"
  run_engine > "${OUT}"
  cat "${OUT}" >> "${MEAS_LOG}"
done

# -------- Aggregate and update PERFORMANCE.md --------------------------------
python3 - << 'PY' "${MEAS_LOG}"
import json, sys, statistics, re, datetime, os, pathlib
meas = pathlib.Path(sys.argv[1])
rows=[]
for line in meas.read_text(encoding='utf-8').splitlines():
    s=line.strip()
    if not s or s.startswith('#'): continue
    try: rows.append(json.loads(s))
    except: pass

def pick(j,k,alts=()):
    if k in j: return j[k]
    for a in alts:
        if a in j: return j[a]
    return None

tps = [pick(r,"tps_true",("tps","true_tps")) for r in rows]
tps = [x for x in tps if x is not None]
p50 = [pick(r,"latency_p50",("p50","lat_p50")) for r in rows if pick(r,"latency_p50",("p50","lat_p50")) is not None]
p95 = [pick(r,"latency_p95",("p95","lat_p95")) for r in rows if pick(r,"latency_p95",("p95","lat_p95")) is not None]
tt  = [pick(r,"total_tokens",("tokens_generated","tokens","tokens_total")) for r in rows if pick(r,"total_tokens",("tokens_generated","tokens","tokens_total")) is not None]
wl  = [pick(r,"wall_time_s",("elapsed_s","wall","walltime_s")) for r in rows if pick(r,"wall_time_s",("elapsed_s","wall","walltime_s")) is not None]

agg = {}
if tps:
    agg["TPS(true)_avg"] = statistics.fmean(tps)
    agg["TPS(true)_p95"] = statistics.quantiles(tps, n=20)[-1] if len(tps)>=20 else max(tps)
if p50: agg["Latency p50 (s)"] = statistics.fmean(p50)
if p95: agg["Latency p95 (s)"] = statistics.fmean(p95)
if tt and wl:
    agg["Tokens_total"] = sum(tt)
    agg["Wall_total (s)"] = sum(wl)
    agg["TPS(true) aggregate"] = agg["Tokens_total"]/agg["Wall_total (s)"]

ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
lines = ["# Performance Notes","",f"_Last updated: **{ts}**_","",
         "## Summary (latest run)"]
for k,v in agg.items():
    lines.append(f"- {k}: **{v:.3f}**" if isinstance(v,float) else f"- {k}: **{v}**")
lines += ["","## Run parameters",
          f"- Threads: **{os.environ.get('THREADS','')}**",
          f"- Precision: **{os.environ.get('PRECISION','')}**",
          f"- Pretranspose: **{os.environ.get('PRETRANSPOSE','')}**",
          f"- Batch: **{os.environ.get('BATCH','')}**",
          f"- Prefetch: **{os.environ.get('PREFETCH','')}**",
          f"- Warmup tokens: **{os.environ.get('WARMUP_TOKENS','')}**",
          f"- Prompts file: **{os.environ.get('PROMPTS','')}**",
          f"- Affinity policy (CLI): **{os.environ.get('AFFINITY','')}**",
          "","## Profiling Artifacts",
          "- `cpuinfo.txt`: **present**",
          "- `runs.jsonl`: **present**"]

perf_md = pathlib.Path("docs/PERFORMANCE.md")
new_block = "\n".join(lines).strip()+"\n"
if not perf_md.exists():
    perf_md.parent.mkdir(parents=True, exist_ok=True)
    perf_md.write_text(new_block, encoding="utf-8")
else:
    text = perf_md.read_text(encoding="utf-8")
    pat = r"(?s)^# Performance Notes.*?(?=^\# |\Z)"
    if re.search(pat, text, flags=re.MULTILINE):
        text = re.sub(pat, new_block, text, flags=re.MULTILINE)
    else:
        text = new_block + "\n\n" + text
    perf_md.write_text(text, encoding="utf-8")
print("Updated docs/PERFORMANCE.md")
PY

# -------- Optional pushgateway ------------------------------------------------
if [[ -n "${PUSHGATEWAY_URL}" ]]; then
  AGG_TPS=$(python3 - <<'PY' "${MEAS_LOG}"
import json,sys
vals=[]
for L in open(sys.argv[1]):
    try:
        j=json.loads(L)
    except:
        continue
    v = j.get("tps_true") or j.get("tps") or 0.0
    vals.append(v)
print(sum(vals)/len(vals) if vals else 0.0)
PY
)
  cat <<EOF | curl --data-binary @- "${PUSHGATEWAY_URL}/metrics/job/ie_bench/instance/$(hostname)"
# HELP ie_bench_tps_true TPS(true) average
# TYPE ie_bench_tps_true gauge
ie_bench_tps_true ${AGG_TPS}
EOF
fi

echo "Report saved to: ${RUN_DIR}"
