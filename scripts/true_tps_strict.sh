#!/usr/bin/env bash
# =============================================================================
# @file true_tps_strict.sh
# @brief Strict benchmark harness that measures true throughput and emits a single JSON object.
#
# Contract:
#   - Engine stdout must contain exactly one JSON object.
#   - Engine stderr may contain logs and is never mixed into stdout.
# =============================================================================

#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   ENGINE_BIN, DEVICE, MODEL_DIR, PROMPTS
# Optional env (with defaults):
#   THREADS, PRECISION, BATCH, PREFETCH, PRETRANSPOSE, AFFINITY, MAX_NEW
#   IE_REQUIRE_MODEL, IE_BYTES_PER_TOKEN, IE_STRIDE_BYTES, IE_VERIFY_TOUCH
#   IE_DEDUP, IE_DEDUP_STRICT, IE_DEDUP_POLICY, IE_DEDUP_CACHE_MB
#   RUNS, ROUNDS
#   VERIFY_HF_IDS, HF_DIR, HF_PY, HF_DEVICE, HF_DTYPE, HF_TOPK, HF_LOGITS_TOL, HF_CHECK_PROMPT
#
# Output:
#   JSONL to stdout:
#     - RUNS per-run JSON objects (engine JSON + injected metrics)
#     - 1 final summary JSON object

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
WARMUP_TOKENS="${WARMUP_TOKENS:-0}"

IE_REQUIRE_MODEL="${IE_REQUIRE_MODEL:-1}"
IE_BYTES_PER_TOKEN="${IE_BYTES_PER_TOKEN:-67108864}"
IE_STRIDE_BYTES="${IE_STRIDE_BYTES:-256}"
IE_VERIFY_TOUCH="${IE_VERIFY_TOUCH:-1}"

IE_DEDUP="${IE_DEDUP:-1}"
IE_DEDUP_STRICT="${IE_DEDUP_STRICT:-0}"
IE_DEDUP_POLICY="${IE_DEDUP_POLICY:-lossless}"
IE_DEDUP_CACHE_MB="${IE_DEDUP_CACHE_MB:-0}"
EXPECTED_TOKENS="${EXPECTED_TOKENS:-}"
REPORT_TOKENS_MAX="${REPORT_TOKENS_MAX:-}"

RUNS="${RUNS:-3}"
ROUNDS="${ROUNDS:-1}"
VERIFY_HF_IDS="${VERIFY_HF_IDS:-0}"
HF_DIR="${HF_DIR:-}"
HF_PY="${HF_PY:-python3}"
HF_DEVICE="${HF_DEVICE:-cpu}"
HF_DTYPE="${HF_DTYPE:-fp32}"
HF_TOPK="${HF_TOPK:-0}"
HF_LOGITS_TOL="${HF_LOGITS_TOL:-}"
HF_CHECK_PROMPT="${HF_CHECK_PROMPT:-1}"

case "${PRECISION}" in
  fp32|bf16|int8|int4) CLI_PREC="${PRECISION}" ;;
  int4w) CLI_PREC="int4" ;;
  *)
    CLI_PREC="fp32"
    echo "WARN: unknown PRECISION '${PRECISION}', defaulting CLI to fp32" >&2
    ;;
esac

export IE_REQUIRE_MODEL IE_BYTES_PER_TOKEN IE_STRIDE_BYTES IE_VERIFY_TOUCH
export IE_DEDUP IE_DEDUP_STRICT IE_DEDUP_POLICY IE_DEDUP_CACHE_MB

_tmpdir="$(mktemp -d)"
cleanup() { rm -rf "${_tmpdir}"; }
trap cleanup EXIT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

page_size_bytes="$(getconf PAGESIZE 2>/dev/null || echo 4096)"

read_vmstat_swaps() {
  awk '
    $1=="pswpin"  {in=$2}
    $1=="pswpout" {out=$2}
    END { if(in=="") in=0; if(out=="") out=0; print in, out }
  ' /proc/vmstat 2>/dev/null || echo "0 0"
}

stat_bytes_or_empty() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    stat -c '%s' "${p}" 2>/dev/null || echo ""
  else
    echo ""
  fi
}

# ---------------------------------
# CUDA guard
# ---------------------------------
CUDA_GUARD="${CUDA_GUARD:-1}"

cuda_guard_preflight() {
  if [[ "${DEVICE}" != "cuda" ]]; then
    return 0
  fi
  if [[ "${CUDA_GUARD}" == "0" ]]; then
    echo "[strict] WARN: CUDA_GUARD=0; skipping CUDA validity guard" >&2
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[strict] ERROR: DEVICE=cuda but nvidia-smi is not available." >&2
    exit 97
  fi

  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[strict] ERROR: DEVICE=cuda but nvidia-smi cannot see a GPU/driver." >&2
    exit 98
  fi

  if command -v strace >/dev/null 2>&1; then
    local trace="${_tmpdir}/cuda_preflight.trace"
    local out="${_tmpdir}/cuda_preflight.out"
    local pf_max_new=8
    local pf_rounds=1

    set +e
    strace -f -e trace=openat,ioctl -o "${trace}" \
      "${ENGINE_BIN}" \
        --device cuda \
        --model-dir "${MODEL_DIR}" \
        --prompts-file "${PROMPTS}" \
        --threads 1 \
        --precision "${CLI_PREC}" \
        --batch 1 \
        --prefetch "${PREFETCH}" \
        --pretranspose "${PRETRANSPOSE}" \
        --affinity "${AFFINITY}" \
        --max-new "${pf_max_new}" \
        --rounds "${pf_rounds}" > "${out}" 2>&1
    local rc=$?
    set -e

    if grep -Eq '(/dev/nvidia|nvidiactl|nvidia-uvm|libcuda\.so)' "${trace}"; then
      echo "[strict] CUDA guard: NVIDIA driver activity detected (OK)" >&2
      return 0
    fi

    echo "[strict] ERROR: DEVICE=cuda but preflight shows NO NVIDIA driver activity." >&2
    echo "[strict] DEBUG: trace=${trace} out=${out} rc=${rc}" >&2
    exit 99
  fi

  echo "[strict] WARN: strace not available; skipping strong CUDA guard" >&2
  return 0
}

# ---------------------------------
# CPU guard (portable): verify open(model.ie.bin) + mmap/read(fd)
# ---------------------------------
STRICT_CPU_GUARD="${STRICT_CPU_GUARD:-1}"

cpu_guard_preflight() {
  if [[ "${DEVICE}" != "cpu" ]]; then
    return 0
  fi
  if [[ "${STRICT_CPU_GUARD}" == "0" ]]; then
    echo "[strict] WARN: STRICT_CPU_GUARD=0; skipping CPU preflight" >&2
    return 0
  fi
  if [[ ! -f "${MODEL_DIR}/model.ie.bin" && -f "${MODEL_DIR}/model.q4.bin" ]]; then
    echo "[strict] WARN: model.ie.bin missing but model.q4.bin present; skipping CPU preflight" >&2
    return 0
  fi
  if ! command -v strace >/dev/null 2>&1; then
    echo "[strict] WARN: strace not available; skipping CPU preflight" >&2
    return 0
  fi

  local trace="${_tmpdir}/cpu_preflight.trace"
  local out="${_tmpdir}/cpu_preflight.out"
  local pf_prec="${CLI_PREC}"
  if [[ ! -f "${MODEL_DIR}/model.ie.bin" && -f "${MODEL_DIR}/model.q4.bin" ]]; then
    pf_prec="int4"
  fi

  set +e
  strace -f -e trace=openat,mmap,read -o "${trace}" \
    "${ENGINE_BIN}" \
      --device cpu \
      --model-dir "${MODEL_DIR}" \
      --prompts-file "${PROMPTS}" \
      --threads 1 \
      --precision "${pf_prec}" \
      --batch 1 \
      --prefetch "${PREFETCH}" \
      --pretranspose "${PRETRANSPOSE}" \
      --affinity "${AFFINITY}" \
      --max-new 1 \
      --rounds 1 > "${out}" 2>&1
  local rc=$?
  set -e

  # Extract FD(s) from: openat(... "model.ie.bin" or "model.q4.bin" ...) = <fd>
  # POSIX awk only (no match(..., ..., arr)).
  local fds
  fds="$(awk '
    /openat\(/ && (/model\.ie\.bin"/ || /model\.q4\.bin"/) {
      n=split($0, parts, "=")
      if (n < 2) next
      fd=parts[n]
      gsub(/^[[:space:]]+/, "", fd)
      gsub(/[[:space:]]+$/, "", fd)
      if (fd ~ /^[0-9]+$/) print fd
    }
  ' "${trace}" | sort -n | uniq | tr '\n' ' ')"

  if [[ -z "${fds// }" ]]; then
    echo "[strict] WARN: CPU preflight did not detect model open; continuing (trace=${trace}, out=${out}, rc=${rc})" >&2
    return 0
  fi

  local ok=0
  for fd in ${fds}; do
    if grep -Eq "mmap\\(.*,[[:space:]]*${fd}[[:space:]]*,[[:space:]]*[0-9]+" "${trace}" || \
       grep -Eq "read\\(${fd}," "${trace}"; then
      ok=1
      break
    fi
  done

  if [[ "${ok}" != "1" ]]; then
    echo "[strict] WARN: CPU preflight opened model.ie.bin/model.q4.bin but did not detect mmap/read (fds=${fds})" >&2
    echo "[strict] DEBUG: trace=${trace} out=${out} rc=${rc}" >&2
    return 0
  fi

  echo "[strict] CPU preflight: model.ie.bin mmap/read detected (OK)" >&2
  return 0
}

cuda_guard_preflight
cpu_guard_preflight
## -----------------------------------------------------------------------------
## @brief Run the engine once and extract the JSON payload.
## -----------------------------------------------------------------------------
run_one() {
  local run_idx="$1"
  local out_stdout="${_tmpdir}/run_${run_idx}.stdout.json"
  local out_stderr="${_tmpdir}/run_${run_idx}.stderr.log"
  local out_raw="${_tmpdir}/run_${run_idx}.raw.txt"
  local out_time="${_tmpdir}/run_${run_idx}.time.txt"
  local trace_path=""
  local verify_out=""

  if [[ "${VERIFY_HF_IDS}" == "1" ]]; then
    if [[ -z "${HF_DIR}" ]]; then
      echo "[strict] ERROR: VERIFY_HF_IDS=1 but HF_DIR is not set" >&2
      exit 96
    fi
    trace_path="${_tmpdir}/run_${run_idx}.trace.jsonl"
    verify_out="${_tmpdir}/run_${run_idx}.verify.json"
    verify_err="${_tmpdir}/run_${run_idx}.verify.err"
    export IE_TRACE_IDS_JSONL="${trace_path}"
  fi

  local sw_in0 sw_out0
  read -r sw_in0 sw_out0 < <(read_vmstat_swaps)

  (
    cd "${MODEL_DIR}"
    /usr/bin/time -v -o "${out_time}" \
      "${ENGINE_BIN}" \
        --device "${DEVICE}" \
        --model-dir "." \
        --prompts-file "${PROMPTS}" \
        --threads "${THREADS}" \
        --precision "${CLI_PREC}" \
        --batch "${BATCH}" \
        --prefetch "${PREFETCH}" \
        --pretranspose "${PRETRANSPOSE}" \
        --affinity "${AFFINITY}" \
        --max-new "${MAX_NEW}" \
        --warmup "${WARMUP_TOKENS}" \
        --rounds "${ROUNDS}" \
        ${EXPECTED_TOKENS:+--expected-tokens "${EXPECTED_TOKENS}"} \
        ${REPORT_TOKENS_MAX:+--report-tokens-max "${REPORT_TOKENS_MAX}"}
  ) > "${out_stdout}" 2> "${out_stderr}"

  if [[ "${VERIFY_HF_IDS}" == "1" ]]; then
    if [[ ! -s "${trace_path}" ]]; then
      cat > "${verify_out}" <<'EOF'
{"ok":false,"prompts_verified":0,"first_failure":{"reason":"trace_empty"},"per_prompt":[]}
EOF
    else
    local verify_cmd=( "${HF_PY}" "${REPO_ROOT}/tools/verify_hf_generation_ids.py" \
      --trace-jsonl "${trace_path}" \
      --hf-dir "${HF_DIR}" \
      --device "${HF_DEVICE}" \
      --dtype "${HF_DTYPE}" \
      --out "${verify_out}" )
    if [[ "${HF_TOPK}" != "0" ]]; then
      verify_cmd+=( --topk "${HF_TOPK}" )
      if [[ -n "${HF_LOGITS_TOL}" ]]; then
        verify_cmd+=( --logits-tol "${HF_LOGITS_TOL}" )
      fi
    fi
    if [[ "${HF_CHECK_PROMPT}" == "1" ]]; then
      verify_cmd+=( --check-prompt )
    fi
      set +e
      "${verify_cmd[@]}" >/dev/null 2> "${verify_err}"
      local verify_rc=$?
      set -e
      if [[ "${verify_rc}" != "0" || ! -s "${verify_out}" ]]; then
        local err_txt=""
        if [[ -s "${verify_err}" ]]; then
          err_txt="$(head -n 20 "${verify_err}" | python3 - <<'PY'
import json,sys
data=sys.stdin.read()
print(json.dumps(data))
PY
)"
        fi
        cat > "${verify_out}" <<EOF
{"ok":false,"prompts_verified":0,"first_failure":{"reason":"verify_failed","rc":${verify_rc},"stderr":${err_txt:-null}},"per_prompt":[]}
EOF
      fi
    fi
  fi

  # Keep a combined log file for quick inspection.
  cat "${out_stderr}" "${out_stdout}" > "${out_raw}" || true

  local sw_in1 sw_out1
  read -r sw_in1 sw_out1 < <(read_vmstat_swaps)

  local sw_in_pages="$((sw_in1 - sw_in0))"
  local sw_out_pages="$((sw_out1 - sw_out0))"
  if [[ "${sw_in_pages}" -lt 0 ]]; then sw_in_pages=0; fi
  if [[ "${sw_out_pages}" -lt 0 ]]; then sw_out_pages=0; fi

  local swap_in_mb swap_out_mb
  swap_in_mb="$(python3 - <<PY
pages=${sw_in_pages}
psz=${page_size_bytes}
print((pages*psz)/1_000_000.0)
PY
)"
  swap_out_mb="$(python3 - <<PY
pages=${sw_out_pages}
psz=${page_size_bytes}
print((pages*psz)/1_000_000.0)
PY
)"

  local maxrss_kb minflt majflt
  maxrss_kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' "${out_time}" 2>/dev/null || echo 0)"
  minflt="$(awk -F: '/Minor \(reclaiming a frame\) page faults/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' "${out_time}" 2>/dev/null || echo 0)"
  majflt="$(awk -F: '/Major \(requiring I\/O\) page faults/ {gsub(/^[ \t]+/,"",$2); print $2; exit}' "${out_time}" 2>/dev/null || echo 0)"

  if [[ -z "${maxrss_kb}" ]]; then maxrss_kb=0; fi
  if [[ -z "${minflt}" ]]; then minflt=0; fi
  if [[ -z "${majflt}" ]]; then majflt=0; fi

  local rss_peak_mb
  rss_peak_mb="$(python3 - <<PY
kb=${maxrss_kb}
print(kb/1024.0)
PY
)"

  python3 - "${out_stdout}" \
    "${rss_peak_mb}" "${minflt}" "${majflt}" \
    "${swap_in_mb}" "${swap_out_mb}" \
    "${MODEL_DIR}" "${ENGINE_BIN}" "${DEVICE}" \
    "${IE_DEDUP}" "${IE_DEDUP_STRICT}" "${IE_DEDUP_POLICY}" "${IE_DEDUP_CACHE_MB}" \
    "${verify_out}" <<'PY'
import sys, json

raw_path=sys.argv[1]
rss_peak=float(sys.argv[2])
minflt=int(float(sys.argv[3]))
majflt=int(float(sys.argv[4]))
swap_in=float(sys.argv[5])
swap_out=float(sys.argv[6])

model_dir=sys.argv[7]
engine_bin=sys.argv[8]
device=sys.argv[9]

ie_dedup=int(float(sys.argv[10]))
ie_dedup_strict=int(float(sys.argv[11]))
ie_dedup_policy=sys.argv[12]
ie_dedup_cache_mb=int(float(sys.argv[13]))
verify_path=sys.argv[14] if len(sys.argv) > 14 else ""

last=None
with open(raw_path,"r",encoding="utf-8",errors="ignore") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            obj=json.loads(line)
        except Exception:
            continue
        if "tokens_generated" in obj and "wall_time_s" in obj and "tps_true" in obj:
            last=obj

if last is None:
    tail=[]
    try:
        with open(raw_path,"r",encoding="utf-8",errors="ignore") as f:
            tail=f.readlines()[-200:]
    except Exception:
        tail=[]
    sys.stderr.write("ERROR: no JSON object found in engine output\n")
    sys.stderr.write("---- context ----\n")
    sys.stderr.write(f"cwd(model_dir)={model_dir}\n")
    sys.stderr.write(f"engine_bin={engine_bin}\n")
    sys.stderr.write(f"device={device}\n")
    sys.stderr.write("---- raw tail (last 200 lines) ----\n")
    sys.stderr.write("".join(tail))
    raise SystemExit(1)

last["rss_peak_mb"]=rss_peak
last["minflt"]=minflt
last["majflt"]=majflt
last["swap_in_mb"]=swap_in
last["swap_out_mb"]=swap_out

last.setdefault("pss_peak_mb", 0.0)
last.setdefault("vms_peak_mb", 0.0)
last.setdefault("rss_floor_mb", 0.0)
last.setdefault("rss_delta_mb", 0.0)

last["ie_dedup"]=ie_dedup
last["ie_dedup_strict"]=ie_dedup_strict
last["ie_dedup_policy"]=ie_dedup_policy
last["ie_dedup_cache_mb"]=ie_dedup_cache_mb

verify=None
if verify_path:
    try:
        with open(verify_path,"r",encoding="utf-8") as vf:
            verify=json.load(vf)
    except Exception:
        verify=None

per_prompt=[]
prompts=last.get("prompts", [])
ver_map={}
if isinstance(verify, dict):
    for r in verify.get("per_prompt", []) or []:
        try:
            ver_map[int(r.get("prompt_index"))]=r
        except Exception:
            continue
verify_reason=None
if isinstance(verify, dict):
    verify_reason=(verify.get("first_failure") or {}).get("reason")

if isinstance(prompts, list):
    for p in prompts:
        if not isinstance(p, dict):
            continue
        pi=p.get("prompt_index")
        rec={
            "prompt_index": pi,
            "prompt": p.get("prompt"),
            "tokens_generated": p.get("tokens_generated"),
            "window_time_s": p.get("window_time_s"),
            "prefill_time_s": p.get("prefill_time_s"),
            "decode_time_s": p.get("decode_time_s"),
            "tps_true": p.get("tps_true"),
            "tps_window": p.get("tps_window"),
            "expected_present": p.get("expected_present"),
            "expected_ok": p.get("expected_all_ok"),
        }
        if pi in ver_map:
            v=ver_map.get(pi) or {}
            ok=v.get("ok")
            prompt_ok=v.get("prompt_ok")
            if prompt_ok is False:
                ok=False
            fm=v.get("first_mismatch") or {}
            if prompt_ok is False and isinstance(v.get("prompt_mismatch"), dict):
                fm=v.get("prompt_mismatch") or fm
            step=fm.get("step")
            eng=fm.get("engine_next_id", fm.get("engine_id"))
            hf=fm.get("hf_next_id", fm.get("hf_id"))
            rec["token_id_check_ok"]=ok
            rec["token_id_check_prompt_ok"]=prompt_ok
            rec["token_id_check_first_mismatch_step"]=step
            rec["token_id_check_engine_id"]=eng
            rec["token_id_check_hf_id"]=hf
        else:
            rec["token_id_check_ok"]=False
            rec["token_id_check_prompt_ok"]=None
            rec["token_id_check_first_mismatch_step"]=None
            rec["token_id_check_engine_id"]=None
            rec["token_id_check_hf_id"]=None
            rec["token_id_check_reason"]=verify_reason or "verify_missing"
        per_prompt.append(rec)

if per_prompt:
    last["per_prompt"]=per_prompt

print(json.dumps(last, separators=(",",":")))
PY
}

for i in $(seq 1 "${RUNS}"); do
  run_one "${i}"
done

model_ie_bin_bytes="$(stat_bytes_or_empty "${MODEL_DIR}/model.ie.bin")"
dedup_defaults_bytes="$(stat_bytes_or_empty "${MODEL_DIR}/model.defaults.bin")"
dedup_masks_bytes="$(stat_bytes_or_empty "${MODEL_DIR}/model.masks.bin")"
dedup_exceptions_bytes="$(stat_bytes_or_empty "${MODEL_DIR}/model.exceptions.bin")"

dedup_total_bytes=""
if [[ -n "${dedup_defaults_bytes}" || -n "${dedup_masks_bytes}" || -n "${dedup_exceptions_bytes}" ]]; then
  dedup_total_bytes="$(python3 - <<PY
d=${dedup_defaults_bytes:-0}
m=${dedup_masks_bytes:-0}
e=${dedup_exceptions_bytes:-0}
print(int(d)+int(m)+int(e))
PY
)"
fi

python3 - <<PY
import json, os, time

def i(env, default="0"):
    s=os.environ.get(env, default)
    try:
        return int(s)
    except Exception:
        return int(default)

def s(env, default=""):
    return os.environ.get(env, default)

def opt_int_from_shell(x):
    try:
        return int(x) if x != "" else None
    except Exception:
        return None

summary = {
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "threads": i("THREADS", "0"),
  "precision": s("PRECISION", "fp32"),
  "model_dir": s("MODEL_DIR", ""),
  "prompts": s("PROMPTS", ""),
  "runs": i("RUNS", "3"),
  "rounds": i("ROUNDS", "1"),
  "device": s("DEVICE", ""),
  "engine_bin": s("ENGINE_BIN", ""),
  "batch": i("BATCH", "1"),
  "max_new": i("MAX_NEW", "128"),
  "affinity": s("AFFINITY", "auto"),
  "pretranspose": s("PRETRANSPOSE", "all"),
  "prefetch": s("PREFETCH", "auto"),
  "ie_require_model": i("IE_REQUIRE_MODEL", "1"),
  "ie_bytes_per_token": i("IE_BYTES_PER_TOKEN", "0"),
  "ie_stride_bytes": i("IE_STRIDE_BYTES", "0"),
  "ie_verify_touch": i("IE_VERIFY_TOUCH", "0"),

  "ie_dedup": i("IE_DEDUP", "0"),
  "ie_dedup_strict": i("IE_DEDUP_STRICT", "0"),
  "ie_dedup_policy": s("IE_DEDUP_POLICY", "lossless"),
  "ie_dedup_cache_mb": i("IE_DEDUP_CACHE_MB", "0"),

  "model_ie_bin_bytes": opt_int_from_shell("${model_ie_bin_bytes}"),
  "dedup_defaults_bytes": opt_int_from_shell("${dedup_defaults_bytes}"),
  "dedup_masks_bytes": opt_int_from_shell("${dedup_masks_bytes}"),
  "dedup_exceptions_bytes": opt_int_from_shell("${dedup_exceptions_bytes}"),
  "dedup_total_bytes": opt_int_from_shell("${dedup_total_bytes}"),
}
print(json.dumps(summary, separators=(",",":")))
PY
