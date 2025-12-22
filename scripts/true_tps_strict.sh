#!/usr/bin/env bash
set -euo pipefail

# Required env:
#   ENGINE_BIN, DEVICE, MODEL_DIR, PROMPTS
# Optional env (with defaults):
#   THREADS, PRECISION, BATCH, PREFETCH, PRETRANSPOSE, AFFINITY, MAX_NEW
#   IE_REQUIRE_MODEL, IE_BYTES_PER_TOKEN, IE_STRIDE_BYTES, IE_VERIFY_TOUCH
#   IE_DEDUP, IE_DEDUP_STRICT, IE_DEDUP_POLICY, IE_DEDUP_CACHE_MB
#   RUNS, ROUNDS
#
# Output:
#   JSONL to stdout:
#     - RUNS per-run JSON objects (engine JSON + injected memory metrics)
#     - 1 final summary JSON object (detected by update_performance_md.py)

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

# Dedup (optional; default disabled)
IE_DEDUP="${IE_DEDUP:-0}"
IE_DEDUP_STRICT="${IE_DEDUP_STRICT:-0}"
IE_DEDUP_POLICY="${IE_DEDUP_POLICY:-lossless}"
IE_DEDUP_CACHE_MB="${IE_DEDUP_CACHE_MB:-0}"

RUNS="${RUNS:-3}"
ROUNDS="${ROUNDS:-1}"

# Precision passthrough (accepts int4; also tolerate int4w alias)
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

page_size_bytes="$(getconf PAGESIZE 2>/dev/null || echo 4096)"

read_memavailable_kb() {
  awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo ""
}

read_vmstat_swaps() {
  # returns: "pswpin pswpout" (pages)
  awk '
    $1=="pswpin"  {in=$2}
    $1=="pswpout" {out=$2}
    END { if(in=="") in=0; if(out=="") out=0; print in, out }
  ' /proc/vmstat 2>/dev/null || echo "0 0"
}

read_psi_mem_some_avg10() {
  awk '
    $1=="some" {
      for(i=1;i<=NF;i++){
        if($i ~ /^avg10=/){
          split($i,a,"=");
          print a[2];
          exit
        }
      }
    }
  ' /proc/pressure/memory 2>/dev/null || echo ""
}

read_psi_mem_full_avg10() {
  awk '
    $1=="full" {
      for(i=1;i<=NF;i++){
        if($i ~ /^avg10=/){
          split($i,a,"=");
          print a[2];
          exit
        }
      }
    }
  ' /proc/pressure/memory 2>/dev/null || echo ""
}

read_proc_status_kb() {
  # $1 = pid, prints: "VmRSS_kb VmSize_kb VmSwap_kb" (0 if missing)
  local pid="$1"
  awk '
    $1=="VmRSS:"  {rss=$2}
    $1=="VmSize:" {vms=$2}
    $1=="VmSwap:" {swp=$2}
    END {
      if(rss=="") rss=0;
      if(vms=="") vms=0;
      if(swp=="") swp=0;
      print rss, vms, swp
    }
  ' "/proc/${pid}/status" 2>/dev/null || echo "0 0 0"
}

read_proc_pss_kb() {
  local pid="$1"
  if [[ -r "/proc/${pid}/smaps_rollup" ]]; then
    awk '$1=="Pss:" {print $2; exit}' "/proc/${pid}/smaps_rollup" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

read_proc_faults() {
  local pid="$1"
  python3 - "$pid" <<'PY'
import sys
pid=sys.argv[1]
try:
    s=open(f"/proc/{pid}/stat","r").read()
    rparen=s.rfind(")")
    tail=s[rparen+2:].split()
    minflt=int(tail[7])
    majflt=int(tail[9])
    print(minflt, majflt)
except Exception:
    print(0, 0)
PY
}

stat_bytes_or_empty() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    stat -c '%s' "${p}" 2>/dev/null || echo ""
  else
    echo ""
  fi
}

run_one() {
  local run_idx="$1"
  local out_raw="${_tmpdir}/run_${run_idx}.raw.txt"

  local sw_in0 sw_out0
  read -r sw_in0 sw_out0 < <(read_vmstat_swaps)

  (
    cd "${MODEL_DIR}"
    exec "${ENGINE_BIN}" \
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
      --rounds "${ROUNDS}"
  ) > "${out_raw}" 2>&1 &
  local pid="$!"

  local rss_floor_kb=0
  local rss_peak_kb=0
  local vms_peak_kb=0
  local pss_peak_kb=0

  local memavail_sum_kb=0
  local memavail_n=0

  local psi_some_sum=0
  local psi_some_n=0
  local psi_full_sum=0
  local psi_full_n=0

  local minflt0 majflt0
  read -r minflt0 majflt0 < <(read_proc_faults "${pid}")

  while kill -0 "${pid}" 2>/dev/null; do
    local rss_kb vms_kb swp_kb
    read -r rss_kb vms_kb swp_kb < <(read_proc_status_kb "${pid}")

    if [[ "${rss_floor_kb}" -eq 0 && "${rss_kb}" -gt 0 ]]; then
      rss_floor_kb="${rss_kb}"
    fi

    if [[ "${rss_kb}" -gt "${rss_peak_kb}" ]]; then rss_peak_kb="${rss_kb}"; fi
    if [[ "${vms_kb}" -gt "${vms_peak_kb}" ]]; then vms_peak_kb="${vms_kb}"; fi

    local pss_kb
    pss_kb="$(read_proc_pss_kb "${pid}")"
    if [[ "${pss_kb}" -gt "${pss_peak_kb}" ]]; then pss_peak_kb="${pss_kb}"; fi

    local memavail_kb
    memavail_kb="$(read_memavailable_kb)"
    if [[ -n "${memavail_kb}" ]]; then
      memavail_sum_kb="$((memavail_sum_kb + memavail_kb))"
      memavail_n="$((memavail_n + 1))"
    fi

    local psi_some psi_full
    psi_some="$(read_psi_mem_some_avg10)"
    psi_full="$(read_psi_mem_full_avg10)"

    if [[ -n "${psi_some}" ]]; then
      psi_some_sum="$(python3 - <<PY
s=${psi_some_sum}
x=${psi_some}
print(s + x)
PY
)"
      psi_some_n="$((psi_some_n + 1))"
    fi

    if [[ -n "${psi_full}" ]]; then
      psi_full_sum="$(python3 - <<PY
s=${psi_full_sum}
x=${psi_full}
print(s + x)
PY
)"
      psi_full_n="$((psi_full_n + 1))"
    fi

    sleep 0.05
  done

  wait "${pid}" || true

  local minflt1 majflt1
  read -r minflt1 majflt1 < <(read_proc_faults "${pid}")
  local minflt_delta="$((minflt1 - minflt0))"
  local majflt_delta="$((majflt1 - majflt0))"
  if [[ "${minflt_delta}" -lt 0 ]]; then minflt_delta=0; fi
  if [[ "${majflt_delta}" -lt 0 ]]; then majflt_delta=0; fi

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

  local memavail_mean_mb="null"
  local memavail_mean_pct="null"
  if [[ "${memavail_n}" -gt 0 ]]; then
    local memavail_mean_kb="$((memavail_sum_kb / memavail_n))"
    memavail_mean_mb="$(python3 - <<PY
kb=${memavail_mean_kb}
print(kb/1024.0)
PY
)"
    local memtotal_kb
    memtotal_kb="$(awk '/MemTotal:/ {print $2; exit}' /proc/meminfo 2>/dev/null || echo 0)"
    if [[ "${memtotal_kb}" -gt 0 ]]; then
      memavail_mean_pct="$(python3 - <<PY
a=${memavail_mean_kb}
t=${memtotal_kb}
print((a/t)*100.0)
PY
)"
    fi
  fi

  local psi_some_mean="null"
  local psi_full_mean="null"
  if [[ "${psi_some_n}" -gt 0 ]]; then
    psi_some_mean="$(python3 - <<PY
s=${psi_some_sum}
n=${psi_some_n}
print(s/n)
PY
)"
  fi
  if [[ "${psi_full_n}" -gt 0 ]]; then
    psi_full_mean="$(python3 - <<PY
s=${psi_full_sum}
n=${psi_full_n}
print(s/n)
PY
)"
  fi

  local rss_floor_mb rss_peak_mb vms_peak_mb pss_peak_mb rss_delta_mb
  rss_floor_mb="$(python3 - <<PY
kb=${rss_floor_kb}
print(kb/1024.0)
PY
)"
  rss_peak_mb="$(python3 - <<PY
kb=${rss_peak_kb}
print(kb/1024.0)
PY
)"
  vms_peak_mb="$(python3 - <<PY
kb=${vms_peak_kb}
print(kb/1024.0)
PY
)"
  pss_peak_mb="$(python3 - <<PY
kb=${pss_peak_kb}
print(kb/1024.0)
PY
)"
  rss_delta_mb="$(python3 - <<PY
a=${rss_peak_kb}
b=${rss_floor_kb}
print((a-b)/1024.0 if a>=b else 0.0)
PY
)"

  python3 - "${out_raw}" \
    "${rss_floor_mb}" "${rss_delta_mb}" "${pss_peak_mb}" "${vms_peak_mb}" \
    "${minflt_delta}" "${majflt_delta}" \
    "${swap_in_mb}" "${swap_out_mb}" \
    "${psi_some_mean}" "${psi_full_mean}" \
    "${memavail_mean_mb}" "${memavail_mean_pct}" \
    "${MODEL_DIR}" "${ENGINE_BIN}" "${DEVICE}" \
    "${IE_DEDUP}" "${IE_DEDUP_STRICT}" "${IE_DEDUP_POLICY}" "${IE_DEDUP_CACHE_MB}" <<'PY'
import sys, json

raw_path=sys.argv[1]
rss_floor=float(sys.argv[2])
rss_delta=float(sys.argv[3])
pss_peak=float(sys.argv[4])
vms_peak=float(sys.argv[5])
minflt=int(float(sys.argv[6]))
majflt=int(float(sys.argv[7]))
swap_in=float(sys.argv[8])
swap_out=float(sys.argv[9])

def to_opt_float(s):
    if s == "null":
        return None
    return float(s)

psi_some=to_opt_float(sys.argv[10])
psi_full=to_opt_float(sys.argv[11])
mem_av_mb=to_opt_float(sys.argv[12])
mem_av_pct=to_opt_float(sys.argv[13])

model_dir=sys.argv[14]
engine_bin=sys.argv[15]
device=sys.argv[16]

ie_dedup=int(float(sys.argv[17]))
ie_dedup_strict=int(float(sys.argv[18]))
ie_dedup_policy=sys.argv[19]
ie_dedup_cache_mb=int(float(sys.argv[20]))

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

# Inject extended memory metrics
last["pss_peak_mb"]=pss_peak
last["vms_peak_mb"]=vms_peak
last["rss_floor_mb"]=rss_floor
last["rss_delta_mb"]=rss_delta
last["minflt"]=minflt
last["majflt"]=majflt
last["swap_in_mb"]=swap_in
last["swap_out_mb"]=swap_out
if psi_some is not None:
    last["psi_mem_some_pct"]=psi_some
if psi_full is not None:
    last["psi_mem_full_pct"]=psi_full
if mem_av_mb is not None:
    last["mem_available_mb"]=mem_av_mb
if mem_av_pct is not None:
    last["mem_available_pct"]=mem_av_pct

# Stamp dedup config into each run (helps downstream reports and comparisons)
last["ie_dedup"]=ie_dedup
last["ie_dedup_strict"]=ie_dedup_strict
last["ie_dedup_policy"]=ie_dedup_policy
last["ie_dedup_cache_mb"]=ie_dedup_cache_mb

print(json.dumps(last, separators=(",",":")))
PY
}

for i in $(seq 1 "${RUNS}"); do
  run_one "${i}"
done

# Dedup artifact sizes (real, from filesystem)
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

  # Dedup config
  "ie_dedup": i("IE_DEDUP", "0"),
  "ie_dedup_strict": i("IE_DEDUP_STRICT", "0"),
  "ie_dedup_policy": s("IE_DEDUP_POLICY", "lossless"),
  "ie_dedup_cache_mb": i("IE_DEDUP_CACHE_MB", "0"),

  # Dedup artifacts (bytes)
  "model_ie_bin_bytes": opt_int_from_shell("${model_ie_bin_bytes}"),
  "dedup_defaults_bytes": opt_int_from_shell("${dedup_defaults_bytes}"),
  "dedup_masks_bytes": opt_int_from_shell("${dedup_masks_bytes}"),
  "dedup_exceptions_bytes": opt_int_from_shell("${dedup_exceptions_bytes}"),
  "dedup_total_bytes": opt_int_from_shell("${dedup_total_bytes}"),
}
print(json.dumps(summary, separators=(",",":")))
PY
