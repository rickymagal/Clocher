#!/usr/bin/env bash
set -euo pipefail

# Required inputs (no autodetection, no defaults)
: "${ENGINE_BIN:?ERROR: ENGINE_BIN must be set}"
: "${MODEL_DIR:?ERROR: MODEL_DIR must be set}"
: "${PROMPTS:?ERROR: PROMPTS must be set}"

if [[ ! -x "$ENGINE_BIN" ]]; then
  echo "ERROR: ENGINE_BIN '$ENGINE_BIN' not found or not executable" >&2
  exit 2
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: MODEL_DIR '$MODEL_DIR' not found" >&2
  exit 2
fi
if [[ ! -f "$PROMPTS" ]]; then
  echo "ERROR: PROMPTS '$PROMPTS' not found" >&2
  exit 2
fi

# Only supported CLI knobs
ROUNDS="${ROUNDS:-5}"
MAX_NEW="${MAX_NEW:-128}"
THREADS="${THREADS:-$(nproc)}"
PRECISION="${PRECISION:-fp32}"              # fp32|bf16|fp16
AFFINITY="${AFFINITY:-auto}"                # auto|compact|scatter
PRETRANSPOSE="${PRETRANSPOSE:-all}"         # none|woh|wxh|all
BATCH="${BATCH:-1}"
PREFETCH="${PREFETCH:-auto}"                # on|off|auto|N
WARMUP="${WARMUP:-1}"                       # warmup iterations per run

# Enforce real-model mode & strict memory touch to avoid fake fast runs
export IE_REQUIRE_MODEL="${IE_REQUIRE_MODEL:-1}"
export IE_VERIFY_TOUCH="${IE_VERIFY_TOUCH:-1}"
export IE_BYTES_PER_TOKEN="${IE_BYTES_PER_TOKEN:-67108864}"  # 64 MiB/token
export IE_STRIDE_BYTES="${IE_STRIDE_BYTES:-256}"

ABS_ENGINE="$(realpath -m "$ENGINE_BIN")"
ABS_PROMPTS="$(realpath -m "$PROMPTS")"
ABS_MODEL_DIR="$(realpath -m "$MODEL_DIR")"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

sum_tokens=0
sum_wall_ns=0
lat_p50_accum=0
lat_p50_n=0
lat_p95_accum=0
lat_p95_n=0
rss_peak_max_mb=0
rss_peak_sum_mb=0
rss_peak_n=0
kv_hits_total=0
kv_miss_total=0

have_time=1
command -v /usr/bin/time >/dev/null 2>&1 || have_time=0

for i in $(seq 1 "$ROUNDS"); do
  out_file="$tmpdir/run_${i}.json"
  tim_file="$tmpdir/time_${i}.txt"

  start_ns="$(date +%s%N)"
  if [[ "$have_time" -eq 1 ]]; then
    # /usr/bin/time -v writes to stderr; capture it for RSS
    {
      cd "$ABS_MODEL_DIR"
      /usr/bin/time -v "$ABS_ENGINE" \
        --prompts-file "$ABS_PROMPTS" \
        --max-new "$MAX_NEW" \
        --threads "$THREADS" \
        --precision "$PRECISION" \
        --affinity "$AFFINITY" \
        --pretranspose "$PRETRANSPOSE" \
        --batch "$BATCH" \
        --prefetch "$PREFETCH" \
        --warmup "$WARMUP" \
        --aggregate
    } >"$out_file" 2>"$tim_file"
  else
    {
      cd "$ABS_MODEL_DIR"
      "$ABS_ENGINE" \
        --prompts-file "$ABS_PROMPTS" \
        --max-new "$MAX_NEW" \
        --threads "$THREADS" \
        --precision "$PRECISION" \
        --affinity "$AFFINITY" \
        --pretranspose "$PRETRANSPOSE" \
        --batch "$BATCH" \
        --prefetch "$PREFETCH" \
        --warmup "$WARMUP" \
        --aggregate
    } >"$out_file"
    : >"$tim_file"
  fi
  end_ns="$(date +%s%N)"

  # Parse engine JSON for tokens and optional latencies/KV stats
  out="$(cat "$out_file")"

  tokens="$(printf '%s\n' "$out" | sed -n 's/.*"tokens_generated"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
  : "${tokens:=0}"

  # Optional fields (treat as 0 if absent)
  lp50="$(printf '%s\n' "$out" | sed -n 's/.*"latency_p50_ms"[[:space:]]*:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' | head -n1)"
  lp95="$(printf '%s\n' "$out" | sed -n 's/.*"latency_p95_ms"[[:space:]]*:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' | head -n1)"
  khit="$(printf '%s\n' "$out" | sed -n 's/.*"kv_hits"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
  kmis="$(printf '%s\n' "$out" | sed -n 's/.*"kv_misses"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -n1)"
  : "${lp50:=0}"
  : "${lp95:=0}"
  : "${khit:=0}"
  : "${kmis:=0}"

  # RSS peak from /usr/bin/time -v (in KB); convert to MB (integer)
  rss_kb="$(sed -n 's/^.*Maximum resident set size (kbytes):[[:space:]]*\([0-9]\+\).*$/\1/p' "$tim_file" | tail -n1 || true)"
  if [[ -z "${rss_kb:-}" ]]; then rss_kb=0; fi
  rss_mb="$(( (rss_kb + 1023) / 1024 ))"

  wall_ns="$((end_ns - start_ns))"
  wall_s="$(awk -v ns="$wall_ns" 'BEGIN{printf "%.9f", ns/1e9}')"
  tps_run="$(awk -v t="$tokens" -v w="$wall_s" 'BEGIN{ if (w==0) print 0; else printf "%.6f", t / w }')"

  # Emit a per-run JSON line that your updater will ingest
  cat <<RUNJSON
{"tokens_generated": $tokens, "wall_time_s": $wall_s, "tps_true": $tps_run, "latency_p50_ms": $lp50, "latency_p95_ms": $lp95, "rss_peak_mb": $rss_mb, "kv_hits": $khit, "kv_misses": $kmis}
RUNJSON

  # Accumulate
  sum_tokens=$((sum_tokens + tokens))
  sum_wall_ns=$((sum_wall_ns + wall_ns))

  if [[ "$lp50" != "0" ]]; then lat_p50_accum="$(awk -v a="$lat_p50_accum" -v b="$lp50" 'BEGIN{printf "%.9f", a+b}')"; lat_p50_n=$((lat_p50_n+1)); fi
  if [[ "$lp95" != "0" ]]; then lat_p95_accum="$(awk -v a="$lat_p95_accum" -v b="$lp95" 'BEGIN{printf "%.9f", a+b}')"; lat_p95_n=$((lat_p95_n+1)); fi

  if (( rss_mb > 0 )); then
    rss_peak_sum_mb=$((rss_peak_sum_mb + rss_mb))
    rss_peak_n=$((rss_peak_n + 1))
    if (( rss_mb > rss_peak_max_mb )); then rss_peak_max_mb="$rss_mb"; fi
  fi

  kv_hits_total=$((kv_hits_total + khit))
  kv_miss_total=$((kv_miss_total + kmis))
done

sum_wall_s="$(awk -v ns="$sum_wall_ns" 'BEGIN{printf "%.9f", ns/1e9}')"
tps_true_overall="$(awk -v tok="$sum_tokens" -v w="$sum_wall_s" 'BEGIN{ if (w==0) print 0; else printf "%.6f", tok / w }')"
lat_p50_mean="$(awk -v s="$lat_p50_accum" -v n="$lat_p50_n" 'BEGIN{ if (n==0) print 0; else printf "%.6f", s/n }')"
lat_p95_mean="$(awk -v s="$lat_p95_accum" -v n="$lat_p95_n" 'BEGIN{ if (n==0) print 0; else printf "%.6f", s/n }')"
rss_peak_mean_mb=$(( rss_peak_n>0 ? (rss_peak_sum_mb / rss_peak_n) : 0 ))

# Final summary object (your updater will also accept this)
cat <<JSON
{
  "runs": $ROUNDS,
  "tokens_generated": $sum_tokens,
  "wall_time_s": $sum_wall_s,
  "tps_true": $tps_true_overall,
  "latency_p50_ms": $lat_p50_mean,
  "latency_p95_ms": $lat_p95_mean,
  "rss_peak_mb": $rss_peak_mean_mb,
  "rss_peak_mb_max": $rss_peak_max_mb,
  "kv_hits": $kv_hits_total,
  "kv_misses": $kv_miss_total,
  "threads": $THREADS,
  "precision": "$PRECISION",
  "affinity": "$AFFINITY",
  "pretranspose": "$PRETRANSPOSE",
  "batch": $BATCH,
  "prefetch": "$PREFETCH",
  "max_new": $MAX_NEW,
  "model_dir": "$ABS_MODEL_DIR",
  "prompts": "$ABS_PROMPTS",
  "ie_require_model": $IE_REQUIRE_MODEL,
  "ie_verify_touch": $IE_VERIFY_TOUCH,
  "ie_bytes_per_token": $IE_BYTES_PER_TOKEN,
  "ie_stride_bytes": $IE_STRIDE_BYTES
}
JSON
