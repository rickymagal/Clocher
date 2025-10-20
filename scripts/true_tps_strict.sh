#!/usr/bin/env bash
# scripts/true_tps_strict.sh
# Mede TPS "verdadeiro" e verifica que o engine abriu os pesos grandes.

set -euo pipefail

# ---------------- cfg ----------------
ENGINE_BIN="${ENGINE_BIN:-build/inference-engine}"
MODEL_DIR="${MODEL_DIR:-models/gpt-oss-20b}"
PROMPTS="${PROMPTS:-benchmarks/prompts_10.txt}"
THREADS="${THREADS:-12}"
PRECISION="${PRECISION:-fp32}"
AFFINITY="${AFFINITY:-compact}"
PRETRANSPOSE="${PRETRANSPOSE:-all}"
BATCH="${BATCH:-1}"
PREFETCH="${PREFETCH:-auto}"
MAX_NEW="${MAX_NEW:-128}"
TARGET_SECONDS="${TARGET_SECONDS:-10.0}"

# ---------------- checks -------------
[[ -x "$ENGINE_BIN" ]] || { echo "[erro] ENGINE_BIN não existe: $ENGINE_BIN" >&2; exit 2; }
[[ -d "$MODEL_DIR"  ]] || { echo "[erro] MODEL_DIR não existe: $MODEL_DIR" >&2; exit 2; }
[[ -f "$PROMPTS"    ]] || { echo "[erro] PROMPTS não existe: $PROMPTS" >&2; exit 2; }

BIN_JSON="$MODEL_DIR/model.ie.json"
BIN_BIN="$MODEL_DIR/model.ie.bin"
[[ -f "$BIN_JSON" ]] || { echo "[erro] ausente: $BIN_JSON" >&2; exit 2; }
[[ -f "$BIN_BIN"  ]] || { echo "[erro] ausente: $BIN_BIN"  >&2; exit 2; }

# ---------------- step A: provar que abre os pesos ----------------
# Nota: engines sérios geralmente fazem mmap, então nem sempre veremos 'read'.
# O mínimo exigido aqui é que o processo abra ambos arquivos via openat.
STRACE_LOG="$(mktemp -t tps_strace.XXXXXX.log)"
(
  cd "$MODEL_DIR"
  # 1 corrida curtinha (max-new 1) só para observar syscalls de abertura.
  # Se não tiver strace instalado, o shell falha aqui e sinaliza o problema.
  strace -f -qq -o "$STRACE_LOG" -e trace=openat,stat \
    "../../$ENGINE_BIN" \
      --prompt "sanity" \
      --threads "$THREADS" --precision "$PRECISION" --affinity "$AFFINITY" --pretranspose "$PRETRANSPOSE" \
      --batch "$BATCH" --prefetch "$PREFETCH" --max-new 1 >/dev/null || true
)

OPEN_JSON="$(grep -E 'openat\(.*model\.ie\.json' -n "$STRACE_LOG" || true)"
OPEN_BIN="$(grep -E 'openat\(.*model\.ie\.bin'  -n "$STRACE_LOG" || true)"
rm -f "$STRACE_LOG"

if [[ -z "$OPEN_JSON" || -z "$OPEN_BIN" ]]; then
  echo "[erro] o binário NÃO abriu model.ie.json/bin (provável no-op de pesos). Use strace manual p/ conferir." >&2
  echo "[dica] rode:  cd $MODEL_DIR && strace -e trace=openat,stat -f ../../$ENGINE_BIN --prompt sanity --max-new 1 2>&1 | grep -E 'model\\.ie\\.(bin|json)'" >&2
  exit 3
fi

# ---------------- step B: janela ~10s medindo tokens/tempo ----------------
PY=$(cat <<'PYCODE'
import time, json, subprocess, os, sys
ENGINE_BIN = os.environ.get("ENGINE_BIN")
MODEL_DIR  = os.environ.get("MODEL_DIR")
PROMPTS    = os.environ.get("PROMPTS")
THREADS    = os.environ.get("THREADS")
PRECISION  = os.environ.get("PRECISION")
AFFINITY   = os.environ.get("AFFINITY")
PRETRANS   = os.environ.get("PRETRANSPOSE")
BATCH      = os.environ.get("BATCH")
PREFETCH   = os.environ.get("PREFETCH")
MAX_NEW    = os.environ.get("MAX_NEW")
TARGET_S   = float(os.environ.get("TARGET_SECONDS","10.0"))

cmd = [os.path.join("..","..",ENGINE_BIN),
       "--prompts-file", os.path.join("..","..",PROMPTS),
       "--aggregate",
       "--threads", THREADS, "--precision", PRECISION,
       "--affinity", AFFINITY, "--pretranspose", PRETRANS,
       "--batch", BATCH, "--prefetch", PREFETCH,
       "--max-new", MAX_NEW]

tok = 0
runs = 0
t0 = time.perf_counter()
while True:
    out = subprocess.run(cmd, cwd=MODEL_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    try:
        j = json.loads(out.stdout)
    except Exception as e:
        sys.stderr.write(f"[erro] JSON inválido da engine: {e}\nstdout={out.stdout}\nstderr={out.stderr}\n")
        sys.exit(4)
    tok += int(j.get("tokens_generated", 0))
    runs += 1
    if (time.perf_counter() - t0) >= TARGET_S: break

wall = (time.perf_counter() - t0)
tps  = (tok / wall) if wall > 0 else 0.0

print(json.dumps({
  "runs": runs,
  "tokens_generated": tok,
  "wall_time_s": wall,
  "tps_true": tps
}, indent=2))
PYCODE
)
python3 - <<PY
$PY
PY
