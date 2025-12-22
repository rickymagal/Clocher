#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/gpt-oss-20b}"
HF_DIR="${HF_DIR:-$MODEL_DIR/hf}"
OUT_DIR="${OUT_DIR:-$MODEL_DIR}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing '$1'"; exit 2; }; }

need python3
need rg
need find

IE_JSON="$OUT_DIR/model.ie.json"
IE_BIN="$OUT_DIR/model.ie.bin"

if [[ ! -f "$IE_JSON" || ! -f "$IE_BIN" ]]; then
  echo "ERROR: missing $IE_JSON / $IE_BIN (run hf_to_iebin_raw.py first)"
  exit 2
fi

echo "[ok] IEBIN present:"
ls -lh "$IE_JSON" "$IE_BIN"

# ---- locate manifest.dedup.json (Downloads -> model dir) ---------------------
MANIFEST_CANDIDATES=(
  "$OUT_DIR/manifest.dedup.json"
  "$OUT_DIR/dedup_manifest.json"
  "$HOME/Downloads/manifest.dedup.json"
  "$HOME/Downloads/dedup_manifest.json"
)

MANIFEST=""
for p in "${MANIFEST_CANDIDATES[@]}"; do
  if [[ -f "$p" ]]; then MANIFEST="$p"; break; fi
done

if [[ -z "$MANIFEST" ]]; then
  echo "ERROR: could not find manifest.dedup.json. Put it in:"
  echo "  $OUT_DIR/manifest.dedup.json"
  exit 2
fi

if [[ "$MANIFEST" != "$OUT_DIR/manifest.dedup.json" ]]; then
  echo "[copy] manifest -> $OUT_DIR/manifest.dedup.json"
  cp -f "$MANIFEST" "$OUT_DIR/manifest.dedup.json"
  MANIFEST="$OUT_DIR/manifest.dedup.json"
fi

echo "[ok] manifest: $MANIFEST"
ls -lh "$MANIFEST"

# ---- find and run dedup artifact generator ----------------------------------
echo "[dedup] searching for generator..."
DEDUP_SCRIPT="$(find scripts -maxdepth 2 -type f -name '*.py' -print | rg -n '(dedup).*\.py$' | head -n 1 | cut -d: -f1 || true)"

if [[ -z "$DEDUP_SCRIPT" ]]; then
  echo "ERROR: no obvious dedup python script found under scripts/."
  echo "Try: ls scripts | rg -n 'dedup'"
  exit 2
fi

echo "[dedup] candidate: $DEDUP_SCRIPT"
echo "[dedup] attempting common invocations..."

set +e
python3 "$DEDUP_SCRIPT" --model-dir "$OUT_DIR" --manifest "$MANIFEST"
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  python3 "$DEDUP_SCRIPT" --model "$OUT_DIR" --manifest "$MANIFEST"
  STATUS=$?
fi
if [[ $STATUS -ne 0 ]]; then
  python3 "$DEDUP_SCRIPT" --model-dir "$OUT_DIR"
  STATUS=$?
fi
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "ERROR: dedup generator failed. Show its usage:"
  python3 "$DEDUP_SCRIPT" --help || true
  exit 2
fi

echo "[dedup] done."

# ---- find and run offline INT4 artifact generator ---------------------------
echo "[int4] searching for PTQ/export script..."
PTQ_SCRIPT="$(find scripts -maxdepth 2 -type f -name '*.py' -print | rg -n '(ptq|int4|quant).*\.py$' | head -n 1 | cut -d: -f1 || true)"

if [[ -z "$PTQ_SCRIPT" ]]; then
  echo "ERROR: no obvious quant/PTQ script found under scripts/."
  echo "Try: ls scripts | rg -n '(ptq|int4|quant)'"
  exit 2
fi

echo "[int4] candidate: $PTQ_SCRIPT"
echo "[int4] attempting common invocations..."

set +e
python3 "$PTQ_SCRIPT" --model-dir "$OUT_DIR" --precision int4
STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  python3 "$PTQ_SCRIPT" --model "$OUT_DIR" --precision int4
  STATUS=$?
fi
if [[ $STATUS -ne 0 ]]; then
  python3 "$PTQ_SCRIPT" --model-dir "$OUT_DIR" --int4
  STATUS=$?
fi
set -e

if [[ $STATUS -ne 0 ]]; then
  echo "ERROR: PTQ/int4 generator failed. Show its usage:"
  python3 "$PTQ_SCRIPT" --help || true
  exit 2
fi

echo "[int4] done."

echo "[ok] listing likely artifacts:"
ls -lah "$OUT_DIR" | rg -n '(dedup|int4|quant|ptq|mask|exceptions|default|replicate|hot|cache|patch|recon)' || true
