#!/usr/bin/env bash
set -eu

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/rebuild_dedup_and_quant.sh --model-dir <path> [--align 256]

This script:
  1) Rebuilds model.ie.compat.json + model.q4.bin by repacking ALL tensors
     from model.ie.json + model.ie.bin into an aligned binary.
  2) Symlinks those artifacts into the model directory root.
  3) Packs a padded tokenizer (optional but recommended).
  4) Regenerates tensor_map.json for runtime lookup.

Notes:
- This script does not set pipefail.
- This script does not perform quantization.
EOF
}

MODEL_DIR=""
ALIGN="256"

while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --align)
      ALIGN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [ -z "$MODEL_DIR" ]; then
  echo "error: --model-dir is required" >&2
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR_ABS="$(cd "$MODEL_DIR" && pwd)"

OUT_DIR="$MODEL_DIR_ABS/dedup_quant_out"
mkdir -p "$OUT_DIR"

OUT_BIN="$OUT_DIR/model.q4.bin"
OUT_MAP="$OUT_DIR/model.ie.compat.json"

echo "[quant] tool: $ROOT/tools/build_hybrid_moe_int4.py" >&2
echo "[quant] model_dir: $MODEL_DIR_ABS" >&2
echo "[quant] out_bin:   $OUT_BIN" >&2
echo "[quant] out_map:   $OUT_MAP" >&2
echo "[quant] align:     $ALIGN" >&2

python3 "$ROOT/tools/build_hybrid_moe_int4.py" \
  --model-dir "$MODEL_DIR_ABS" \
  --out-bin "$OUT_BIN" \
  --out-map "$OUT_MAP" \
  --align "$ALIGN"

ln -sf "$OUT_BIN" "$MODEL_DIR_ABS/model.q4.bin"
ln -sf "$OUT_MAP" "$MODEL_DIR_ABS/model.ie.compat.json"

HF_TOK="$MODEL_DIR_ABS/hf/original/tokenizer.json"

if [ -f "$HF_TOK" ]; then
  echo "[tokenizer] pack: $HF_TOK -> $MODEL_DIR_ABS/tokenizer.ie.bin" >&2
  cp -f "$HF_TOK" "$MODEL_DIR_ABS/tokenizer.json"
  python3 "$ROOT/scripts/pack_tokenizer.py" --model-dir "$MODEL_DIR_ABS"
else
  echo "[tokenizer] skipped: missing $HF_TOK" >&2
fi

echo "[tensor_map] build from: $MODEL_DIR_ABS/model.ie.json" >&2
python3 "$ROOT/scripts/make_tensor_map_from_iejson.py" \
  --model-dir "$MODEL_DIR_ABS" \
  --out "$MODEL_DIR_ABS/tensor_map.json"

echo "[ok] artifacts:" >&2
ls -lh \
  "$MODEL_DIR_ABS/model.ie.json" \
  "$MODEL_DIR_ABS/model.ie.bin" \
  "$MODEL_DIR_ABS/model.ie.compat.json" \
  "$MODEL_DIR_ABS/model.q4.bin" \
  "$MODEL_DIR_ABS/tensor_map.json" \
  2>/dev/null || true
