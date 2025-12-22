#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-models/gpt-oss-20b}"
MANIFEST="${2:-${MODEL_DIR}/manifest.dedup.json}"

SCALE_ENCODING="${SCALE_ENCODING:-fp16}"   # fp16 or log2_u8_q3
GROUP_SIZE="${GROUP_SIZE:-64}"

BASE_IE_JSON="${MODEL_DIR}/model.ie.json"

OUT_DIR="${MODEL_DIR}/dedup_quant_out"
mkdir -p "${OUT_DIR}"

OUT_BIN="${OUT_DIR}/dedup_source.bin"
OUT_IE_JSON="${OUT_DIR}/dedup_source.ie.json"
TENSOR_MAP="${OUT_DIR}/tensor_map.for_dedup.json"
GROUPS="${OUT_DIR}/groups.for_dedup.json"

echo "[1/3] Build compact dedup+quant source"
python3 tools/build_dedup_source_from_manifest.py \
  --base-ie-json "${BASE_IE_JSON}" \
  --manifest "${MANIFEST}" \
  --out-bin "${OUT_BIN}" \
  --out-ie-json "${OUT_IE_JSON}" \
  --group-size "${GROUP_SIZE}" \
  --scale-encoding "${SCALE_ENCODING}" \
  --include-blocks

echo "[2/3] Prepare tensor_map + groups"
python3 tools/dedup_prepare_and_extract_all.py \
  --ie-json "${OUT_IE_JSON}" \
  --manifest "${MANIFEST}" \
  --out-tensor-map "${TENSOR_MAP}" \
  --out-groups "${GROUPS}"

echo "[3/3] Run lossless dedup extractor"
python3 tools/dedup_extract_int4.py \
  --model-dir "${OUT_DIR}" \
  --tensor-map "${TENSOR_MAP}" \
  --groups "${GROUPS}" \
  --out-prefix "${MODEL_DIR}/model.dedup"

echo "Done."
echo "Outputs:"
echo "  ${MODEL_DIR}/model.dedup.defaults.bin"
echo "  ${MODEL_DIR}/model.dedup.masks.bin"
echo "  ${MODEL_DIR}/model.dedup.exceptions.bin"
echo "  ${MODEL_DIR}/model.dedup.json"
