#!/usr/bin/env bash
set -euo pipefail

HF_DIR="${1:-models/gpt-oss-20b/hf}"
OUT_DIR="${2:-models/gpt-oss-20b}"
Q4="${OUT_DIR}/model.q4.bin"
SC="${OUT_DIR}/model.q4.scales.fp16.bin"
PARTS_DIR="quant/parts"
FINAL_MAN="quant/q4_manifest.expanded.json"
CHUNK="${CHUNK_ROWS:-128}"

mkdir -p "$(dirname "$Q4")" "$PARTS_DIR" "quant"

# zera blobs (recomeço controlado)
: > "$Q4"
: > "$SC"

# processa cada shard em processo isolado (RAM zera a cada shard)
for s in $(ls -1 "${HF_DIR}"/pytorch_model-*.bin | sort); do
  p="$PARTS_DIR/$(basename "$s").json"
  echo "[driver] shard=$(basename "$s")"
  /usr/bin/env -i PATH="$PATH" LC_ALL=C.UTF-8 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    python3 -u scripts/gen_q4_bytes_worker.py \
      --shard "$s" --q4-bytes "$Q4" --scales "$SC" \
      --manifest-part "$p" --chunk-rows "$CHUNK"
done

# merge ordenado dos manifests
python3 - <<'PY'
import json,glob,sys
parts = sorted(glob.glob("quant/parts/pytorch_model-*.json"))
merged = {}
for p in parts:
    with open(p,"r") as f:
        d = json.load(f)
    # mantém ordem de leitura das keys por shard; dict mantém ordem em Py>=3.7
    for k,v in d.items():
        merged[k]=v
with open("quant/q4_manifest.expanded.json","w") as f:
    json.dump(merged,f,indent=2)
print(f"[driver] wrote quant/q4_manifest.expanded.json with {len(merged)} entries")
PY

echo "[driver] done. q4=$(du -h "$Q4" | awk '{print $1}') scales=$(du -h "$SC" | awk '{print $1}')"
