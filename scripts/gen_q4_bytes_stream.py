#!/usr/bin/env python3
# Streamed INT4 (weight-only, per-row) packer for huge HF checkpoints
# - Writes packed INT4 bytes + FP16 scales incrementally (low RAM)
# - Emits expanded manifest mapping each exact key to the INT4 spec
# Usage:
#   python3 scripts/gen_q4_bytes_stream.py \
#     --hf-dir models/gpt-oss-20b/hf \
#     --q4-bytes models/gpt-oss-20b/model.q4.bin \
#     --scales   models/gpt-oss-20b/model.q4.scales.fp16.bin \
#     --manifest quant/q4_manifest.expanded.json \
#     --chunk-rows 256

import argparse, os, re, glob, json, gc
from pathlib import Path
import numpy as np
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--q4-bytes", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--chunk-rows", type=int, default=256)
    return p.parse_args()

# Filters (mesmos do teu pipeline)
INC_W = re.compile(r".*\.weight$")
EXC = [re.compile(x) for x in [
    r".*\.bias$",
    r".*(embed|embedding).*",
    r".*(layernorm|layer_norm|rms_norm|ln).*",
    r".*(norm.*weight).*",
    r".*(lm_head|final_linear)\.weight$",
    r".*embed_out\.weight$",
]]

def wanted(k: str) -> bool:
    if not INC_W.match(k): return False
    for rx in EXC:
        if rx.match(k): return False
    return True

def pack_int4_rows(q_int8: np.ndarray) -> bytes:
    # q_int8 shape: (rows, cols) in [-8..7]
    rows, cols = q_int8.shape
    if cols % 2:
        # pad 1 coluna p/ emparelhar nibble
        q_int8 = np.pad(q_int8, ((0,0),(0,1)), mode="constant")
        cols += 1
    # Usa 0xF para pegar nibble em complemento de dois
    q_u = q_int8.astype(np.int16)
    lo = (q_u[:, 0::2] & 0xF).astype(np.uint8)
    hi = (q_u[:, 1::2] & 0xF).astype(np.uint8)
    packed = (hi << 4) | lo  # shape (rows, cols//2)
    return packed.tobytes()

def main():
    args = parse_args()
    hf_dir = Path(args.hf_dir)
    q4_path = Path(args.q4_bytes).resolve()
    sc_path = Path(args.scales).resolve()
    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    q4_path.parent.mkdir(parents=True, exist_ok=True)

    # Ordem consistente de shards
    shards = sorted(glob.glob(str(hf_dir / "pytorch_model-*.bin")))
    if not shards:
        raise SystemExit(f"No shards in {hf_dir}")

    # Saídas (append desde zero)
    if q4_path.exists(): q4_path.unlink()
    if sc_path.exists(): sc_path.unlink()
    fq = open(q4_path, "ab", buffering=1024*1024)
    fs = open(sc_path, "ab", buffering=1024*1024)

    manifest = {}
    total_keys = 0
    total_rows = 0
    bytes_q = 0
    bytes_s = 0

    for shard in shards:
        sd = torch.load(shard, map_location="cpu")
        for k, W in sd.items():
            if not (wanted(k) and isinstance(W, torch.Tensor) and W.ndim == 2 and W.is_floating_point()):
                continue

            W = W.float()  # garante FP32
            rows, cols = W.shape
            rows_done = 0

            # Manifest entry para ESTA key
            manifest[k] = {
                "dtype": "int4",
                "scheme": "weight_only",
                "per": "row",
                "bits": 4,
                "pack": "nibble_lohi",
                "scale_dtype": "fp16",
                "zero_point": 0,
                "symmetric": True,
                "align": 256,
                "rows": rows,
                "cols": cols,
                "packed_bin": str(q4_path),
                "scale_bin": str(sc_path),
            }

            with torch.no_grad():
                step = max(1, args.chunk_rows)
                for start in range(0, rows, step):
                    end = min(start + step, rows)
                    chunk = W[start:end, :]  # [r, cols]

                    # scales per-row: max(|w|)/7 -> range [-7..7]
                    maxabs = torch.amax(torch.abs(chunk), dim=1).clamp_min(1e-12)  # [r]
                    scales = (maxabs / 7.0).cpu().numpy().astype(np.float16)       # [r]
                    fs.write(scales.tobytes()); bytes_s += scales.nbytes

                    inv = (7.0 / maxabs).unsqueeze(1)                               # [r,1]
                    q = torch.round(chunk * inv).clamp_(-8, 7).to(torch.int8).cpu().numpy()  # [r, cols]

                    fq.write(pack_int4_rows(q)); bytes_q += ( (q.shape[1] + (q.shape[1]&1)) // 2 ) * q.shape[0]
                    rows_done += (end - start)
                    total_rows += (end - start)

            total_keys += 1
            # solta memória cedo
            del W
        del sd
        gc.collect()

    fq.flush(); fs.flush()
    fq.close(); fs.close()

    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] keys={total_keys} rows={total_rows} q4={bytes_q/1024/1024:.2f} MiB scales={bytes_s/1024/1024:.2f} MiB")
    print(f"[ok] wrote manifest -> {man_path.resolve()}")
    print(f"[ok] q4 bytes      -> {q4_path}")
    print(f"[ok] scales        -> {sc_path}")

if __name__ == "__main__":
    main()
