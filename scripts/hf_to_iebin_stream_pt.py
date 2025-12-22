#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import safe_open

ALIGN = 64

def align_up(x: int, a: int) -> int:
    return (x + (a - 1)) & ~(a - 1)

def tensor_dtype_tag(t: torch.Tensor) -> str:
    dt = t.dtype
    if dt == torch.float32: return "f32"
    if dt == torch.float16: return "f16"
    if dt == torch.bfloat16: return "bf16"
    if dt == torch.int8: return "i8"
    if dt == torch.uint8: return "u8"
    if dt == torch.int16: return "i16"
    if dt == torch.int32: return "i32"
    if dt == torch.int64: return "i64"
    if dt == torch.uint16: return "u16"
    if dt == torch.uint32: return "u32"
    if dt == torch.uint64: return "u64"
    raise SystemExit(f"ERROR: unsupported dtype: {dt}")

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        # Store bf16 as raw uint16 bits (no numpy bf16 needed)
        return t.view(torch.uint16).numpy().tobytes(order="C")
    return t.numpy().tobytes(order="C")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"ERROR: missing {idx_path}")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    weight_map = idx.get("weight_map", {})
    if not weight_map:
        raise SystemExit("ERROR: index has no weight_map")

    # Group keys per shard file
    shard_to_keys = {}
    for k, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(k)

    # Deterministic order
    for shard in shard_to_keys:
        shard_to_keys[shard].sort()
    shard_files = sorted(shard_to_keys.keys())

    bin_path = out_dir / "model.ie.bin"
    json_path = out_dir / "model.ie.json"

    meta = {
        "format": "iebin_v1",
        "hf_dir": str(hf_dir),
        "tensors": []
    }

    offset = 0
    with open(bin_path, "wb") as fbin:
        for shard_name in shard_files:
            shard_path = hf_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(f"ERROR: missing shard {shard_path}")

            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                for key in shard_to_keys[shard_name]:
                    t = sf.get_tensor(key)
                    tag = tensor_dtype_tag(t)
                    shape = list(t.shape)

                    raw = tensor_to_bytes(t)
                    offset_aligned = align_up(offset, ALIGN)
                    if offset_aligned != offset:
                        fbin.write(b"\x00" * (offset_aligned - offset))
                        offset = offset_aligned

                    fbin.write(raw)
                    nbytes = len(raw)

                    meta["tensors"].append({
                        "name": key,
                        "dtype": tag,
                        "shape": shape,
                        "offset": offset,
                        "nbytes": nbytes,
                        "shard": shard_name
                    })

                    offset += nbytes

    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[iebin] wrote {json_path} and {bin_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
