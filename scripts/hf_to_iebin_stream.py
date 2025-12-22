#!/usr/bin/env python3
import argparse
import gc
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
from safetensors import safe_open

ALIGN = 64

DTYPE_MAP = {
    "F16": ("f16", np.float16),
    "BF16": ("bf16", np.float16),  # fallback: store as f16 if BF16 unsupported in consumer
    "F32": ("f32", np.float32),
    "I8": ("i8", np.int8),
    "U8": ("u8", np.uint8),
    "I16": ("i16", np.int16),
    "U16": ("u16", np.uint16),
    "I32": ("i32", np.int32),
    "U32": ("u32", np.uint32),
    "I64": ("i64", np.int64),
    "U64": ("u64", np.uint64),
}

def align_up(x: int, a: int) -> int:
    return (x + (a - 1)) // a * a

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def load_index(hf_dir: Path) -> Tuple[Dict[str, str], List[str]]:
    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"ERROR: missing {idx_path}")
    idx = read_json(idx_path)
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("ERROR: invalid index.json (missing weight_map)")
    files = sorted({v for v in weight_map.values()})
    return weight_map, files

def np_ensure_contig(a: np.ndarray) -> np.ndarray:
    if not a.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(a)
    return a

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map, shard_files = load_index(hf_dir)

    out_bin = out_dir / "model.ie.bin"
    out_json = out_dir / "model.ie.json"

    # Deterministic ordering of tensor keys
    keys = sorted(weight_map.keys())

    meta: Dict[str, Any] = {
        "format": "iebin_v1",
        "source": {"hf_dir": str(hf_dir)},
        "tensors": {},
    }

    offset = 0
    with open(out_bin, "wb") as fbin:
        # Write tensors sequentially; open shards lazily when needed.
        current_shard = None
        current_path: Path | None = None
        current_keys: List[str] = []

        def open_shard(path: Path):
            return safe_open(str(path), framework="numpy")

        for k in keys:
            shard_name = weight_map[k]
            shard_path = hf_dir / shard_name

            if current_path != shard_path:
                if current_shard is not None:
                    current_shard.__exit__(None, None, None)
                    current_shard = None
                    current_keys = []
                    current_path = None
                    gc.collect()

                if not shard_path.is_file():
                    raise SystemExit(f"ERROR: missing shard {shard_path}")

                current_shard = open_shard(shard_path)
                current_path = shard_path
                current_keys = list(current_shard.keys())

            if current_shard is None or k not in current_keys:
                raise SystemExit(f"ERROR: tensor '{k}' not found in shard '{shard_name}'")

            # Read single tensor
            arr = current_shard.get_tensor(k)

            # Normalize dtype naming
            st_dtype = str(arr.dtype)
            # safetensors gives numpy dtype names; map to our tag set
            if arr.dtype == np.float16:
                dtype_tag = "f16"
                out_arr = arr
            elif arr.dtype == np.float32:
                dtype_tag = "f32"
                out_arr = arr
            elif arr.dtype == np.int8:
                dtype_tag = "i8"
                out_arr = arr
            elif arr.dtype == np.uint8:
                dtype_tag = "u8"
                out_arr = arr
            elif arr.dtype == np.int16:
                dtype_tag = "i16"
                out_arr = arr
            elif arr.dtype == np.uint16:
                dtype_tag = "u16"
                out_arr = arr
            elif arr.dtype == np.int32:
                dtype_tag = "i32"
                out_arr = arr
            elif arr.dtype == np.uint32:
                dtype_tag = "u32"
                out_arr = arr
            elif arr.dtype == np.int64:
                dtype_tag = "i64"
                out_arr = arr
            elif arr.dtype == np.uint64:
                dtype_tag = "u64"
                out_arr = arr
            else:
                raise SystemExit(f"ERROR: unsupported dtype for '{k}': {st_dtype}")

            out_arr = np_ensure_contig(out_arr)
            raw = out_arr.tobytes(order="C")

            # Align
            off_aligned = align_up(offset, ALIGN)
            if off_aligned != offset:
                fbin.write(b"\x00" * (off_aligned - offset))
                offset = off_aligned

            fbin.write(raw)
            nbytes = len(raw)

            meta["tensors"][k] = {
                "dtype": dtype_tag,
                "shape": list(out_arr.shape),
                "offset": offset,
                "nbytes": nbytes,
            }

            offset += nbytes

            # Free ASAP
            del arr
            del out_arr
            gc.collect()

        if current_shard is not None:
            current_shard.__exit__(None, None, None)

    out_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {out_bin}")
    print(f"Wrote: {out_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
