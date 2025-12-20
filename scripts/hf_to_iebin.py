# tools/hf_to_iebin.py
import json, os, sys
from pathlib import Path

def sizeof_dtype(dtype: str) -> int:
    # Map torch dtype string to element size in bytes.
    return {
        "torch.uint8": 1,
        "torch.int8": 1,
        "torch.bfloat16": 2,
        "torch.float16": 2,
        "torch.float32": 4,
        "torch.int32": 4,
        "torch.int64": 8,
    }.get(dtype, None)

def main() -> int:
    model_dir = Path("models/gpt-oss-20b")
    hf_dir = model_dir / "hf"
    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        print(f"ERROR: missing {idx_path}", file=sys.stderr)
        return 2

    # Lazy import to avoid forcing a dependency if not needed elsewhere.
    try:
        from safetensors.numpy import load_file as st_load_numpy
    except Exception as e:
        print("ERROR: please install safetensors:  pip install safetensors", file=sys.stderr)
        raise

    index = json.loads(idx_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        print("ERROR: index JSON missing non-empty weight_map", file=sys.stderr)
        return 2

    # Collect tensors per shard.
    shard_to_tensors = {}
    for name, shard in weight_map.items():
        shard_to_tensors.setdefault(shard, []).append(name)

    # Note: we read each shard once, write tensors contiguously in output bin.
    out_bin_path = model_dir / "model.ie.bin"
    out_json_path = model_dir / "model.ie.json"
    tensors_meta = []
    offset = 0

    with out_bin_path.open("wb") as fout:
        for shard_rel in sorted(shard_to_tensors):
            shard_path = hf_dir / shard_rel
            if not shard_path.exists():
                print(f"ERROR: shard file missing: {shard_path}", file=sys.stderr)
                return 2

            arrs = st_load_numpy(str(shard_path))
            # Iterate tensors present in this shard, keep stable order.
            for name in sorted(shard_to_tensors[shard_rel]):
                if name not in arrs:
                    print(f"ERROR: tensor {name} not found in {shard_rel}", file=sys.stderr)
                    return 2
                arr = arrs[name]
                # Normalize dtype string to torch-like tag
                npdt = str(arr.dtype)
                if npdt == "uint8":
                    dtype = "torch.uint8"
                elif npdt in ("bfloat16", "bf16"):
                    dtype = "torch.bfloat16"
                elif npdt in ("float16",):
                    dtype = "torch.float16"
                elif npdt in ("float32",):
                    dtype = "torch.float32"
                elif npdt in ("int32",):
                    dtype = "torch.int32"
                elif npdt in ("int64",):
                    dtype = "torch.int64"
                else:
                    # Fall back: try itemsize
                    dtype = f"torch.unknown({npdt})"

                esize = arr.dtype.itemsize
                need = sizeof_dtype(dtype)
                if need is None:
                    # Unknown dtype: still write raw bytes as produced by numpy
                    need = esize
                # Write raw bytes C-contiguous
                buf = arr.tobytes(order="C")
                nbytes = len(buf)
                fout.write(buf)

                tensors_meta.append({
                    "name": name,
                    "dtype": dtype,
                    "shape": list(arr.shape),
                    "offset": offset,
                    "nbytes": nbytes,
                })
                offset += nbytes

    out = {
        "weights_bin": "model.ie.bin",
        "tensors": tensors_meta,
    }
    out_json_path.write_text(json.dumps(out, indent=2))
    print(f"[iebin] wrote: {out_json_path} (tensors={len(tensors_meta)})")
    print(f"[iebin] wrote: {out_bin_path} (bytes={out_bin_path.stat().st_size})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
