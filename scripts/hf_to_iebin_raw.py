#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

ALIGN = 64

def align_up(x: int, a: int = ALIGN) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_safetensors_header(f):
    # safetensors format: u64 little-endian header length, then JSON header, then raw tensors
    import struct
    hdr_len_bytes = f.read(8)
    if len(hdr_len_bytes) != 8:
        raise RuntimeError("Failed to read safetensors header length")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_json = f.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise RuntimeError("Failed to read safetensors header JSON")
    header = json.loads(hdr_json.decode("utf-8"))
    return header, 8 + hdr_len

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="Directory containing model.safetensors.index.json and shard .safetensors files")
    ap.add_argument("--out-dir", required=True, help="Output directory for model.ie.json/bin")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"ERROR: missing {idx_path}")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    weight_map = idx.get("weight_map", {})
    if not weight_map:
        raise SystemExit("ERROR: index has empty weight_map")

    # Group keys by shard filename
    shard_to_keys = {}
    for k, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(k)

    out_bin = out_dir / "model.ie.bin"
    out_json = out_dir / "model.ie.json"

    # Deterministic order
    shards = sorted(shard_to_keys.keys())
    for s in shards:
        shard_to_keys[s].sort()

    manifest = {
        "format": "iebin_v1_raw_safetensors",
        "hf_dir": str(hf_dir),
        "tensors": []
    }

    # Write bin
    cur_off = 0
    with open(out_bin, "wb") as bout:
        for shard_name in shards:
            shard_path = hf_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(f"ERROR: missing shard {shard_path}")

            with open(shard_path, "rb") as fin:
                header, data_base = read_safetensors_header(fin)

                for key in shard_to_keys[shard_name]:
                    meta = header.get(key)
                    if meta is None:
                        raise SystemExit(f"ERROR: tensor '{key}' not found in header of {shard_name}")

                    dtype = meta.get("dtype")
                    shape = meta.get("shape")
                    offs = meta.get("data_offsets")
                    if dtype is None or shape is None or offs is None:
                        raise SystemExit(f"ERROR: malformed meta for '{key}' in {shard_name}")

                    start, end = int(offs[0]), int(offs[1])
                    nbytes = end - start
                    if nbytes <= 0:
                        raise SystemExit(f"ERROR: bad data_offsets for '{key}' in {shard_name}: {offs}")

                    # Align in output
                    aligned = align_up(cur_off, ALIGN)
                    if aligned != cur_off:
                        bout.write(b"\x00" * (aligned - cur_off))
                        cur_off = aligned

                    # Copy raw bytes directly from safetensors payload
                    fin.seek(data_base + start, os.SEEK_SET)

                    remaining = nbytes
                    buf_sz = 8 * 1024 * 1024
                    while remaining > 0:
                        chunk = fin.read(min(buf_sz, remaining))
                        if not chunk:
                            raise SystemExit(f"ERROR: unexpected EOF while reading '{key}' from {shard_name}")
                        bout.write(chunk)
                        remaining -= len(chunk)

                    manifest["tensors"].append({
                        "name": key,
                        "dtype": dtype,          # e.g., "BF16", "F16", "F32", ...
                        "shape": shape,
                        "nbytes": nbytes,
                        "offset": cur_off,
                        "shard": shard_name,
                        "shard_data_offset": start,
                    })
                    cur_off += nbytes

    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_json} and {out_bin}")

if __name__ == "__main__":
    main()
