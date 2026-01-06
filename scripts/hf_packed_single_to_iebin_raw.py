#!/usr/bin/env python3
import argparse
import json
import os
import struct
from pathlib import Path

ALIGN = 64

def align_up(x: int, a: int = ALIGN) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_safetensors_header(fp):
    hdr_len_bytes = fp.read(8)
    if len(hdr_len_bytes) != 8:
        raise RuntimeError("Failed to read safetensors header length")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_json = fp.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise RuntimeError("Failed to read safetensors header JSON")
    header = json.loads(hdr_json.decode("utf-8"))
    return header, 8 + hdr_len

def load_dtypes_map(path: Path):
    j = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(j, dict) or not j:
        raise SystemExit(f"ERROR: {path} is not a non-empty dict")
    return j

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True, help="Path to packed model.safetensors")
    ap.add_argument("--dtypes-json", required=True, help="Path to dtypes.json (name -> dtype_id)")
    ap.add_argument("--out-dir", required=True, help="Output directory for model.ie.json/bin")
    args = ap.parse_args()

    st_path = Path(args.safetensors).resolve()
    dt_path = Path(args.dtypes_json).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not st_path.is_file():
        raise SystemExit(f"ERROR: missing {st_path}")
    if not dt_path.is_file():
        raise SystemExit(f"ERROR: missing {dt_path}")

    dtypes_map = load_dtypes_map(dt_path)

    out_bin = out_dir / "model.ie.bin"
    out_json = out_dir / "model.ie.json"

    cur_off = 0
    manifest = {
        "format": "iebin_v1",
        "tensors": []
    }

    with open(st_path, "rb") as fin, open(out_bin, "wb") as bout:
        header, data_base = read_safetensors_header(fin)

        # safetensors header includes a "__metadata__" key sometimes; skip non-tensors
        keys = [k for k in header.keys() if k != "__metadata__"]
        keys.sort()

        for name in keys:
            meta = header.get(name)
            if not isinstance(meta, dict):
                continue

            offs = meta.get("data_offsets")
            shape = meta.get("shape")
            if offs is None or shape is None:
                raise SystemExit(f"ERROR: malformed meta for '{name}'")

            start, end = int(offs[0]), int(offs[1])
            nbytes = end - start
            if nbytes <= 0:
                raise SystemExit(f"ERROR: bad data_offsets for '{name}': {offs}")

            # dtype_id must come from dtypes.json (no guessing)
            if name not in dtypes_map:
                raise SystemExit(f"ERROR: '{name}' missing from dtypes.json (refusing to guess dtype_id)")
            dtype_id = int(dtypes_map[name])

            aligned = align_up(cur_off, ALIGN)
            if aligned != cur_off:
                bout.write(b"\x00" * (aligned - cur_off))
                cur_off = aligned

            fin.seek(data_base + start, os.SEEK_SET)

            remaining = nbytes
            buf_sz = 8 * 1024 * 1024
            while remaining > 0:
                chunk = fin.read(min(buf_sz, remaining))
                if not chunk:
                    raise SystemExit(f"ERROR: unexpected EOF while reading '{name}'")
                bout.write(chunk)
                remaining -= len(chunk)

            manifest["tensors"].append({
                "name": name,
                "dtype": dtype_id,
                "shape": shape,
                "nbytes": nbytes,
                "offset": cur_off,
            })
            cur_off += nbytes

    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_json} and {out_bin}")

if __name__ == "__main__":
    main()
