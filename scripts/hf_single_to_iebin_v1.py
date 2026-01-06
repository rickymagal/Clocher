#!/usr/bin/env python3
import argparse
import json
import os
import struct
from collections import Counter
from pathlib import Path

ALIGN = 64

def align_up(x: int, a: int = ALIGN) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_safetensors_header(f):
    hdr_len_bytes = f.read(8)
    if len(hdr_len_bytes) != 8:
        raise RuntimeError("Failed to read safetensors header length")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_json = f.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise RuntimeError("Failed to read safetensors header JSON")
    header = json.loads(hdr_json.decode("utf-8"))
    return header, 8 + hdr_len

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True, help="Path to a single model.safetensors file")
    ap.add_argument("--dtypes", required=True, help="Path to dtypes.json (must provide engine dtype codes)")
    ap.add_argument("--out-dir", required=True, help="Output directory for model.ie.json and model.ie.bin")
    args = ap.parse_args()

    st_path = Path(args.safetensors).resolve()
    dt_path = Path(args.dtypes).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not st_path.is_file():
        raise SystemExit(f"ERROR: missing {st_path}")
    if not dt_path.is_file():
        raise SystemExit(f"ERROR: missing {dt_path}")

    dtypes_map = load_json(dt_path)
    if not isinstance(dtypes_map, dict) or not dtypes_map:
        raise SystemExit("ERROR: dtypes.json is not a non-empty object")

    out_bin = out_dir / "model.ie.bin"
    out_json = out_dir / "model.ie.json"

    with open(st_path, "rb") as fin:
        header, data_base = read_safetensors_header(fin)

        # safetensors headers include "__metadata__"
        keys = [k for k in header.keys() if k != "__metadata__"]
        keys.sort()

        manifest = {
            "format": "iebin_v1",
            "version": 1,
            "weights_bin": "model.ie.bin",
            "dtype": None,  # set below (mode)
            "tensors": [],
        }

        dtype_hist = Counter()

        cur_off = 0
        with open(out_bin, "wb") as bout:
            for key in keys:
                meta = header.get(key, None)
                if meta is None:
                    raise SystemExit(f"ERROR: missing meta for tensor '{key}'")

                shape = meta.get("shape")
                offs = meta.get("data_offsets")
                if shape is None or offs is None:
                    raise SystemExit(f"ERROR: malformed meta for '{key}'")

                start, end = int(offs[0]), int(offs[1])
                nbytes = end - start
                if nbytes <= 0:
                    raise SystemExit(f"ERROR: bad data_offsets for '{key}': {offs}")

                dt = dtypes_map.get(key, None)
                if dt is None:
                    raise SystemExit(f"ERROR: dtypes.json has no entry for '{key}'")

                if not isinstance(dt, int):
                    raise SystemExit(f"ERROR: dtypes.json dtype for '{key}' is not an int: {dt!r}")

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
                        raise SystemExit(f"ERROR: unexpected EOF while reading '{key}'")
                    bout.write(chunk)
                    remaining -= len(chunk)

                manifest["tensors"].append({
                    "name": key,
                    "dtype": dt,
                    "shape": shape,
                    "offset": cur_off,
                    "size_bytes": nbytes,
                })
                dtype_hist[dt] += 1
                cur_off += nbytes

    # Choose a top-level dtype = most common per-tensor dtype (matches existing IEBIN style)
    if not dtype_hist:
        raise SystemExit("ERROR: wrote no tensors")
    manifest["dtype"] = dtype_hist.most_common(1)[0][0]

    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_json} and {out_bin}")
    print(f"Tensors: {len(manifest['tensors'])}")

if __name__ == "__main__":
    main()
