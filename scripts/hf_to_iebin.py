#!/usr/bin/env python3
import argparse
import json
import os
import struct
from pathlib import Path

ALIGN = 64

def align_up(x: int, a: int = ALIGN) -> int:
    return (x + (a - 1)) & ~(a - 1)

def read_safetensors_header(fin):
    hdr_len_bytes = fin.read(8)
    if len(hdr_len_bytes) != 8:
        raise RuntimeError("Failed to read safetensors header length")
    (hdr_len,) = struct.unpack("<Q", hdr_len_bytes)
    hdr_json = fin.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise RuntimeError("Failed to read safetensors header JSON")
    header = json.loads(hdr_json.decode("utf-8"))
    return header, 8 + hdr_len

def infer_top_dtype(tensors):
    dtypes = {t.get("dtype") for t in tensors if t.get("dtype")}
    if len(dtypes) == 1:
        dt = next(iter(dtypes))
        if dt == "BF16":
            return "bf16"
        if dt == "F16":
            return "f16"
        if dt == "F32":
            return "f32"
        return dt.lower()
    if "BF16" in dtypes:
        return "bf16"
    if "F16" in dtypes:
        return "f16"
    if "F32" in dtypes:
        return "f32"
    return "mixed"

def _load_q4_map(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "rules" in obj:
        raise SystemExit("ERROR: --q4-map expects an expanded manifest (not rules).")
    if not isinstance(obj, dict):
        raise SystemExit("ERROR: --q4-map must be a JSON object")
    specs = {}
    for name, meta in obj.items():
        if not isinstance(meta, dict):
            continue
        if "q4_offset" in meta or "int4_bin" in meta:
            specs[name] = meta
    if not specs:
        raise SystemExit("ERROR: --q4-map has no expanded entries (missing q4_offset or int4_bin)")
    return specs

def _resolve_rel(base: Path, p: str) -> Path:
    if os.path.isabs(p):
        return Path(p)
    return (base / p).resolve()

def _read_slice(path: Path, offset: int, nbytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        data = f.read(nbytes)
    if len(data) != nbytes:
        raise SystemExit(f"ERROR: short read from {path} (wanted {nbytes}, got {len(data)})")
    return data

def _q4_supported(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if ".mlp.experts." in name:
        return False
    if ".self_attn." in name:
        return True
    if name in ("lm_head.weight", "model.lm_head.weight", "transformer.lm_head.weight"):
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, help="Directory containing model.safetensors.index.json and shard .safetensors files")
    ap.add_argument("--out-dir", required=True, help="Output directory for model.ie.json/bin")
    ap.add_argument("--q4-map", default=None, help="Expanded Q4 manifest (e.g., quant/q4_manifest.expanded.json)")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise SystemExit(f"ERROR: missing {idx_path}")

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    weight_map = idx.get("weight_map", {})
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("ERROR: index has empty weight_map")

    shard_to_keys = {}
    for name, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(name)

    shards = sorted(shard_to_keys.keys())
    for s in shards:
        shard_to_keys[s].sort()

    q4_specs = None
    q4_base = None
    if args.q4_map:
        q4_map_path = Path(args.q4_map).resolve()
        if not q4_map_path.is_file():
            raise SystemExit(f"ERROR: --q4-map not found: {q4_map_path}")
        q4_specs = _load_q4_map(q4_map_path)
        q4_base = q4_map_path.parent

    out_bin = out_dir / ("model.q4.bin" if q4_specs else "model.ie.bin")
    out_json = out_dir / ("model.ie.compat.json" if q4_specs else "model.ie.json")

    tensors = []
    cur_off = 0

    with open(out_bin, "wb") as bout:
        for shard_name in shards:
            shard_path = hf_dir / shard_name
            if not shard_path.is_file():
                raise SystemExit(f"ERROR: missing shard {shard_path}")

            with open(shard_path, "rb") as fin:
                header, data_base = read_safetensors_header(fin)

                for key in shard_to_keys[shard_name]:
                    if q4_specs and key in q4_specs and _q4_supported(key):
                        meta = q4_specs[key]
                        q4_bin = meta.get("packed_bin") or meta.get("int4_bin")
                        sc_bin = meta.get("scale_bin")
                        if not q4_bin or not sc_bin:
                            raise SystemExit(f"ERROR: q4 entry '{key}' missing packed_bin/scale_bin")

                        q4_path = _resolve_rel(q4_base, str(q4_bin))
                        sc_path = _resolve_rel(q4_base, str(sc_bin))

                        if "q4_offset" in meta:
                            q4_bytes = _read_slice(q4_path, int(meta["q4_offset"]), int(meta["q4_nbytes"]))
                        else:
                            q4_bytes = Path(q4_path).read_bytes()

                        if "scale_offset" in meta:
                            sc_bytes = _read_slice(sc_path, int(meta["scale_offset"]), int(meta["scale_nbytes"]))
                        else:
                            sc_bytes = Path(sc_path).read_bytes()

                        rows = int(meta.get("rows", 0) or 0)
                        cols = int(meta.get("cols", 0) or 0)
                        scale_dtype = str(meta.get("scale_dtype", "fp16")).lower()
                        scale_dtype = "F16" if "16" in scale_dtype else "BF16"
                        per = str(meta.get("per", "row")).lower()
                        sc_shape = [rows] if per == "row" and rows > 0 else [1]

                        aligned = align_up(cur_off, ALIGN)
                        if aligned != cur_off:
                            bout.write(b"\x00" * (aligned - cur_off))
                            cur_off = aligned

                        q4_off = cur_off
                        bout.write(q4_bytes)
                        cur_off += len(q4_bytes)

                        aligned = align_up(cur_off, ALIGN)
                        if aligned != cur_off:
                            bout.write(b"\x00" * (aligned - cur_off))
                            cur_off = aligned

                        sc_off = cur_off
                        bout.write(sc_bytes)
                        cur_off += len(sc_bytes)

                        tensors.append({
                            "name": f"{key}_blocks",
                            "dtype": "int4",
                            "shape": [rows, cols] if rows and cols else [],
                            "nbytes": len(q4_bytes),
                            "offset": q4_off,
                        })
                        tensors.append({
                            "name": f"{key}_scales",
                            "dtype": scale_dtype,
                            "shape": sc_shape,
                            "nbytes": len(sc_bytes),
                            "offset": sc_off,
                        })

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
                            raise SystemExit(f"ERROR: unexpected EOF while reading '{key}' from {shard_name}")
                        bout.write(chunk)
                        remaining -= len(chunk)

                    tensors.append({
                        "name": key,
                        "dtype": dtype,
                        "shape": shape,
                        "nbytes": nbytes,
                        "offset": cur_off,
                    })
                    cur_off += nbytes

    manifest = {
        "version": 1,
        "dtype": infer_top_dtype(tensors),
        "weights_bin": out_bin.name,
        "format": "iebin_v1_raw_safetensors",
        "hf_dir": str(hf_dir),
        "tensors": tensors,
    }

    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_json} and {out_bin}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
