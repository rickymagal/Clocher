#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def _get_int(obj, *keys):
    for k in keys:
        v = obj.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, (float, str)):
            try:
                iv = int(v)
                return iv
            except Exception:
                pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Directory containing model.ie.json")
    ap.add_argument("--out", default=None, help="Output tensor_map.json path (default: <model-dir>/tensor_map.json)")
    ap.add_argument("--strict", action="store_true", help="Fail on duplicates/missing fields")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    ie_path = model_dir / "model.ie.json"
    if not ie_path.is_file():
        raise SystemExit(f"ERROR: model.ie.json not found: {ie_path}")

    out_path = Path(args.out).resolve() if args.out else (model_dir / "tensor_map.json")

    ie = json.loads(ie_path.read_text(encoding="utf-8"))
    tensors = ie.get("tensors")
    if not isinstance(tensors, list):
        raise SystemExit("ERROR: model.ie.json has no 'tensors' list")

    seen = set()
    out_tensors = []

    for t in tensors:
        if not isinstance(t, dict):
            if args.strict:
                raise SystemExit("ERROR: non-object tensor entry")
            continue

        name = t.get("name")
        dtype = t.get("dtype")
        shape = t.get("shape")

        offset = _get_int(t, "offset", "offset_bytes")
        size_bytes = _get_int(t, "size_bytes", "nbytes", "size")

        if not isinstance(name, str) or not name:
            if args.strict:
                raise SystemExit("ERROR: tensor missing name")
            continue
        if not isinstance(dtype, str) or not dtype:
            if args.strict:
                raise SystemExit(f"ERROR: tensor '{name}' missing dtype")
            dtype = "F32"
        if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
            if args.strict:
                raise SystemExit(f"ERROR: tensor '{name}' missing/invalid shape")
            shape = []
        if offset is None or size_bytes is None:
            if args.strict:
                raise SystemExit(f"ERROR: tensor '{name}' missing offset/size")
            continue

        if name in seen:
            if args.strict:
                raise SystemExit(f"ERROR: duplicate tensor name: {name}")
            continue
        seen.add(name)

        out_tensors.append({
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "offset": int(offset),
            "size_bytes": int(size_bytes),
        })

    out_obj = {
        "format": "tensor_map_v1",
        "tensors": out_tensors,
    }

    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"OK: wrote {out_path} ({len(out_tensors)} tensors)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
