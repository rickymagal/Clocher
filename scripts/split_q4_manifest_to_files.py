#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--q4-bytes", required=True)
    ap.add_argument("--scales", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-map", required=True)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    q4_path = Path(args.q4_bytes)
    sc_path = Path(args.scales)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q4 = q4_path.read_bytes()
    sc = sc_path.read_bytes()

    items = []
    used = set()
    for name, spec in man.items():
        rows = int(spec["rows"])
        cols = int(spec["cols"])

        q4_off = int(spec.get("q4_offset", spec.get("int4_offset", 0)))
        sc_off = int(spec.get("scale_offset", 0))

        q4_len = int(spec.get("q4_nbytes", rows * ((cols + 1) // 2)))
        sc_len = int(spec.get("scale_nbytes", rows * 2))

        base = safe_name(name)
        if base in used:
            i = 1
            while f"{base}_{i}" in used:
                i += 1
            base = f"{base}_{i}"
        used.add(base)

        int4_file = out_dir / f"{base}.int4"
        scale_file = out_dir / f"{base}.scale.fp16"

        int4_file.write_bytes(q4[q4_off:q4_off + q4_len])
        scale_file.write_bytes(sc[sc_off:sc_off + sc_len])

        items.append({
            "name": name,
            "int4_bin": str(int4_file.resolve()),
            "scale_bin": str(scale_file.resolve()),
            "per": "row",
            "rows": rows,
            "cols": cols,
            "scale_dtype": "fp16",
            "pack": "nibble_lohi",
            "zp": 0,
            "symmetric": True,
        })

    Path(args.out_map).write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"[ok] wrote {len(items)} tensors into {out_dir}")
    print(f"[ok] wrote q4 map -> {args.out_map}")

if __name__ == "__main__":
    main()
