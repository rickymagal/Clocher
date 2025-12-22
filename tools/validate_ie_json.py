#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

DTYPE_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
}

def prod(xs: List[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out

def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}"
        v /= 1024.0
    return f"{n} B"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ie-json", required=True, help="Path to model.ie.json")
    ap.add_argument("--check-shards", action="store_true", help="Verify shard_data_offset+nbytes within shard file size")
    args = ap.parse_args()

    ie_path = Path(args.ie_json)
    data = json.loads(ie_path.read_text())

    if "tensors" not in data or not isinstance(data["tensors"], list):
        raise SystemExit("ERROR: model.ie.json missing 'tensors' list")

    hf_dir = Path(data.get("hf_dir", "")).expanduser()
    tensors = data["tensors"]

    # 1) dtype/shape/nbytes consistency + duplicate names
    seen = set()
    bad = 0
    for t in tensors:
        name = t["name"]
        if name in seen:
            print(f"ERROR: duplicate tensor name: {name}")
            bad += 1
        seen.add(name)

        dtype = t["dtype"]
        shape = t["shape"]
        nbytes = int(t["nbytes"])
        if dtype not in DTYPE_SIZES:
            print(f"ERROR: unknown dtype {dtype} for {name}")
            bad += 1
            continue

        expected = prod(shape) * DTYPE_SIZES[dtype]
        if expected != nbytes:
            print(f"ERROR: nbytes mismatch for {name}: got {nbytes}, expected {expected}")
            bad += 1

    # 2) packed layout overlap check
    spans: List[Tuple[int, int, str]] = []
    for t in tensors:
        off = int(t["offset"])
        nb = int(t["nbytes"])
        spans.append((off, off + nb, t["name"]))
    spans.sort(key=lambda x: x[0])

    last_end = 0
    for i, (s, e, name) in enumerate(spans):
        if e < s:
            print(f"ERROR: negative span for {name}: [{s}, {e})")
            bad += 1
        if i == 0:
            last_end = e
            continue
        prev_s, prev_e, prev_name = spans[i - 1]
        if s < prev_e:
            print(f"ERROR: overlap: {prev_name} [{prev_s},{prev_e}) overlaps {name} [{s},{e})")
            bad += 1
        last_end = max(last_end, e)

    packed_min = spans[0][0] if spans else 0
    packed_max = spans[-1][1] if spans else 0
    print(f"Packed range: [{packed_min}, {packed_max}) = {human_bytes(packed_max - packed_min)}")
    print(f"Tensors: {len(tensors)}; duplicates: {len(tensors) - len(seen)}")

    # 3) optional shard bound checks
    if args.check_shards:
        if not hf_dir:
            print("WARNING: hf_dir missing; cannot check shards")
        else:
            shard_sizes: Dict[str, int] = {}
            for t in tensors:
                shard = t.get("shard")
                if not shard:
                    continue
                shard_path = hf_dir / shard
                if shard not in shard_sizes:
                    try:
                        shard_sizes[shard] = shard_path.stat().st_size
                    except FileNotFoundError:
                        print(f"ERROR: shard file not found: {shard_path}")
                        bad += 1
                        shard_sizes[shard] = -1

                size = shard_sizes[shard]
                if size >= 0:
                    sdo = int(t["shard_data_offset"])
                    nb = int(t["nbytes"])
                    if sdo < 0 or sdo + nb > size:
                        print(
                            f"ERROR: shard OOB: {t['name']} needs [{sdo}, {sdo + nb}) "
                            f"but {shard} size is {size}"
                        )
                        bad += 1

    if bad == 0:
        print("OK: model.ie.json passes validation checks.")
        return 0

    print(f"FAILED: {bad} error(s).")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
