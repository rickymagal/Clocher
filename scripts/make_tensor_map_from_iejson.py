#!/usr/bin/env python3
"""
Generate models/<MODEL>/tensor_map.json from models/<MODEL>/model.ie.json
in the exact schema expected by engine/src/io/tensor_map.c.

Output schema:
{
  "tensors": [
    {
      "name": "...",
      "dtype": "bf16" | "u8" | "int4",
      "offset_bytes": <int>,
      "size_bytes": <int>,
      "shape": [ ... ]   # optional if present in model.ie.json
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


ALLOWED_DTYPES = {"bf16", "u8", "int4"}


def map_dtype(dt: Any) -> str:
    """
    tensor_map.c accepts only: bf16, u8, int4.
    model.ie.json in this repo often uses numeric dtypes:
      0: f32, 1: f16, 2: bf16, 3: s8, 4: u8
    Everything else is conservatively mapped to u8 to keep loader happy.
    """
    if isinstance(dt, int):
        if dt == 2:
            return "bf16"
        if dt in (3, 4):
            return "u8"
        return "u8"

    if isinstance(dt, str):
        s = dt.strip().lower()
        if s in ALLOWED_DTYPES:
            return s
        if s in ("bf16", "bfloat16"):
            return "bf16"
        if s in ("u8", "uint8", "byte"):
            return "u8"
        if s in ("int4", "i4"):
            return "int4"
        return "u8"

    return "u8"


def get_int_field(obj: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for k in keys:
        v = obj.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            pass
    return int(default)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate tensor_map.json from model.ie.json (schema compatible with tensor_map.c)."
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Model directory containing model.ie.json (e.g., models/gpt-oss-20b).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for tensor_map.json (default: <model-dir>/tensor_map.json).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any tensor is missing offset/size fields or has unknown dtype.",
    )

    args = ap.parse_args()

    model_dir = args.model_dir
    ie_path = os.path.join(model_dir, "model.ie.json")
    out_path = args.out or os.path.join(model_dir, "tensor_map.json")

    if not os.path.isfile(ie_path):
        print(f"error: model.ie.json not found: {ie_path}", file=sys.stderr)
        return 2

    with open(ie_path, "r", encoding="utf-8") as f:
        ie = json.load(f)

    tensors_in = ie.get("tensors")
    if not isinstance(tensors_in, list):
        print("error: model.ie.json missing tensors[]", file=sys.stderr)
        return 2

    tensors_out: List[Dict[str, Any]] = []
    unknown_dtype_count = 0
    missing_fields_count = 0

    for t in tensors_in:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name:
            continue

        off = get_int_field(t, "offset_bytes", "offset", default=0)
        sz = get_int_field(t, "size_bytes", "nbytes", default=0)

        dt_raw = t.get("dtype")
        dt = map_dtype(dt_raw)

        if args.strict:
            if off <= 0 or sz <= 0:
                missing_fields_count += 1
            if isinstance(dt_raw, str) and dt_raw.strip().lower() not in ALLOWED_DTYPES:
                unknown_dtype_count += 1
            if isinstance(dt_raw, int) and dt_raw not in (2, 3, 4):
                unknown_dtype_count += 1

        out_obj: Dict[str, Any] = {
            "name": name,
            "dtype": dt,
            "offset_bytes": off,
            "size_bytes": sz,
        }

        shape = t.get("shape")
        if isinstance(shape, list) and all(isinstance(x, int) for x in shape):
            out_obj["shape"] = shape

        tensors_out.append(out_obj)

    payload = {"tensors": tensors_out}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"wrote {out_path} tensors={len(tensors_out)}")
    if args.strict:
        if missing_fields_count or unknown_dtype_count:
            print(
                f"strict: missing_fields={missing_fields_count} unknown_dtype={unknown_dtype_count}",
                file=sys.stderr,
            )
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
