#!/usr/bin/env python3
import json, sys, pathlib

MAP = {
    # Floating-point family
    "F64": 0,
    "F32": 3,
    "BF16": 2,
    "F16": 1,
    "FP8": 6,
    "FP4": 7,

    # Integer / quantized family
    "INT8": 4,
    "INT4": 5,
    "UINT8": 8,
    "UINT4": 9,

    # Unsigned exponent quantization formats (used for scales)
    "UE8": 10,
    "UE4": 11,
    "U8": 12,
    "U4": 13,
}

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
j = json.load(src.open())
out = {}

for k, v in j.items():
    if isinstance(v, str):
        code = MAP.get(v.upper())
        if code is None:
            raise SystemExit(f"ERROR: unknown dtype string '{v}' for tensor '{k}'")
        out[k] = code
    elif isinstance(v, (int, float)):
        out[k] = int(v)
    else:
        raise SystemExit(f"ERROR: unexpected dtype value {v!r} for '{k}'")

dst.write_text(json.dumps(out, indent=2))
print(f"OK: wrote numeric dtypes -> {dst} ({len(out)} entries)")
