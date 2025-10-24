# =============================================================================
# File: scripts/ptq_from_bin.py
# =============================================================================
#!/usr/bin/env python3
"""
PTQ from an existing row-major FP32 weight .bin.

Given a flat float32 binary file and its shape, run weight-only PTQ and emit
artifacts compatible with the engine's IEBIN v1 JSON.

Example:
  python3 scripts/ptq_from_bin.py \
      --weights out/qproj.f32.bin --rows 4096 --cols 4096 \
      --out-prefix out/qproj_q4 --wbits 4 --mode per_row --scale-dtype fp16
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Input FP32 .bin (row-major)")
    ap.add_argument("--rows", type=int, required=True, help="Rows of the matrix")
    ap.add_argument("--cols", type=int, required=True, help="Cols of the matrix")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for PTQ artifacts")
    ap.add_argument("--mode", default="per_row", choices=["per_row", "per_tensor"])
    ap.add_argument("--accuracy-threshold", type=float, default=0.995)
    ap.add_argument("--wbits", type=int, default=8, choices=[8, 4],
                    help="Weight bit-width: 8 (default) or 4")
    ap.add_argument("--scale-dtype", default="fp16", choices=["fp16", "fp32"],
                    help="Scale storage dtype (default: fp16)")
    args = ap.parse_args()

    # Simple validation
    if args.rows <= 0 or args.cols <= 0:
        print("ERROR: invalid shape", file=sys.stderr)
        sys.exit(2)

    # Invoke the common calibrator
    cmd = [
        sys.executable, "benchmarks/ptq_calib.py",
        "--weights", args.weights,
        "--rows", str(args.rows),
        "--cols", str(args.cols),
        "--mode", args.mode,
        "--out-prefix", args.out_prefix,
        "--accuracy-threshold", str(args.accuracy_threshold),
        "--wbits", str(args.wbits),
        "--scale-dtype", args.scale_dtype,
    ]
    print("[bin-shape]", json.dumps({
        "weights": args.weights,
        "shape": [args.rows, args.cols],
        "mode": args.mode,
        "wbits": args.wbits,
        "scale_dtype": args.scale_dtype,
        "out_prefix": args.out_prefix,
    }))
    cp = subprocess.run(cmd)
    sys.exit(cp.returncode)


if __name__ == "__main__":
    main()
