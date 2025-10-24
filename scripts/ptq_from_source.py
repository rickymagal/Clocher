# =============================================================================
# File: scripts/ptq_from_source.py
# =============================================================================
#!/usr/bin/env python3
"""
One-shot PTQ from source model: reads TRUE shape from PyTorch or ONNX,
exports the tensor as row-major float32 .bin, then runs weight-only PTQ
with the exact rows/cols (no guessing).

Usage examples:

  # PyTorch (state_dict key):
  python3 scripts/ptq_from_source.py \
    --source torch \
    --checkpoint /path/to/model.ckpt \
    --key rnn.weight_hh_l0 \
    --out-prefix out/Wxh_q4 \
    --wbits 4 --mode per_row --scale-dtype fp16

  # ONNX (initializer name):
  python3 scripts/ptq_from_source.py \
    --source onnx \
    --onnx /path/to/model.onnx \
    --init RNN/weight_hh_l0 \
    --out-prefix out/Wxh_q8 \
    --wbits 8 --mode per_tensor --scale-dtype fp32

Optional:
  --transpose
  --mode per_row|per_tensor
  --wbits 8|4
  --scale-dtype fp16|fp32
  --accuracy-threshold 0.995
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def export_torch(checkpoint: str, key: str, out_bin: str, do_transpose: bool):
    """
    Export a 2D tensor from a PyTorch checkpoint to row-major float32 .bin.

    Returns
    -------
    tuple(int, int)
        The tensor shape as (rows, cols).
    """
    try:
        import torch
    except Exception:
        print("ERROR: torch is required. pip install torch", file=sys.stderr)
        sys.exit(2)

    ckpt = torch.load(checkpoint, map_location="cpu")
    if hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        print("ERROR: unsupported checkpoint format", file=sys.stderr)
        sys.exit(2)

    if key not in sd:
        print(f"ERROR: key '{key}' not found in state_dict", file=sys.stderr)
        sys.exit(3)

    t = sd[key]
    if not hasattr(t, "shape"):
        print(f"ERROR: key '{key}' is not tensor-like", file=sys.stderr)
        sys.exit(3)

    ten = t.detach().cpu().float()
    if ten.ndim != 2:
        print(f"ERROR: selected tensor is {ten.ndim}D; expected 2D [rows, cols]", file=sys.stderr)
        sys.exit(3)

    if do_transpose:
        ten = ten.t().contiguous()

    rows, cols = int(ten.shape[0]), int(ten.shape[1])
    os.makedirs(os.path.dirname(out_bin) or ".", exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(ten.numpy().astype("float32").tobytes())
    return rows, cols


def export_onnx(onnx_path: str, init_name: str, out_bin: str, do_transpose: bool):
    """
    Export a 2D initializer from an ONNX model to row-major float32 .bin.

    Returns
    -------
    tuple(int, int)
        The tensor shape as (rows, cols).
    """
    try:
        import onnx
        from onnx import numpy_helper
    except Exception:
        print("ERROR: onnx is required. pip install onnx", file=sys.stderr)
        sys.exit(2)

    model = onnx.load(onnx_path)
    inits = {init.name: init for init in model.graph.initializer}
    if init_name not in inits:
        print(f"ERROR: initializer '{init_name}' not found", file=sys.stderr)
        sys.exit(3)

    arr = numpy_helper.to_array(inits[init_name]).astype("float32", copy=False)
    if arr.ndim != 2:
        print(f"ERROR: selected initializer is {arr.ndim}D; expected 2D [rows, cols]", file=sys.stderr)
        sys.exit(3)

    if do_transpose:
        arr = arr.T.copy()

    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    os.makedirs(os.path.dirname(out_bin) or ".", exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(arr.astype("float32", copy=False).tobytes())
    return rows, cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["torch", "onnx"])
    ap.add_argument("--checkpoint", help="PyTorch checkpoint (.pt/.pth/.ckpt)")
    ap.add_argument("--key", help="State_dict key to export (PyTorch)")
    ap.add_argument("--onnx", help="ONNX model path")
    ap.add_argument("--init", help="Initializer name to export (ONNX)")
    ap.add_argument("--out-prefix", required=True, help="Output prefix for PTQ artifacts")
    ap.add_argument("--transpose", action="store_true", help="Transpose 2D tensor before saving")
    ap.add_argument("--mode", default="per_row", choices=["per_row", "per_tensor"])
    ap.add_argument("--accuracy-threshold", type=float, default=0.995)
    ap.add_argument("--wbits", type=int, default=8, choices=[8, 4],
                    help="Weight bit-width: 8 (default) or 4")
    ap.add_argument("--scale-dtype", default="fp16", choices=["fp16", "fp32"],
                    help="Scale storage dtype (default: fp16)")
    args = ap.parse_args()

    out_bin = args.out_prefix + ".f32.bin"

    if args.source == "torch":
        if not args.checkpoint or not args.key:
            print("ERROR: --checkpoint and --key are required for source=torch", file=sys.stderr)
            sys.exit(2)
        rows, cols = export_torch(args.checkpoint, args.key, out_bin, args.transpose)
    else:
        if not args.onnx or not args.init:
            print("ERROR: --onnx and --init are required for source=onnx", file=sys.stderr)
            sys.exit(2)
        rows, cols = export_onnx(args.onnx, args.init, out_bin, args.transpose)

    if rows <= 0 or cols <= 0:
        print("ERROR: invalid shape", file=sys.stderr)
        sys.exit(3)

    meta = {
        "exported_fp32_bin": out_bin,
        "shape": [rows, cols],
        "mode": args.mode,
        "out_prefix": args.out_prefix,
        "accuracy_threshold": args.accuracy_threshold,
        "wbits": args.wbits,
        "scale_dtype": args.scale_dtype,
    }
    print("[shape]", json.dumps(meta))

    # Run the calibrator with TRUE rows/cols
    cmd = [
        sys.executable, "benchmarks/ptq_calib.py",
        "--weights", out_bin,
        "--rows", str(rows),
        "--cols", str(cols),
        "--mode", args.mode,
        "--out-prefix", args.out_prefix,
        "--accuracy-threshold", str(args.accuracy_threshold),
        "--wbits", str(args.wbits),
        "--scale-dtype", args.scale_dtype,
    ]
    cp = subprocess.run(cmd, text=True)
    sys.exit(cp.returncode)


if __name__ == "__main__":
    main()
