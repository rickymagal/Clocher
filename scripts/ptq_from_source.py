#!/usr/bin/env python3
"""
One-shot PTQ from source model: reads TRUE shape from PyTorch or ONNX,
exports the tensor as row-major float32 .bin, then runs INT8 PTQ with the
exact rows/cols (no guessing).

Usage examples:

  # PyTorch (state_dict key):
  python3 scripts/ptq_from_source.py \
    --source torch \
    --checkpoint /path/to/model.ckpt \
    --key rnn.weight_hh_l0 \
    --out-prefix out/Wxh_int8

  # ONNX (initializer name):
  python3 scripts/ptq_from_source.py \
    --source onnx \
    --onnx /path/to/model.onnx \
    --init RNN/weight_hh_l0 \
    --out-prefix out/Wxh_int8

Optional:
  --transpose            # if your tensor must be transposed before saving
  --mode per_row         # PTQ scale mode (per_row|per_tensor), default per_row
  --accuracy-threshold 0.995
"""

import argparse
import json
import os
import subprocess
import sys

def export_torch(checkpoint, key, out_bin, do_transpose):
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
    shape = tuple(int(x) for x in ten.shape)

    if do_transpose:
        if ten.ndim != 2:
            print(f"ERROR: --transpose requested but '{key}' is not 2D", file=sys.stderr)
            sys.exit(3)
        ten = ten.t().contiguous()
        shape = tuple(int(x) for x in ten.shape)

    os.makedirs(os.path.dirname(out_bin) or ".", exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(ten.numpy().astype("float32").tobytes())

    return shape  # rows, cols, ...

def export_onnx(onnx_path, init_name, out_bin, do_transpose):
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
    shape = tuple(int(x) for x in arr.shape)

    if do_transpose:
        if arr.ndim != 2:
            print(f"ERROR: --transpose requested but '{init_name}' is not 2D", file=sys.stderr)
            sys.exit(3)
        arr = arr.T.copy()
        shape = tuple(int(x) for x in arr.shape)

    os.makedirs(os.path.dirname(out_bin) or ".", exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(arr.astype("float32", copy=False).tobytes())

    return shape

def main():
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
    args = ap.parse_args()

    out_bin = args.out_prefix + ".f32.bin"

    if args.source == "torch":
        if not args.checkpoint or not args.key:
            print("ERROR: --checkpoint and --key are required for source=torch", file=sys.stderr)
            sys.exit(2)
        shape = export_torch(args.checkpoint, args.key, out_bin, args.transpose)
    else:
        if not args.onnx or not args.init:
            print("ERROR: --onnx and --init are required for source=onnx", file=sys.stderr)
            sys.exit(2)
        shape = export_onnx(args.onnx, args.init, out_bin, args.transpose)

    # Expect 2D weight for GEMV/PTQ. If not 2D, refuse.
    if len(shape) != 2:
        print(f"ERROR: exported tensor has shape {shape}, expected 2D [rows, cols]", file=sys.stderr)
        sys.exit(3)

    rows, cols = int(shape[0]), int(shape[1])

    meta = {
        "exported_fp32_bin": out_bin,
        "shape": [rows, cols],
        "mode": args.mode,
        "out_prefix": args.out_prefix,
        "accuracy_threshold": args.accuracy_threshold,
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
    ]
    cp = subprocess.run(cmd, text=True)
    sys.exit(cp.returncode)

if __name__ == "__main__":
    main()
