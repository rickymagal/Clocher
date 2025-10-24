#!/usr/bin/env python3
"""
Export selected initializers from an ONNX graph to row-major float32 .bin files.

Usage:

  # List all initializer names and shapes
  python3 scripts/export_tensors_onnx.py --onnx model.onnx --list

  # Export by name
  python3 scripts/export_tensors_onnx.py \
    --onnx model.onnx \
    --tensor "rnn/weight_hh_l0:bin/Wxh.bin" \
    --tensor "classifier/weight:bin/Woh.bin" \
    --transpose Woh  # if you need to transpose that 2D tensor before saving
"""

import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to ONNX model.")
    ap.add_argument("--list", action="store_true", help="List initializer names and shapes, then exit.")
    ap.add_argument("--tensor", action="append", default=[], help='Mapping "initializer_name:out_path.bin". Repeatable.')
    ap.add_argument("--alias", action="append", default=[], help='Optional alias for each --tensor (same order).')
    ap.add_argument("--transpose", action="append", default=[], help="Alias names to transpose before saving.")
    args = ap.parse_args()

    try:
        import onnx
        from onnx import numpy_helper
    except Exception as e:
        print("ERROR: This exporter requires onnx. Please `pip install onnx`.", file=sys.stderr)
        sys.exit(2)

    model = onnx.load(args.onnx)
    inits = {init.name: init for init in model.graph.initializer}

    if args.list:
        print("# Initializers:")
        for name, init in inits.items():
            arr = numpy_helper.to_array(init)
            print(f"{name}: {arr.shape}")
        return

    if not args.tensor:
        print("ERROR: No --tensor mappings given. Use --list first.", file=sys.stderr)
        sys.exit(2)

    aliases = list(args.alias)
    while len(aliases) < len(args.tensor):
        aliases.append(f"T{len(aliases)}")

    transpose_set = set(args.transpose or [])

    for idx, mapping in enumerate(args.tensor):
        if ":" not in mapping:
            print(f"ERROR: invalid --tensor mapping '{mapping}'. Expected 'name:out.bin'", file=sys.stderr)
            sys.exit(2)
        name, out_path = mapping.split(":", 1)
        if name not in inits:
            print(f"ERROR: initializer '{name}' not found.", file=sys.stderr)
            sys.exit(3)

        arr = numpy_helper.to_array(inits[name]).astype("float32", copy=False)
        alias = aliases[idx]

        if alias in transpose_set:
            if arr.ndim != 2:
                print(f"ERROR: --transpose {alias} requested, but '{name}' is not 2D.", file=sys.stderr)
                sys.exit(3)
            arr = arr.T.copy()

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(arr.astype("float32", copy=False).tobytes())
        print(f"[ok] wrote {out_path} shape={tuple(arr.shape)} row_major_f32")

if __name__ == "__main__":
    main()
