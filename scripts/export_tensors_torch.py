#!/usr/bin/env python3
"""
Export selected tensors from a PyTorch checkpoint to row-major float32 .bin files.

Usage examples:

  # List all tensor names in a checkpoint
  python3 scripts/export_tensors_torch.py --checkpoint model.ckpt --list

  # Export specific tensors by state_dict key
  python3 scripts/export_tensors_torch.py \
    --checkpoint model.ckpt \
    --tensor "rnn.weight_hh_l0:bin/Wxh.bin" \
    --tensor "classifier.weight:bin/Woh.bin" \
    --transpose Woh  # if classifier.weight is [V, H] and you need [H, V], don't use --transpose
                     # if it's [H, V] and you need [V, H], add its alias in --transpose

Notes:
- Tensors are written in **row-major float32**; we do not change layout unless you request
  a transpose via --transpose <alias>.
- Use --alias to give a short name to each export so you can target it in --transpose.
- You can repeat --tensor and --alias multiple times; order matters: alias i corresponds to tensor i.
"""

import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt/.pth/.ckpt) containing a state_dict or a full object with .state_dict().")
    ap.add_argument("--list", action="store_true", help="Only list tensor names and shapes, then exit.")
    ap.add_argument("--tensor", action="append", default=[],
                    help='Mapping "state_dict_key:out_path.bin". Repeatable.')
    ap.add_argument("--alias", action="append", default=[],
                    help='Optional alias for each --tensor (e.g., Wxh, Woh). Repeatable in the same order as --tensor.')
    ap.add_argument("--transpose", action="append", default=[],
                    help="Alias names to transpose before saving (e.g., --transpose Woh).")
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print("ERROR: This exporter requires torch. Please `pip install torch`.", file=sys.stderr)
        sys.exit(2)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        print("ERROR: Unsupported checkpoint format.", file=sys.stderr)
        sys.exit(2)

    if args.list:
        print("# Available tensors in checkpoint:")
        for k, v in sd.items():
            try:
                shape = tuple(v.shape)
            except Exception:
                shape = "N/A"
            print(f"{k}: {shape}")
        return

    if not args.tensor:
        print("ERROR: No --tensor mappings given. Use --list to discover names.", file=sys.stderr)
        sys.exit(2)

    # Normalize alias list length to tensor count
    aliases = list(args.alias)
    while len(aliases) < len(args.tensor):
        aliases.append(f"T{len(aliases)}")  # default alias T0, T1, ...

    transpose_set = set(args.transpose or [])

    for idx, mapping in enumerate(args.tensor):
        if ":" not in mapping:
            print(f"ERROR: invalid --tensor mapping '{mapping}'. Expected 'key:out.bin'", file=sys.stderr)
            sys.exit(2)
        key, out_path = mapping.split(":", 1)
        if key not in sd:
            print(f"ERROR: key '{key}' not found in state_dict.", file=sys.stderr)
            sys.exit(3)

        ten = sd[key]
        if not hasattr(ten, "shape"):
            print(f"ERROR: key '{key}' is not a tensor-like object.", file=sys.stderr)
            sys.exit(3)

        alias = aliases[idx]
        t = ten.detach().cpu().float()

        if alias in transpose_set:
            if t.ndim != 2:
                print(f"ERROR: --transpose {alias} requested, but '{key}' is not 2D.", file=sys.stderr)
                sys.exit(3)
            t = t.t().contiguous()

        # Write row-major float32
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(t.numpy().astype("float32").tobytes())
        print(f"[ok] wrote {out_path} shape={tuple(t.shape)} row_major_f32")

if __name__ == "__main__":
    main()
