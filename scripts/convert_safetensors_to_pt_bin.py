#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)

    sd = safe_load_file(str(inp), device="cpu")
    torch.save(sd, str(outp))
    print(f"[ok] wrote {outp} ({inp.name} -> pytorch bin)")

if __name__ == "__main__":
    main()
